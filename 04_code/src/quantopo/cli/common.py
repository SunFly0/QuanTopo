"""Configuration, path handling, and model construction for command-line tools."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from quantopo.ablations.decoders import DeterministicRealizationRegressor, build_decoder_control
from quantopo.ablations.quantizers import build_quantizer
from quantopo.ablations.readouts import build_plan_readout
from quantopo.contracts.config import (
    DatasetConfig,
    QuanTopoConfig,
    load_config,
    load_dataset_config,
)
from quantopo.data.assets import AssetManifest, validate_assets
from quantopo.data.specs import get_dataset_spec
from quantopo.flow.normalization import RealizationTransform
from quantopo.flow.objective import RollInGate
from quantopo.flow.transformer import ConditionalFlow
from quantopo.models.factory import MethodComponents, default_components
from quantopo.models.planning import StageB1Model
from quantopo.models.quantopo import StageB2Model
from quantopo.models.representation import StageAModel
from quantopo.planner.autoregressive import AutoregressivePlanner
from quantopo.representation.encoders import PlanEncoder, RealizationEncoder
from quantopo.representation.quantizer import EMAQuantizer
from quantopo.text.clip import FrozenCLIPTextEncoder
from quantopo.text.topology_predictor import TextTopologyPredictor
from quantopo.topology.readout import PersistenceReadout
from quantopo.training.statistics_artifact import (
    StatisticsBindings,
    load_realization_statistics,
)


@dataclass(frozen=True)
class CLIContext:
    project_root: Path
    model_path: Path
    data_path: Path
    output_dir: Path
    model: QuanTopoConfig
    dataset: DatasetConfig
    assets: AssetManifest


def project_relative_path(
    project_root: Path, encoded: str | Path, *, label: str, must_exist: bool = False
) -> Path:
    relative = Path(encoded)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a project-relative path without parent traversal")
    root = project_root.resolve()
    resolved = (root / relative).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} must remain under the project root")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"missing {label}: {relative.as_posix()}")
    return resolved


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-config", default="configs/model/quantopo.yaml")
    parser.add_argument("--data-config", default="configs/data/humanml3d.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="runs/quantopo")
    parser.add_argument("--split", default="train")
    parser.add_argument("--dry-run", action="store_true")


def prepare_context(arguments: argparse.Namespace) -> CLIContext:
    root = Path(arguments.project_root).resolve()
    model_path = project_relative_path(
        root, arguments.model_config, label="model config", must_exist=True
    )
    data_path = project_relative_path(
        root, arguments.data_config, label="data config", must_exist=True
    )
    output_dir = project_relative_path(root, arguments.output_dir, label="output directory")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("output directory points to an existing file")
    overrides = list(arguments.override)
    if arguments.device is not None:
        overrides.append(f"runtime.device={arguments.device}")
    model = load_config(model_path, overrides)
    dataset = load_dataset_config(data_path)
    assets = validate_assets(dataset, split=arguments.split, project_root=root)
    output_dir.mkdir(parents=True, exist_ok=True)
    return CLIContext(root, model_path, data_path, output_dir, model, dataset, assets)


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def run_report(context: CLIContext, *, stage: str, dry_run: bool) -> dict[str, Any]:
    return {
        "stage": stage,
        "dry_run": dry_run,
        "model": asdict(context.model),
        "dataset": asdict(context.dataset),
        "assets": {
            "root": _relative(context.project_root, context.assets.root),
            "split": context.assets.split,
            "motion_count": len(context.assets.motion_ids),
            "mean": _relative(context.project_root, context.assets.mean_path),
            "std": _relative(context.project_root, context.assets.std_path),
            "offsets": _relative(context.project_root, context.assets.offsets_path),
        },
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _combined_digest(*values: str) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


def build_stage_a(context: CLIContext, components: MethodComponents | None = None) -> StageAModel:
    components = default_components() if components is None else components
    config = context.model
    representation = config.representation
    decoder = config.decoder
    offsets = torch.from_numpy(np.load(context.assets.offsets_path).astype(np.float32))
    mean = torch.from_numpy(np.load(context.assets.mean_path).astype(np.float32))
    std = torch.from_numpy(np.load(context.assets.std_path).astype(np.float32))
    return StageAModel(
        PlanEncoder(
            context.dataset.ric_width,
            width=representation.plan_width,
            groups=representation.group_norm_groups,
            dropout=representation.dropout,
        ),
        (
            EMAQuantizer(
                representation.codebook_size,
                representation.embedding_dim,
                decay=representation.codebook_decay,
                smoothing=representation.codebook_smoothing,
                dead_code_start=representation.dead_code_start,
                dead_code_interval=representation.dead_code_interval,
                dead_code_threshold=representation.dead_code_threshold,
            )
            if components.quantizer == "ema_vq"
            else build_quantizer(
                components.quantizer,
                codebook_size=representation.codebook_size,
                embedding_dim=representation.embedding_dim,
            )
        ),
        RealizationEncoder(
            context.dataset.ric_width,
            plan_dim=representation.embedding_dim,
            width=representation.continuous_width,
            groups=representation.group_norm_groups,
            dropout=representation.dropout,
            output_dim=representation.realization_dim,
        ),
        build_decoder_control(
            components.decoder,
            get_dataset_spec(context.dataset.name),
            offsets,
            plan_dim=representation.embedding_dim,
            realization_dim=representation.realization_dim,
            physical_dim=decoder.physical_width,
            part_bond_dim=decoder.part_bond_dim,
            joint_bond_dim=decoder.joint_bond_dim,
            identity_dim=decoder.identity_dim,
            frame_width=decoder.frame_width,
            groups=representation.group_norm_groups,
            dropout=representation.dropout,
        ),
        build_plan_readout(
            components.plan_readout,
            plan_dim=representation.embedding_dim,
            descriptor_dim=config.topology.descriptor_dim,
        ),
        ric_mean=mean,
        ric_std=std,
        loss_coefficients=components.stage_a_loss,
    )


def build_stage_b1(
    context: CLIContext,
    representation: StageAModel,
    *,
    clip_checkpoint: str | Path,
    descriptor_scale: torch.Tensor | None = None,
    components: MethodComponents | None = None,
    text_encoder: FrozenCLIPTextEncoder | None = None,
) -> StageB1Model:
    components = default_components() if components is None else components
    config = context.model
    text = config.text
    planner = config.planner
    readout = PersistenceReadout(config.topology.grid_size, config.topology.bandwidth)
    return StageB1Model(
        representation=representation,
        text_encoder=(
            FrozenCLIPTextEncoder.from_open_clip_checkpoint(
                clip_checkpoint, project_root=context.project_root
            )
            if text_encoder is None
            else text_encoder
        ),
        topology_predictor=TextTopologyPredictor(
            text_width=text.width,
            query_count=text.part_queries,
            attention_heads=text.attention_heads,
            graph_rounds=text.graph_rounds,
            point_slots=text.point_slots,
            head_hidden=text.head_hidden,
            readout=readout,
            length_conditioned=components.topology_length_conditioning,
        ),
        planner=AutoregressivePlanner(
            codebook_size=config.representation.codebook_size,
            text_width=text.width,
            descriptor_dim=config.topology.descriptor_dim,
            layers=planner.layers,
            width=planner.width,
            heads=planner.heads,
            ffn=planner.ffn,
            dropout=planner.dropout,
            max_positions=planner.max_positions,
            modulation_hidden=planner.modulation_hidden,
            modulation_alpha=planner.modulation_alpha,
            modulation_clip=planner.modulation_clip,
            modulation_enabled=components.topology_modulation,
            length_conditioned=components.planner_length_conditioning,
        ),
        loss_coefficients=components.stage_b1_loss,
        descriptor_scale=(
            torch.ones(config.topology.descriptor_dim)
            if descriptor_scale is None
            else descriptor_scale
        ),
    )


def build_flow(context: CLIContext) -> ConditionalFlow:
    config = context.model
    flow = config.flow
    return ConditionalFlow(
        input_dim=flow.input_dim,
        text_width=config.text.width,
        plan_dim=config.representation.embedding_dim,
        width=flow.width,
        blocks=flow.blocks,
        heads=flow.heads,
        ffn=flow.ffn,
        dropout=flow.dropout,
    )


def statistics_bindings(context: CLIContext, stage_a_digest: str) -> StatisticsBindings:
    split_path = context.assets.root / f"{context.assets.split}.txt"
    return StatisticsBindings(
        parent=stage_a_digest,
        split=context.assets.split,
        split_digest=file_digest(split_path),
        architecture=file_digest(context.model_path),
        dataset=file_digest(context.data_path),
        ric_statistics=_combined_digest(
            file_digest(context.assets.mean_path), file_digest(context.assets.std_path)
        ),
        source=package_digest(),
    )


def load_realization_transform(
    context: CLIContext,
    path: str | Path,
    *,
    expected_stage_a_digest: str,
) -> RealizationTransform:
    resolved = project_relative_path(
        context.project_root, path, label="realization statistics", must_exist=True
    )
    return load_realization_statistics(
        resolved, statistics_bindings(context, expected_stage_a_digest)
    )


def build_stage_b2(
    context: CLIContext,
    planning: StageB1Model,
    transform: RealizationTransform,
    components: MethodComponents | None = None,
) -> StageB2Model:
    components = default_components() if components is None else components
    deterministic = components.realization_generator == "deterministic_regression"
    if not deterministic and components.realization_generator != "conditional_flow":
        raise ValueError("unknown realization generator")
    training = context.model.training
    return StageB2Model(
        planning=planning,
        transform=transform,
        flow=None if deterministic else build_flow(context),
        regressor=(
            DeterministicRealizationRegressor(
                plan_dim=context.model.representation.embedding_dim,
                width=context.model.flow.width,
                blocks=context.model.flow.blocks,
                heads=context.model.flow.heads,
                ffn=context.model.flow.ffn,
                dropout=context.model.flow.dropout,
            )
            if deterministic
            else None
        ),
        condition_dropout=context.model.flow.condition_dropout,
        rollin_gate=RollInGate(
            clean_steps=training.rollin_clean_steps,
            proposal_probability=(
                training.rollin_proposal_probability
                if components.rollin_probability is None
                else components.rollin_probability
            ),
            acceptance_factor=training.rollin_acceptance_factor,
        ),
    )


def load_model_checkpoint(
    context: CLIContext,
    encoded_path: str | Path,
    model: torch.nn.Module,
    *,
    expected_stage: str,
    expected_parent: str | None = None,
    expected_topology: str | None = None,
) -> tuple[dict[str, Any], str]:
    path = project_relative_path(
        context.project_root,
        encoded_path,
        label=f"Stage {expected_stage} checkpoint",
        must_exist=True,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    if payload.get("stage") != expected_stage or metadata.get("stage") != expected_stage:
        raise ValueError(f"checkpoint is not a Stage {expected_stage} checkpoint")
    expected_bindings = {
        "architecture": file_digest(context.model_path),
        "dataset": file_digest(context.data_path),
        "skeleton": file_digest(context.assets.offsets_path),
        "statistics": _combined_digest(
            file_digest(context.assets.mean_path), file_digest(context.assets.std_path)
        ),
        "source": package_digest(),
    }
    for name, expected in expected_bindings.items():
        if metadata.get(name) != expected:
            raise ValueError(f"Stage {expected_stage} checkpoint {name} binding does not match")
    if metadata.get("parent") != expected_parent:
        raise ValueError(f"Stage {expected_stage} checkpoint parent binding does not match")
    if expected_topology is not None and metadata.get("topology") != expected_topology:
        raise ValueError(f"Stage {expected_stage} checkpoint topology binding does not match")
    model.load_state_dict(payload["model"], strict=True)
    return payload, file_digest(path)
