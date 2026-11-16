"""Configuration structures, loading, and validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when configuration would change or ambiguously define the method."""


@dataclass(frozen=True)
class TopologyConfig:
    sample_count: int
    knn_k: int
    time_weight: float
    point_slots: int
    lifetime_threshold: float
    grid_size: int
    bandwidth: float
    descriptor_dim: int


@dataclass(frozen=True)
class RepresentationConfig:
    codebook_size: int
    embedding_dim: int
    plan_stride: int
    plan_width: int
    realization_dim: int
    continuous_width: int
    dropout: float
    group_norm_groups: int
    codebook_decay: float
    codebook_smoothing: float
    dead_code_start: int
    dead_code_interval: int
    dead_code_threshold: float


@dataclass(frozen=True)
class DecoderConfig:
    part_count: int
    physical_width: int
    part_bond_dim: int
    joint_bond_dim: int
    identity_dim: int
    frame_width: int
    temporal_dilations: tuple[int, ...]


@dataclass(frozen=True)
class TextConfig:
    backbone: str
    width: int
    part_queries: int
    attention_heads: int
    graph_rounds: int
    point_slots: int
    head_hidden: int


@dataclass(frozen=True)
class PlannerConfig:
    layers: int
    width: int
    heads: int
    ffn: int
    dropout: float
    max_positions: int
    modulation_hidden: int
    modulation_alpha: float
    modulation_clip: float


@dataclass(frozen=True)
class FlowConfig:
    input_dim: int
    width: int
    blocks: int
    heads: int
    ffn: int
    dropout: float
    condition_dropout: float


@dataclass(frozen=True)
class SamplingConfig:
    ar_temperature: float
    top_k: int | None
    top_p: float | None
    euler_steps: int
    guidance_scale: float


@dataclass(frozen=True)
class StageConfig:
    steps: int
    batch_size: int


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float
    beta1: float
    beta2: float
    epsilon: float
    weight_decay: float
    warmup_steps: int
    decay_fractions: tuple[float, ...]
    decay_factor: float
    gradient_clip: float


@dataclass(frozen=True)
class TrainingConfig:
    stage_a: StageConfig
    stage_b1: StageConfig
    stage_b2: StageConfig
    optimizer: OptimizerConfig
    validate_every: int
    precision: str
    rollin_clean_steps: int
    rollin_proposal_probability: float
    rollin_acceptance_factor: float
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class RuntimeConfig:
    device: str


@dataclass(frozen=True)
class QuanTopoConfig:
    topology: TopologyConfig
    representation: RepresentationConfig
    decoder: DecoderConfig
    text: TextConfig
    planner: PlannerConfig
    flow: FlowConfig
    sampling: SamplingConfig
    training: TrainingConfig
    runtime: RuntimeConfig


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: str
    joint_count: int
    ric_width: int
    frame_rate: float
    mean_path: str
    std_path: str
    offsets_path: str


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a mapping")
    return value


def _strict(data: Mapping[str, Any], allowed: Iterable[str], path: str) -> None:
    allowed_set = set(allowed)
    unknown = sorted(set(data) - allowed_set)
    if unknown:
        prefix = f"{path}." if path else ""
        raise ConfigError(f"unknown configuration key: {prefix}{unknown[0]}")
    missing = sorted(allowed_set - set(data))
    if missing:
        prefix = f"{path}." if path else ""
        raise ConfigError(f"missing configuration key: {prefix}{missing[0]}")


def _section(raw: Mapping[str, Any], name: str, keys: Sequence[str]) -> Mapping[str, Any]:
    data = _mapping(raw.get(name), name)
    _strict(data, keys, name)
    return data


def _positive_config_values(config: QuanTopoConfig) -> None:
    groups = (
        config.topology,
        config.representation,
        config.decoder,
        config.text,
        config.planner,
        config.flow,
    )
    for group in groups:
        for key, value in vars(group).items():
            if isinstance(value, int | float) and not isinstance(value, bool) and value <= 0:
                if key in {"dropout", "condition_dropout"} and value == 0:
                    continue
                raise ConfigError(f"{type(group).__name__}.{key} must be positive")


def _validate(config: QuanTopoConfig) -> None:
    _positive_config_values(config)
    expected_descriptor = 2 * config.topology.grid_size**2
    if config.topology.descriptor_dim != expected_descriptor:
        raise ConfigError(
            "descriptor_dim must equal 2 * grid_size^2 "
            f"({expected_descriptor}), got {config.topology.descriptor_dim}"
        )
    if config.topology.descriptor_dim != 128:
        raise ConfigError("descriptor_dim must be 128")
    if config.representation.codebook_size != 512:
        raise ConfigError("codebook_size must be 512")
    if config.representation.embedding_dim != config.representation.plan_width:
        raise ConfigError("embedding_dim must equal plan_width")
    if config.flow.input_dim != config.representation.realization_dim:
        raise ConfigError("flow.input_dim must equal representation.realization_dim")
    if config.decoder.part_count != config.text.part_queries or config.decoder.part_count != 6:
        raise ConfigError("decoder.part_count and text.part_queries must both equal six")
    if config.text.point_slots != config.topology.point_slots:
        raise ConfigError("text.point_slots must equal topology.point_slots")


_OVERRIDABLE = {
    "training.stage_a.steps",
    "training.stage_a.batch_size",
    "training.stage_b1.steps",
    "training.stage_b1.batch_size",
    "training.stage_b2.steps",
    "training.stage_b2.batch_size",
    "runtime.device",
}


def _apply_overrides(raw: MutableMapping[str, Any], overrides: Sequence[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise ConfigError(f"override must be key=value: {override}")
        key, encoded = override.split("=", 1)
        if key not in _OVERRIDABLE:
            raise ConfigError(f"override is not permitted: {key}")
        cursor: MutableMapping[str, Any] = raw
        pieces = key.split(".")
        for piece in pieces[:-1]:
            child = cursor.get(piece)
            if not isinstance(child, MutableMapping):
                raise ConfigError(f"override path does not exist: {key}")
            cursor = child
        cursor[pieces[-1]] = yaml.safe_load(encoded)


def load_config(path: Path, overrides: Sequence[str] = ()) -> QuanTopoConfig:
    raw_value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    raw = dict(_mapping(raw_value, "root"))
    _strict(
        raw,
        [
            "topology",
            "representation",
            "decoder",
            "text",
            "planner",
            "flow",
            "sampling",
            "training",
            "runtime",
        ],
        "",
    )
    _apply_overrides(raw, overrides)

    topology = _section(
        raw,
        "topology",
        [
            "sample_count",
            "knn_k",
            "time_weight",
            "point_slots",
            "lifetime_threshold",
            "grid_size",
            "bandwidth",
            "descriptor_dim",
        ],
    )
    representation = _section(
        raw,
        "representation",
        [
            "codebook_size",
            "embedding_dim",
            "plan_stride",
            "plan_width",
            "realization_dim",
            "continuous_width",
            "dropout",
            "group_norm_groups",
            "codebook_decay",
            "codebook_smoothing",
            "dead_code_start",
            "dead_code_interval",
            "dead_code_threshold",
        ],
    )
    decoder = _section(
        raw,
        "decoder",
        [
            "part_count",
            "physical_width",
            "part_bond_dim",
            "joint_bond_dim",
            "identity_dim",
            "frame_width",
            "temporal_dilations",
        ],
    )
    text = _section(
        raw,
        "text",
        [
            "backbone",
            "width",
            "part_queries",
            "attention_heads",
            "graph_rounds",
            "point_slots",
            "head_hidden",
        ],
    )
    planner = _section(
        raw,
        "planner",
        [
            "layers",
            "width",
            "heads",
            "ffn",
            "dropout",
            "max_positions",
            "modulation_hidden",
            "modulation_alpha",
            "modulation_clip",
        ],
    )
    flow = _section(
        raw,
        "flow",
        ["input_dim", "width", "blocks", "heads", "ffn", "dropout", "condition_dropout"],
    )
    sampling = _section(
        raw,
        "sampling",
        ["ar_temperature", "top_k", "top_p", "euler_steps", "guidance_scale"],
    )
    training = _section(
        raw,
        "training",
        [
            "stage_a",
            "stage_b1",
            "stage_b2",
            "optimizer",
            "validate_every",
            "precision",
            "rollin_clean_steps",
            "rollin_proposal_probability",
            "rollin_acceptance_factor",
            "seeds",
        ],
    )
    runtime = _section(raw, "runtime", ["device"])

    stage_keys = ["steps", "batch_size"]
    optimizer_keys = [
        "learning_rate",
        "beta1",
        "beta2",
        "epsilon",
        "weight_decay",
        "warmup_steps",
        "decay_fractions",
        "decay_factor",
        "gradient_clip",
    ]
    stage_a_raw = _mapping(training["stage_a"], "training.stage_a")
    stage_b1_raw = _mapping(training["stage_b1"], "training.stage_b1")
    stage_b2_raw = _mapping(training["stage_b2"], "training.stage_b2")
    optimizer_raw = _mapping(training["optimizer"], "training.optimizer")
    _strict(stage_a_raw, stage_keys, "training.stage_a")
    _strict(stage_b1_raw, stage_keys, "training.stage_b1")
    _strict(stage_b2_raw, stage_keys, "training.stage_b2")
    _strict(optimizer_raw, optimizer_keys, "training.optimizer")

    config = QuanTopoConfig(
        topology=TopologyConfig(**topology),
        representation=RepresentationConfig(**representation),
        decoder=DecoderConfig(
            **{**decoder, "temporal_dilations": tuple(decoder["temporal_dilations"])}
        ),
        text=TextConfig(**text),
        planner=PlannerConfig(**planner),
        flow=FlowConfig(**flow),
        sampling=SamplingConfig(**sampling),
        training=TrainingConfig(
            stage_a=StageConfig(**stage_a_raw),
            stage_b1=StageConfig(**stage_b1_raw),
            stage_b2=StageConfig(**stage_b2_raw),
            optimizer=OptimizerConfig(
                **{
                    **optimizer_raw,
                    "decay_fractions": tuple(optimizer_raw["decay_fractions"]),
                }
            ),
            validate_every=training["validate_every"],
            precision=training["precision"],
            rollin_clean_steps=training["rollin_clean_steps"],
            rollin_proposal_probability=training["rollin_proposal_probability"],
            rollin_acceptance_factor=training["rollin_acceptance_factor"],
            seeds=tuple(training["seeds"]),
        ),
        runtime=RuntimeConfig(**runtime),
    )
    _validate(config)
    return config


def load_dataset_config(path: Path) -> DatasetConfig:
    raw = _mapping(yaml.safe_load(Path(path).read_text(encoding="utf-8")), "dataset")
    keys = [
        "name",
        "root",
        "joint_count",
        "ric_width",
        "frame_rate",
        "mean_path",
        "std_path",
        "offsets_path",
    ]
    _strict(raw, keys, "dataset")
    for key in ("root", "mean_path", "std_path", "offsets_path"):
        candidate = Path(str(raw[key]))
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ConfigError(
                f"dataset.{key} must be a project-relative path without parent traversal"
            )
    config = DatasetConfig(**raw)
    expected = {
        "humanml3d": (22, 263, 20.0),
        "kitml": (21, 251, 12.5),
    }
    if config.name not in expected:
        raise ConfigError(f"unsupported dataset: {config.name}")
    if (config.joint_count, config.ric_width, config.frame_rate) != expected[config.name]:
        raise ConfigError(f"dataset dimensions/frame rate do not match {config.name}")
    return config
