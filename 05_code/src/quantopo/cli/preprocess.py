"""Build deterministic training persistence targets from processed physical joints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from quantopo.contracts.config import load_config, load_dataset_config
from quantopo.data.assets import validate_assets
from quantopo.data.dataset import MotionCaptionDataset
from quantopo.topology.cache import TopologyCache
from quantopo.topology.persistence import compute_persistence
from quantopo.topology.readout import PersistenceReadout
from quantopo.topology.states import build_trajectory_states

from .common import project_relative_path


def build_cache(
    model_config: Path,
    data_config: Path,
    split: str,
    output: Path,
    *,
    project_root: Path | None = None,
) -> None:
    root = (Path.cwd() if project_root is None else Path(project_root)).resolve()
    model_path = project_relative_path(root, model_config, label="model config", must_exist=True)
    data_path = project_relative_path(root, data_config, label="data config", must_exist=True)
    output_path = project_relative_path(root, output, label="topology cache")
    model = load_config(model_path)
    data = load_dataset_config(data_path)
    manifest = validate_assets(data, split=split, project_root=root)
    split_path = manifest.root / f"{split}.txt"
    cache = TopologyCache(
        output_path,
        manifest.root,
        split_path,
        model.topology,
        variant_fingerprint="main",
    )
    readout = PersistenceReadout(model.topology.grid_size, model.topology.bandwidth)
    dataset = MotionCaptionDataset(data, manifest, split=split, seed=0)
    for index in range(len(dataset)):
        record = dataset[index]
        source = dataset.source_ric_path(index)
        view = dataset.view_identity(index)
        joints = record.joints.unsqueeze(0)
        mask = torch.ones(1, record.joints.shape[0], dtype=torch.bool)
        sample = build_trajectory_states(joints, mask, model.topology.sample_count)
        valid = sample.mask[0]
        diagram = compute_persistence(
            sample.raw_states[0, valid],
            knn_k=model.topology.knn_k,
            time_weight=model.topology.time_weight,
            point_slots=model.topology.point_slots,
            lifetime_threshold=model.topology.lifetime_threshold,
        )
        descriptor = readout(diagram.points.unsqueeze(0), diagram.mask.unsqueeze(0))[0]
        selected_indices = sample.indices[0, valid] + view.crop_start
        cache.write(
            record.motion_id,
            source,
            selected_indices,
            diagram,
            descriptor,
            view_identity=view,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--data-config", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    try:
        build_cache(
            arguments.model_config,
            arguments.data_config,
            arguments.split,
            arguments.output,
            project_root=arguments.project_root,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
