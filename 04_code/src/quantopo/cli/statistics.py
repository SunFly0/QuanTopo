"""Fit train-only population statistics for normalized realization coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from quantopo.data.collate import collate_motion_records
from quantopo.data.dataset import MotionCaptionDataset
from quantopo.data.normalization import RunningMoments
from quantopo.training.engine import move_to_device
from quantopo.training.statistics_artifact import save_realization_statistics

from .common import (
    add_common_arguments,
    atomic_json,
    build_stage_a,
    load_model_checkpoint,
    prepare_context,
    project_relative_path,
    run_report,
    statistics_bindings,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--stage-a-checkpoint", required=True)
    parser.add_argument("--output", default="artifacts/realization_stats.npz")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        context = prepare_context(arguments)
        output = project_relative_path(
            context.project_root, arguments.output, label="statistics output"
        )
        model = build_stage_a(context)
        _, stage_a_digest = load_model_checkpoint(
            context,
            arguments.stage_a_checkpoint,
            model,
            expected_stage="A",
        )
        report = run_report(context, stage="statistics", dry_run=arguments.dry_run)
        report["output"] = Path(arguments.output).as_posix()
        atomic_json(context.output_dir / "resolved_run.json", report)
        if arguments.dry_run:
            print(json.dumps(report, sort_keys=True))
            return
        model = model.to(context.model.runtime.device).eval()
        dataset = MotionCaptionDataset(
            context.dataset,
            context.assets,
            split=context.assets.split,
            seed=arguments.seed,
        )
        loader = DataLoader(
            dataset,
            batch_size=context.model.training.stage_a.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_motion_records,
        )
        moments = RunningMoments(128)
        with torch.no_grad():
            for batch in loader:
                batch = move_to_device(batch, context.model.runtime.device)
                _, native = model.encode(batch)
                moments.update(native.values, batch.frame_mask)
        mean, scale = moments.finalize(std_floor=1e-4)
        save_realization_statistics(
            output,
            mean,
            scale,
            statistics_bindings(context, stage_a_digest),
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
