"""Generate motions from text and explicit durations with selected QuanTopo checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from quantopo.export.results import export_generation_result
from quantopo.models.quantopo import QuanTopo

from .common import (
    add_common_arguments,
    atomic_json,
    build_stage_a,
    build_stage_b1,
    build_stage_b2,
    load_model_checkpoint,
    load_realization_transform,
    prepare_context,
    project_relative_path,
    run_report,
)


def _motion_length(encoded: str) -> int:
    try:
        value = int(encoded)
    except ValueError as error:
        raise argparse.ArgumentTypeError("length must be an integer in 4..196") from error
    if not 4 <= value <= 196:
        raise argparse.ArgumentTypeError("length must be in 4..196")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--stage-a-checkpoint")
    parser.add_argument("--stage-b1-checkpoint")
    parser.add_argument("--stage-b2-checkpoint")
    parser.add_argument("--clip-checkpoint", default="assets/clip/ViT-B-32.pt")
    parser.add_argument("--realization-statistics", default="artifacts/realization_stats.npz")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text")
    group.add_argument("--input-jsonl")
    parser.add_argument("--length", type=_motion_length)
    return parser


def _inputs(
    project_root: Path, arguments: argparse.Namespace
) -> tuple[tuple[str, ...], torch.Tensor]:
    if arguments.input_jsonl:
        path = project_relative_path(
            project_root, arguments.input_jsonl, label="input JSONL", must_exist=True
        )
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not records:
            raise ValueError("input JSONL is empty")
        captions = tuple(str(record["text"]) for record in records)
        lengths = torch.tensor([int(record["length"]) for record in records], dtype=torch.long)
    else:
        if arguments.text is None or arguments.length is None:
            raise ValueError("provide --text with --length, or --input-jsonl")
        captions = (arguments.text,)
        lengths = torch.tensor([arguments.length], dtype=torch.long)
    if bool(torch.any(lengths < 4)) or bool(torch.any(lengths > 196)):
        raise ValueError("all supplied lengths must lie in 4..196")
    return captions, lengths


def _require_checkpoints(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> None:
    missing = [
        label
        for label, value in (
            ("Stage A", arguments.stage_a_checkpoint),
            ("Stage B1", arguments.stage_b1_checkpoint),
            ("Stage B2", arguments.stage_b2_checkpoint),
        )
        if value is None
    ]
    if missing:
        parser.error(f"{', '.join(missing)} checkpoint(s) required")


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    arguments = parser.parse_args(argv)
    _require_checkpoints(parser, arguments)
    try:
        context = prepare_context(arguments)
        captions, lengths = _inputs(context.project_root, arguments)
        representation = build_stage_a(context)
        _, stage_a_digest = load_model_checkpoint(
            context,
            arguments.stage_a_checkpoint,
            representation,
            expected_stage="A",
        )
        planning = build_stage_b1(
            context,
            representation,
            clip_checkpoint=arguments.clip_checkpoint,
        )
        _, stage_b1_digest = load_model_checkpoint(
            context,
            arguments.stage_b1_checkpoint,
            planning,
            expected_stage="B1",
            expected_parent=stage_a_digest,
        )
        transform = load_realization_transform(
            context,
            arguments.realization_statistics,
            expected_stage_a_digest=stage_a_digest,
        )
        stage_b2 = build_stage_b2(context, planning, transform)
        _, stage_b2_digest = load_model_checkpoint(
            context,
            arguments.stage_b2_checkpoint,
            stage_b2,
            expected_stage="B2",
            expected_parent=stage_b1_digest,
        )
        report = run_report(context, stage="generate", dry_run=arguments.dry_run)
        report["captions"] = list(captions)
        report["lengths"] = lengths.tolist()
        atomic_json(context.output_dir / "resolved_run.json", report)
        if arguments.dry_run:
            print(json.dumps(report, sort_keys=True))
            return
        device = torch.device(context.model.runtime.device)
        model = (
            QuanTopo(
                planning=stage_b2.planning,
                transform=stage_b2.transform,
                flow=stage_b2.flow,
                euler_steps=context.model.sampling.euler_steps,
            )
            .to(device)
            .eval()
        )
        lengths = lengths.to(device)
        output = model.generate(
            captions,
            lengths,
            ar_generator=torch.Generator(device=device).manual_seed(arguments.seed),
            flow_generator=torch.Generator(device=device).manual_seed(arguments.seed + 1),
        )
        export_generation_result(
            context.output_dir,
            output=output,
            captions=captions,
            lengths=lengths,
            seeds={"ar": arguments.seed, "flow": arguments.seed + 1},
            checkpoint_digests={
                "stage_a": stage_a_digest,
                "stage_b1": stage_b1_digest,
                "stage_b2": stage_b2_digest,
            },
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
