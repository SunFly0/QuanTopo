"""Train one QuanTopo stage."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

from quantopo.contracts.tensors import MotionBatch, PersistenceBatch
from quantopo.data.collate import collate_motion_records
from quantopo.data.dataset import MotionCaptionDataset, MotionViewIdentity
from quantopo.models.factory import MethodComponents
from quantopo.topology.cache import TopologyCache
from quantopo.training.checkpoint import CheckpointMetadata, save_checkpoint_atomic
from quantopo.training.distributed import (
    DistributedContext,
    global_to_local_batch,
    make_sampler,
    wrap_ddp,
)
from quantopo.training.engine import EngineConfig
from quantopo.training.selection import CheckpointSelector, SelectionMetric
from quantopo.training.stages import StageATrainer, StageB1Trainer, StageB2Trainer

from .common import (
    CLIContext,
    add_common_arguments,
    atomic_json,
    build_stage_a,
    build_stage_b1,
    build_stage_b2,
    file_digest,
    load_model_checkpoint,
    load_realization_transform,
    package_digest,
    prepare_context,
    project_relative_path,
    run_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--stage", choices=("a", "b1", "b2"), required=True)
    parser.add_argument("--stage-a-checkpoint")
    parser.add_argument("--stage-b1-checkpoint")
    parser.add_argument("--clip-checkpoint", default="assets/clip/ViT-B-32.pt")
    parser.add_argument("--realization-statistics", default="artifacts/realization_stats.npz")
    parser.add_argument("--topology-cache", default="artifacts/topology/train")
    return parser


def _required_parents(parser: argparse.ArgumentParser, arguments: argparse.Namespace) -> None:
    if arguments.stage in {"b1", "b2"} and not arguments.stage_a_checkpoint:
        parser.error("Stage A checkpoint is required for Stage B1/B2")
    if arguments.stage == "b2" and not arguments.stage_b1_checkpoint:
        parser.error("Stage B1 checkpoint is required for Stage B2")


def _dataset(context: CLIContext, seed: int) -> MotionCaptionDataset:
    return MotionCaptionDataset(
        context.dataset,
        context.assets,
        split=context.assets.split,
        seed=seed,
    )


def _loader(
    dataset: MotionCaptionDataset,
    global_batch_size: int,
    generator: torch.Generator,
    distributed_context: DistributedContext,
    seed: int,
) -> DataLoader[MotionBatch]:
    sampler = make_sampler(dataset, distributed_context, seed)
    return DataLoader(
        dataset,  # type: ignore[arg-type]
        batch_size=global_to_local_batch(global_batch_size, distributed_context.world_size),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_motion_records,
        generator=generator,
    )


def _infinite(loader: DataLoader[MotionBatch]) -> Iterator[MotionBatch]:
    epoch = 0
    while True:
        dataset = loader.dataset
        if isinstance(dataset, MotionCaptionDataset):
            dataset.set_epoch(epoch)
        sampler = loader.sampler
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


@dataclass(frozen=True)
class _TopologySource:
    path: Path
    view: MotionViewIdentity
    selected_indices: torch.Tensor


def _topology_cache(
    context: CLIContext,
    encoded: str,
    *,
    variant_fingerprint: str = "main",
    sample_count: int | None = None,
) -> tuple[TopologyCache, dict[str, _TopologySource]]:
    root = project_relative_path(
        context.project_root, encoded, label="topology cache", must_exist=True
    )
    cache = TopologyCache(
        root,
        context.assets.root,
        context.assets.root / f"{context.assets.split}.txt",
        context.model.topology,
        variant_fingerprint=variant_fingerprint,
    )
    dataset = _dataset(context, seed=0)
    sources = {
        dataset.samples[index].key: _TopologySource(
            path=dataset.source_ric_path(index),
            view=dataset.view_identity(index),
            selected_indices=dataset.selected_global_indices(
                index,
                context.model.topology.sample_count if sample_count is None else sample_count,
            ),
        )
        for index in range(len(dataset))
    }
    return cache, sources


def _reference_batch(
    batch: MotionBatch,
    cache: TopologyCache,
    sources: dict[str, _TopologySource],
) -> PersistenceBatch:
    entries = [
        cache.load(
            motion_id,
            sources[motion_id].path,
            view_identity=sources[motion_id].view,
            expected_selected_indices=sources[motion_id].selected_indices,
        )
        for motion_id in batch.motion_ids
    ]
    return PersistenceBatch(
        points=torch.stack([entry.diagram.points for entry in entries]),
        mask=torch.stack([entry.diagram.mask for entry in entries]),
        descriptor=torch.stack([entry.descriptor for entry in entries]),
    )


def _descriptor_scale(cache: TopologyCache, sources: dict[str, _TopologySource]) -> torch.Tensor:
    descriptors = torch.stack(
        [
            cache.load(
                motion_id,
                source.path,
                view_identity=source.view,
                expected_selected_indices=source.selected_indices,
            ).descriptor
            for motion_id, source in sources.items()
        ]
    )
    return descriptors.std(dim=0, correction=0).clamp_min(1e-3)


def _combined_digest(*values: str) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


def _metadata(
    context: CLIContext,
    stage: str,
    *,
    parent: str | None,
    topology_manifest: Path | None,
) -> CheckpointMetadata:
    return CheckpointMetadata(
        stage=stage,
        architecture=file_digest(context.model_path),
        dataset=file_digest(context.data_path),
        skeleton=file_digest(context.assets.offsets_path),
        statistics=_combined_digest(
            file_digest(context.assets.mean_path), file_digest(context.assets.std_path)
        ),
        topology=("none" if topology_manifest is None else file_digest(topology_manifest)),
        source=package_digest(),
        parent=parent,
    )


def _save_checkpoint(
    context: CLIContext,
    distributed_context: DistributedContext,
    trainer: StageATrainer | StageB1Trainer | StageB2Trainer,
    *,
    stage: str,
    parent: str | None,
    topology_manifest: Path | None,
    generators: dict[str, torch.Generator],
    report: dict[str, object],
    name: str = "final",
    selection_state: dict[str, object] | None = None,
) -> None:
    if not distributed_context.is_primary:
        return
    save_checkpoint_atomic(
        context.output_dir / f"stage_{stage.lower()}_{name}.pt",
        model=trainer.model,
        optimizer=trainer.engine.optimizer,
        schedule=trainer.engine.schedule,
        scaler=trainer.engine.scaler,
        update=trainer.engine.update,
        metadata=_metadata(
            context,
            stage,
            parent=parent,
            topology_manifest=topology_manifest,
        ),
        generators=generators,
        resolved_config=report,
        manifests=report["assets"],  # type: ignore[arg-type]
        selection_state=selection_state,
    )


def _save_periodic(
    context: CLIContext,
    distributed_context: DistributedContext,
    trainer: StageATrainer | StageB1Trainer | StageB2Trainer,
    *,
    stage: str,
    parent: str | None,
    topology_manifest: Path | None,
    generators: dict[str, torch.Generator],
    report: dict[str, object],
    validation_callback: Callable[[torch.nn.Module, int], float] | None = None,
    selector: CheckpointSelector | None = None,
) -> None:
    if not trainer.engine.should_validate():
        return
    improved = False
    if validation_callback is not None:
        if selector is None:
            raise ValueError("validation callback requires a checkpoint selector")
        value = validation_callback(trainer.model, trainer.engine.update)
        improved = selector.observe(trainer.engine.update, value)
    selection_state = None if selector is None else selector.state_dict()
    names = ["latest", f"update_{trainer.engine.update:07d}"]
    if improved:
        names.append("best")
    for name in names:
        _save_checkpoint(
            context,
            distributed_context,
            trainer,
            stage=stage,
            parent=parent,
            topology_manifest=topology_manifest,
            generators=generators,
            report=report,
            name=name,
            selection_state=selection_state,
        )


def _run_stage_a(
    context: CLIContext,
    arguments: argparse.Namespace,
    report: dict[str, object],
    distributed_context: DistributedContext,
    components: MethodComponents | None = None,
    validation_callback: Callable[[torch.nn.Module, int], float] | None = None,
) -> None:
    model = build_stage_a(context, components).to(distributed_context.device)
    cache, sources = _topology_cache(
        context,
        arguments.topology_cache,
        variant_fingerprint=getattr(arguments, "topology_variant_fingerprint", "main"),
        sample_count=getattr(arguments, "topology_sample_count", None),
    )
    rank_seed = arguments.seed + 10_000 * distributed_context.rank
    data_generator = torch.Generator().manual_seed(rank_seed)
    loader = _loader(
        _dataset(context, arguments.seed),
        context.model.training.stage_a.batch_size,
        data_generator,
        distributed_context,
        arguments.seed,
    )
    trainer = StageATrainer(
        model,
        total_updates=context.model.training.stage_a.steps,
        engine_config=EngineConfig.from_training(context.model.training),
    )
    trainer.forward_model = wrap_ddp(trainer.model, distributed_context)
    selector = (
        None
        if validation_callback is None
        else CheckpointSelector(SelectionMetric("validation_objective", "min"))
    )
    for batch in _infinite(loader):
        reference = _reference_batch(batch, cache, sources)
        trainer.train_step(batch, reference.descriptor)
        _save_periodic(
            context,
            distributed_context,
            trainer,
            stage="A",
            parent=None,
            topology_manifest=cache.manifest_path,
            generators={"data": data_generator},
            report=report,
            validation_callback=validation_callback,
            selector=selector,
        )
        if trainer.engine.update >= context.model.training.stage_a.steps:
            break
    _save_checkpoint(
        context,
        distributed_context,
        trainer,
        stage="A",
        parent=None,
        topology_manifest=cache.manifest_path,
        generators={"data": data_generator},
        report=report,
    )
    distributed_context.barrier()


def _run_stage_b1(
    context: CLIContext,
    arguments: argparse.Namespace,
    report: dict[str, object],
    distributed_context: DistributedContext,
    components: MethodComponents | None = None,
    validation_callback: Callable[[torch.nn.Module, int], float] | None = None,
) -> None:
    representation = build_stage_a(context, components)
    cache, sources = _topology_cache(
        context,
        arguments.topology_cache,
        variant_fingerprint=getattr(arguments, "topology_variant_fingerprint", "main"),
        sample_count=getattr(arguments, "topology_sample_count", None),
    )
    _, stage_a_digest = load_model_checkpoint(
        context,
        arguments.stage_a_checkpoint,
        representation,
        expected_stage="A",
        expected_topology=file_digest(cache.manifest_path),
    )
    model = build_stage_b1(
        context,
        representation,
        clip_checkpoint=arguments.clip_checkpoint,
        descriptor_scale=_descriptor_scale(cache, sources),
        components=components,
    ).to(distributed_context.device)
    rank_seed = arguments.seed + 10_000 * distributed_context.rank
    data_generator = torch.Generator().manual_seed(rank_seed)
    loader = _loader(
        _dataset(context, arguments.seed),
        context.model.training.stage_b1.batch_size,
        data_generator,
        distributed_context,
        arguments.seed,
    )
    trainer = StageB1Trainer(
        model,
        total_updates=context.model.training.stage_b1.steps,
        engine_config=EngineConfig.from_training(context.model.training),
    )
    trainer.forward_model = wrap_ddp(trainer.model, distributed_context)
    selector = (
        None
        if validation_callback is None
        else CheckpointSelector(SelectionMetric("validation_objective", "min"))
    )
    for batch in _infinite(loader):
        trainer.train_step(batch, _reference_batch(batch, cache, sources))
        _save_periodic(
            context,
            distributed_context,
            trainer,
            stage="B1",
            parent=stage_a_digest,
            topology_manifest=cache.manifest_path,
            generators={"data": data_generator},
            report=report,
            validation_callback=validation_callback,
            selector=selector,
        )
        if trainer.engine.update >= context.model.training.stage_b1.steps:
            break
    _save_checkpoint(
        context,
        distributed_context,
        trainer,
        stage="B1",
        parent=stage_a_digest,
        topology_manifest=cache.manifest_path,
        generators={"data": data_generator},
        report=report,
    )
    distributed_context.barrier()


def _device_generator(device: str | torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def _run_stage_b2(
    context: CLIContext,
    arguments: argparse.Namespace,
    report: dict[str, object],
    distributed_context: DistributedContext,
    components: MethodComponents | None = None,
    fid_callback: Callable[[torch.nn.Module, int], float] | None = None,
) -> None:
    representation = build_stage_a(context, components)
    _, stage_a_digest = load_model_checkpoint(
        context, arguments.stage_a_checkpoint, representation, expected_stage="A"
    )
    planning = build_stage_b1(
        context,
        representation,
        clip_checkpoint=arguments.clip_checkpoint,
        components=components,
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
    model = build_stage_b2(context, planning, transform, components).to(distributed_context.device)
    rank_seed = arguments.seed + 10_000 * distributed_context.rank
    data_generator = torch.Generator().manual_seed(rank_seed)
    loader = _loader(
        _dataset(context, arguments.seed),
        context.model.training.stage_b2.batch_size,
        data_generator,
        distributed_context,
        arguments.seed,
    )
    trainer = StageB2Trainer(
        model,
        total_updates=context.model.training.stage_b2.steps,
        engine_config=EngineConfig.from_training(context.model.training),
    )
    trainer.forward_model = wrap_ddp(trainer.model, distributed_context)
    selector = None if fid_callback is None else CheckpointSelector(SelectionMetric("fid", "min"))
    device = distributed_context.device
    generators = {
        "data": data_generator,
        "noise": _device_generator(device, rank_seed + 1),
        "time": _device_generator(device, rank_seed + 2),
        "dropout": _device_generator(device, rank_seed + 3),
        "proposal": _device_generator(device, rank_seed + 4),
        "ar": _device_generator(device, rank_seed + 5),
    }
    for batch in _infinite(loader):
        trainer.train_step(
            batch,
            noise_generator=generators["noise"],
            time_generator=generators["time"],
            dropout_generator=generators["dropout"],
            proposal_generator=generators["proposal"],
            ar_generator=generators["ar"],
        )
        _save_periodic(
            context,
            distributed_context,
            trainer,
            stage="B2",
            parent=stage_b1_digest,
            topology_manifest=None,
            generators=generators,
            report=report,
            validation_callback=fid_callback,
            selector=selector,
        )
        if trainer.engine.update >= context.model.training.stage_b2.steps:
            break
    _save_checkpoint(
        context,
        distributed_context,
        trainer,
        stage="B2",
        parent=stage_b1_digest,
        topology_manifest=None,
        generators=generators,
        report=report,
    )
    distributed_context.barrier()


def _dry_construct(
    context: CLIContext,
    arguments: argparse.Namespace,
    components: MethodComponents | None = None,
) -> torch.nn.Module:
    representation = build_stage_a(context, components)
    if arguments.stage == "a":
        return representation
    expected_topology = None
    if arguments.stage == "b1":
        cache, _ = _topology_cache(
            context,
            arguments.topology_cache,
            variant_fingerprint=getattr(arguments, "topology_variant_fingerprint", "main"),
            sample_count=getattr(arguments, "topology_sample_count", None),
        )
        expected_topology = file_digest(cache.manifest_path)
    _, stage_a_digest = load_model_checkpoint(
        context,
        arguments.stage_a_checkpoint,
        representation,
        expected_stage="A",
        expected_topology=expected_topology,
    )
    planning = build_stage_b1(
        context,
        representation,
        clip_checkpoint=arguments.clip_checkpoint,
        components=components,
    )
    if arguments.stage == "b1":
        return planning
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
    model = build_stage_b2(context, planning, transform, components)
    # The parent is checked above; construction validates every B2 tensor interface.
    if not stage_b1_digest:
        raise RuntimeError("empty Stage B1 checkpoint digest")
    return model


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    arguments = parser.parse_args(argv)
    _required_parents(parser, arguments)
    distributed_context: DistributedContext | None = None
    try:
        project_relative_path(
            Path(arguments.project_root), arguments.output_dir, label="output directory"
        )
        context = prepare_context(arguments)
        distributed_context = DistributedContext.from_environment(context.model.runtime.device)
        rank_seed = arguments.seed + 10_000 * distributed_context.rank
        torch.manual_seed(rank_seed)
        report: dict[str, object] = run_report(
            context, stage=arguments.stage, dry_run=arguments.dry_run
        )
        report["distributed"] = {
            "world_size": distributed_context.world_size,
            "global_batch": getattr(context.model.training, f"stage_{arguments.stage}").batch_size,
            "local_batch": global_to_local_batch(
                getattr(context.model.training, f"stage_{arguments.stage}").batch_size,
                distributed_context.world_size,
            ),
        }
        if distributed_context.is_primary:
            atomic_json(context.output_dir / "resolved_run.json", report)
        if arguments.dry_run:
            model = _dry_construct(context, arguments)
            report["parameters"] = sum(parameter.numel() for parameter in model.parameters())
            if distributed_context.is_primary:
                atomic_json(context.output_dir / "resolved_run.json", report)
                print(json.dumps(report, sort_keys=True))
            return
        if arguments.stage == "a":
            _run_stage_a(context, arguments, report, distributed_context)
        elif arguments.stage == "b1":
            _run_stage_b1(context, arguments, report, distributed_context)
        else:
            _run_stage_b2(context, arguments, report, distributed_context)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    finally:
        if distributed_context is not None:
            distributed_context.close()


if __name__ == "__main__":
    main()
