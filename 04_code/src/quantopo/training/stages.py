"""Thin stage-specific adapters over the common training engine."""

from __future__ import annotations

from typing import cast

import torch

from quantopo.contracts.tensors import MotionBatch, PersistenceBatch
from quantopo.models.planning import StageB1Model, StageB1Output
from quantopo.models.quantopo import StageB2Model, StageB2Output
from quantopo.models.representation import StageAModel, StageAOutput

from .engine import EngineConfig, StepRecord, TrainingEngine, move_to_device
from .freezing import (
    assert_parameter_ownership,
    freeze_module,
    trainable_parameter_names,
    unfreeze_module,
)


class _BaseTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        *,
        total_updates: int,
        warmup_updates: int | None,
        engine_config: EngineConfig | None,
        forward_model: torch.nn.Module | None = None,
    ) -> None:
        if warmup_updates is not None and engine_config is not None:
            raise ValueError("provide warmup_updates or engine_config, not both")
        resolved_config = (
            EngineConfig(warmup_updates=warmup_updates)
            if engine_config is None and warmup_updates is not None
            else engine_config
        )
        if resolved_config is None:
            raise ValueError("trainer requires an engine configuration")
        self.model = model
        self.forward_model = model if forward_model is None else forward_model
        self.trainable_names = trainable_parameter_names(model)
        assert_parameter_ownership(model, self.trainable_names)
        self.engine = TrainingEngine(
            model,
            total_updates=total_updates,
            config=resolved_config,
        )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device


class StageATrainer(_BaseTrainer):
    def __init__(
        self,
        model: StageAModel,
        *,
        total_updates: int,
        warmup_updates: int | None = None,
        engine_config: EngineConfig | None = None,
        forward_model: torch.nn.Module | None = None,
    ) -> None:
        unfreeze_module(model)
        model.quantizer.unfreeze()
        model.train()
        super().__init__(
            model,
            total_updates=total_updates,
            warmup_updates=warmup_updates,
            engine_config=engine_config,
            forward_model=forward_model,
        )

    def train_step(self, batch: MotionBatch, reference_descriptor: torch.Tensor) -> StepRecord:
        batch = move_to_device(batch, self.device)
        reference_descriptor = reference_descriptor.to(self.device)
        return self.engine.step(
            lambda: cast(StageAOutput, self.forward_model(batch, reference_descriptor)).loss
        )


class StageB1Trainer(_BaseTrainer):
    def __init__(
        self,
        model: StageB1Model,
        *,
        total_updates: int,
        warmup_updates: int | None = None,
        engine_config: EngineConfig | None = None,
        forward_model: torch.nn.Module | None = None,
    ) -> None:
        freeze_module(model.representation)
        model.representation.quantizer.freeze()
        freeze_module(model.text_encoder)
        unfreeze_module(model.topology_predictor)
        unfreeze_module(model.planner)
        model.train()
        super().__init__(
            model,
            total_updates=total_updates,
            warmup_updates=warmup_updates,
            engine_config=engine_config,
            forward_model=forward_model,
        )

    def train_step(self, batch: MotionBatch, reference: PersistenceBatch) -> StepRecord:
        batch = move_to_device(batch, self.device)
        reference = move_to_device(reference, self.device)
        return self.engine.step(
            lambda: cast(StageB1Output, self.forward_model(batch, reference)).losses.total
        )


class StageB2Trainer(_BaseTrainer):
    def __init__(
        self,
        model: StageB2Model,
        *,
        total_updates: int,
        warmup_updates: int | None = None,
        engine_config: EngineConfig | None = None,
        forward_model: torch.nn.Module | None = None,
    ) -> None:
        freeze_module(model.planning)
        freeze_module(model.transform)
        unfreeze_module(model.realization_module)
        model.train()
        super().__init__(
            model,
            total_updates=total_updates,
            warmup_updates=warmup_updates,
            engine_config=engine_config,
            forward_model=forward_model,
        )

    def train_step(
        self,
        batch: MotionBatch,
        *,
        noise_generator: torch.Generator,
        time_generator: torch.Generator,
        dropout_generator: torch.Generator,
        proposal_generator: torch.Generator,
        ar_generator: torch.Generator,
    ) -> StepRecord:
        batch = move_to_device(batch, self.device)
        return self.engine.step(
            lambda: cast(
                StageB2Output,
                self.forward_model(
                    batch,
                    update=self.engine.update,
                    noise_generator=noise_generator,
                    time_generator=time_generator,
                    dropout_generator=dropout_generator,
                    proposal_generator=proposal_generator,
                    ar_generator=ar_generator,
                ),
            ).loss
        )
