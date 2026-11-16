"""Resolve ablation configurations into model components."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType
from typing import Any

from quantopo.contracts.config import QuanTopoConfig
from quantopo.models.factory import MethodComponents, default_components

from .registry import build_ablation
from .summaries import build_topology_sensitivity


@dataclass(frozen=True)
class AblationVariant:
    name: str
    components: MethodComponents
    config_diff: Mapping[str, tuple[Any, Any]]
    replacements: Mapping[str, str]
    cache_fingerprint: str
    artifact_requirements: tuple[str, ...] = ()


def _fingerprint(name: str, components: MethodComponents) -> str:
    payload = {"name": name, "components": asdict(components)}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_core_variant(name: str, config: QuanTopoConfig) -> AblationVariant:
    built = build_ablation(name, config)
    controls = built.config_tree["controls"]
    components = default_components()
    stage_a = components.stage_a_loss
    if controls["loss"]["plan_topology_weight"] != stage_a.plan_topology:
        stage_a = replace(stage_a, plan_topology=controls["loss"]["plan_topology_weight"])
    if controls["loss"]["anchor_weight"] != stage_a.anchor:
        stage_a = replace(stage_a, anchor=controls["loss"]["anchor_weight"])
    decoder = controls["decoder"]["variant"]
    decoder_names = {
        "anatomical_mps": "anatomical",
        "shared": "shared_decoder",
        "flat_mps": "flat_mps",
        "hierarchical_conditional_mlp": "hierarchical_conditional_mlp",
    }
    decoder = decoder_names[decoder]
    if controls["decoder"]["routing"] == "random_seed2027":
        decoder = "random_routing_seed2027"
    components = replace(
        components,
        stage_a_loss=stage_a,
        topology_modulation=controls["planner"]["topology_modulation"],
        topology_length_conditioning=controls["topology"]["predictor_inputs"] != "text_only",
        decoder=decoder,
        rollin_probability=controls["flow"]["rollin_probability"],
        realization_generator=controls["realization"]["generator"],
    )
    return AblationVariant(
        name,
        components,
        built.config_diff,
        built.replacements,
        _fingerprint(name, components),
    )


def build_extended_variant(
    catalog: str, key: str, config: QuanTopoConfig | None = None
) -> AblationVariant:
    components = default_components()
    requirements: tuple[str, ...] = ()
    if catalog == "summary":
        components = replace(components, summary=key)
        if key in {"acf", "psd", "edm"}:
            requirements = ("fitted_summary_transform",)
    elif catalog == "topology_sensitivity":
        components = replace(components, topology_sensitivity=key)
        if key == "block_balanced":
            requirements = ("fitted_block_scales",)
    elif catalog == "quantizer":
        components = replace(components, quantizer=key)
    elif catalog == "readout":
        components = replace(components, plan_readout=key)
    else:
        raise KeyError(f"unknown ablation catalog: {catalog}")
    default = default_components()
    diff = {
        field: (getattr(default, field), getattr(components, field))
        for field in asdict(default)
        if getattr(default, field) != getattr(components, field)
    }
    cache_fingerprint = _fingerprint(key, components)
    if catalog == "topology_sensitivity":
        if config is None:
            raise ValueError("topology sensitivity requires the model configuration")
        cache_fingerprint = build_topology_sensitivity(key, config.topology).cache_fingerprint
    return AblationVariant(
        key,
        components,
        MappingProxyType(diff),
        MappingProxyType({}),
        cache_fingerprint,
        requirements,
    )
