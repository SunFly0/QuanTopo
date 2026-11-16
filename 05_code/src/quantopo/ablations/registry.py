"""Ablation names and their configuration changes."""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping
from dataclasses import asdict
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from quantopo.contracts.config import QuanTopoConfig

from .spec import AblationSpec, BuiltAblation

_DEFAULT_CONTROLS: dict[str, Any] = {
    "loss": {"plan_topology_weight": 0.1, "anchor_weight": 0.5},
    "planner": {"topology_modulation": True},
    "flow": {"rollin_probability": 0.2},
    "topology": {"predictor_inputs": "text_length"},
    "decoder": {"variant": "anatomical_mps", "routing": "anatomical"},
    "realization": {"generator": "conditional_flow"},
}


def _set_leaf(tree: dict[str, Any], path: str, value: Any) -> None:
    pieces = path.split(".")
    cursor = tree
    for piece in pieces[:-1]:
        child = cursor.get(piece)
        if not isinstance(child, dict):
            raise KeyError(f"ablation path does not exist: {path}")
        cursor = child
    if pieces[-1] not in cursor:
        raise KeyError(f"ablation path does not exist: {path}")
    cursor[pieces[-1]] = value


def _leaf_values(tree: Mapping[str, Any], prefix: str = "") -> Iterator[tuple[str, Any]]:
    for key, value in tree.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            yield from _leaf_values(value, path)
        else:
            yield path, value


class AblationRegistry:
    def __init__(self, specs: tuple[AblationSpec, ...]) -> None:
        names = [spec.name for spec in specs]
        if len(names) != len(set(names)):
            raise ValueError("duplicate ablation registry key")
        self._specs = MappingProxyType({spec.name: spec for spec in specs})

    def __getitem__(self, name: str) -> AblationSpec:
        try:
            return self._specs[name]
        except KeyError as error:
            raise KeyError(f"unknown ablation: {name}") from error

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def build(self, name: str, full_config: QuanTopoConfig) -> BuiltAblation:
        spec = self[name]
        base_config = asdict(full_config)
        baseline = copy.deepcopy(_DEFAULT_CONTROLS)
        variant = copy.deepcopy(baseline)
        for path, value in spec.overrides.items():
            _set_leaf(variant, path, copy.deepcopy(value))
        before = dict(_leaf_values(baseline))
        after = dict(_leaf_values(variant))
        diff = {path: (before[path], after[path]) for path in before if before[path] != after[path]}
        unexpected = set(diff) - set(spec.allowed_paths)
        if unexpected or not diff:
            raise RuntimeError(
                f"ablation {name} changed invalid paths {sorted(unexpected)} or no paths"
            )
        return BuiltAblation(
            name=name,
            base_config=base_config,
            config_tree={"model": copy.deepcopy(base_config), "controls": variant},
            config_diff=MappingProxyType(diff),
            replacements=spec.replacements,
        )


def _spec(
    name: str,
    path: str,
    value: Any,
    *,
    component: str | None = None,
) -> AblationSpec:
    replacements = {} if component is None else {path: component}
    return AblationSpec.create(name, overrides={path: value}, replacements=replacements)


CORE_ABLATIONS = AblationRegistry(
    (
        _spec("no_plan_topology", "loss.plan_topology_weight", 0.0),
        _spec("no_anchor", "loss.anchor_weight", 0.0),
        _spec("no_topology_modulation", "planner.topology_modulation", False),
        _spec("no_rollin", "flow.rollin_probability", 0.0),
        _spec("text_only_topology", "topology.predictor_inputs", "text_only"),
        _spec(
            "shared_decoder",
            "decoder.variant",
            "shared",
            component="SharedContextDecoder",
        ),
        _spec("flat_mps", "decoder.variant", "flat_mps", component="FlatMPSDecoder"),
        _spec(
            "hierarchical_conditional_mlp",
            "decoder.variant",
            "hierarchical_conditional_mlp",
            component="HierarchicalConditionalMLPDecoder",
        ),
        _spec(
            "random_routing_seed2027",
            "decoder.routing",
            "random_seed2027",
            component="RandomRoutingDecoder",
        ),
        _spec(
            "regression_realization",
            "realization.generator",
            "deterministic_regression",
            component="DeterministicRealizationRegressor",
        ),
    )
)

ALL_CORE_ABLATIONS = tuple(CORE_ABLATIONS)


def build_ablation(name: str, full_config: QuanTopoConfig) -> BuiltAblation:
    return CORE_ABLATIONS.build(name, full_config)


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _unique_mapping,
)


def load_ablation_selection(path: str | Path, registry: AblationRegistry) -> dict[str, str]:
    raw = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    if not isinstance(raw, dict) or set(raw) != {"rows"} or not isinstance(raw["rows"], dict):
        raise ValueError("ablation selection must contain exactly one rows mapping")
    rows = {str(row): str(key) for row, key in raw["rows"].items()}
    for key in rows.values():
        registry[key]
    if len(rows.values()) != len(set(rows.values())):
        raise ValueError("duplicate ablation registry key in selection")
    return rows
