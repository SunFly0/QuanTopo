"""Data structures for ablation definitions and resolved variants."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class AblationSpec:
    name: str
    allowed_paths: tuple[str, ...]
    overrides: Mapping[str, Any]
    replacements: Mapping[str, str]

    @classmethod
    def create(
        cls,
        name: str,
        *,
        overrides: Mapping[str, Any],
        replacements: Mapping[str, str] | None = None,
    ) -> AblationSpec:
        if not name or not overrides:
            raise ValueError("an ablation needs a name and at least one override")
        return cls(
            name=name,
            allowed_paths=tuple(overrides),
            overrides=MappingProxyType(dict(overrides)),
            replacements=MappingProxyType(dict(replacements or {})),
        )


@dataclass(frozen=True)
class BuiltAblation:
    name: str
    base_config: dict[str, Any]
    config_tree: dict[str, Any]
    config_diff: Mapping[str, tuple[Any, Any]]
    replacements: Mapping[str, str]

    def checkpoint_record(self) -> dict[str, Any]:
        return {
            "ablation": self.name,
            "config_diff": {
                path: {"full": before, "variant": after}
                for path, (before, after) in self.config_diff.items()
            },
            "replacements": dict(self.replacements),
        }
