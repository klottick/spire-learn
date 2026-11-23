"""One recipe API over the M10 scenario generators and ``baalorlord-standard-v1``.

Every stream, resume, collector and baseline caller goes through ``load_recipe``,
``generate_shard`` and ``build_panel``. M10 recipes (``ironclad-curriculum-v1``/``v2``)
delegate unchanged to ``scenario_stream`` and the development panel, so their shards,
panels and hashes are byte-identical to calling those functions directly.

Standard recipes need the Gym's catalog validator for fuzzed shards and for panels. It is
bound once per process: ``gym_validator()`` binds the pinned Gym lazily on first use (so
it only works inside the runtime container; outside it ``GymUnavailable`` says so), and
``bind_validator`` injects a fake in host tests. One validator object per process keeps the Standard shard cache effective, and a
resumed process regenerates byte-identical shards because generation depends only on the
recipe, the shard position and the Gym's (deterministic) verdicts. Importing this module
never imports the Gym.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping
import hashlib
from pathlib import Path
import threading

from .scenario_stream import ScenarioRecipe, ScenarioShard, generate_scenario_shard, load_scenario_recipe
from .scenario_support import json_value, read_json
from .standard_catalog import Validate, bind_gym
from .standard_stream import (
    DEFAULT_ALIASES, DEFAULT_ENTRY_RESOURCES, GENERATOR as STANDARD_GENERATOR, PANEL_SCHEMA as STANDARD_PANEL_SCHEMA, StandardRecipe,
    build_standard_panel, generate_standard_shard, levels_for_shard, load_standard_recipe,
)

Recipe = ScenarioRecipe | StandardRecipe

_LOCK = threading.Lock()
_VALIDATOR: Validate | None = None
_RECIPE_CACHE: OrderedDict[tuple, StandardRecipe] = OrderedDict()
_RECIPE_CACHE_CAPACITY = 4
_NO_GYM = "Standard {what} require the Gym validator; run inside the runtime container"


class GymUnavailable(RuntimeError):
    """The pinned Gym is not importable here (host-side tooling outside the runtime container)."""


def bind_validator(validate: Validate | None) -> Validate | None:
    """Bind the process's Standard catalog validator (``None`` unbinds); returns the previous one."""
    global _VALIDATOR
    if validate is not None and not callable(validate):
        raise TypeError("validate must be callable")
    with _LOCK:
        previous, _VALIDATOR = _VALIDATOR, validate
    return previous


def _native_build_info() -> Mapping[str, object]:
    from spire_lab import _native
    return _native.build_info()


def gym_validator(what: str = "fuzzed shards", recipe: Recipe | None = None, *,
                  build_info: Callable[[], Mapping[str, object]] | None = None) -> Validate:
    """The bound validator, binding the pinned Gym on first use (runtime container only).

    A validator bound from the real Gym records the loaded build's revision (``live_gym_revision``).
    With a Standard ``recipe``, that revision must equal the recipe's table revision: a v1.1.1
    recipe is never validated or played on the v1.2.0 Gym, nor the reverse. Validators injected by
    ``bind_validator`` (host tests) carry no revision unless they set one, and are not checked.
    ``build_info`` replaces the loaded extension's ``build_info`` (tests).

    Without the Gym (host-side resume inspection past the first fuzz boundary, say) this
    raises ``GymUnavailable`` instead of an ``ImportError``.
    """
    global _VALIDATOR
    with _LOCK:
        if _VALIDATOR is None:
            try:
                validator = bind_gym()[0]
                identity = (build_info or _native_build_info)()
            except ImportError as error:
                raise GymUnavailable(_NO_GYM.format(what=what)) from error
            validator.live_gym_revision = identity.get("revision")
            _VALIDATOR = validator
        live = getattr(_VALIDATOR, "live_gym_revision", None)
        if isinstance(recipe, StandardRecipe) and live is not None and live != recipe.gym_revision:
            raise ValueError(f"the loaded Spire Gym is revision {live}, but the Standard recipe's run-states "
                             f"table is for Gym {recipe.gym_revision}; run it in that Gym's runtime")
        return _VALIDATOR


def is_standard_recipe_file(path: Path) -> bool:
    return read_json(path).get("generator_version") == STANDARD_GENERATOR


def _standard_cache_key(path: Path) -> tuple | None:
    """The recipe's bytes and its table's and alias file's actual SHA256s, or None if unreadable.

    The table is rehashed on every load, so a cached recipe is reused only for the exact same
    authenticated bytes; any problem falls back to the full loader, which reports it.
    """
    try:
        source = Path(path).resolve()
        data = source.read_bytes()
        spec = read_json(source).get("states_table")
        table = source.parent / spec["path"]
        return (str(source), data, str(table.resolve()), hashlib.sha256(table.read_bytes()).hexdigest(),
                hashlib.sha256(DEFAULT_ALIASES.read_bytes()).hexdigest(),
                hashlib.sha256(DEFAULT_ENTRY_RESOURCES.read_bytes()).hexdigest())
    except (OSError, ValueError, TypeError, KeyError):
        return None


def load_recipe(path: Path) -> Recipe:
    """Load a recipe, dispatching on ``generator_version``."""
    if not is_standard_recipe_file(path):
        return load_scenario_recipe(path)
    key = _standard_cache_key(path)
    if key is not None:
        with _LOCK:
            cached = _RECIPE_CACHE.get(key)
            if cached is not None:
                _RECIPE_CACHE.move_to_end(key)
                return cached
    recipe = load_standard_recipe(path)
    if key is not None:
        with _LOCK:
            _RECIPE_CACHE[key] = recipe
            while len(_RECIPE_CACHE) > _RECIPE_CACHE_CAPACITY:
                _RECIPE_CACHE.popitem(last=False)
    return recipe


def require_recipe_revision(recipe: Recipe, gym_revision: str) -> None:
    """A Standard recipe's table was built for one Gym revision; a config for another must not use it."""
    if isinstance(recipe, StandardRecipe) and recipe.gym_revision != gym_revision:
        raise ValueError(f"the Standard recipe's run-states table is for Gym {recipe.gym_revision}, "
                         f"but the configuration is for Gym {gym_revision}")


def generate_shard(recipe: Recipe, *, split: str, shard_index: int, arm: str) -> ScenarioShard:
    """The recipe's shard; a fuzzed Standard shard uses the process's Gym validator."""
    if isinstance(recipe, StandardRecipe):
        validate = gym_validator(recipe=recipe) if max(levels_for_shard(recipe, shard_index)) else None
        return generate_standard_shard(recipe, split=split, shard_index=shard_index, arm=arm, validate=validate)
    return generate_scenario_shard(recipe, split=split, shard_index=shard_index, arm=arm)


def build_panel(recipe: Recipe, split: str = "selection") -> Mapping[str, object]:
    """The evaluation panel: M10's 64-case development panel, or a Standard selection/final panel."""
    if isinstance(recipe, StandardRecipe):
        # Plain JSON values, like the development panel, so callers can serialize and slice it.
        return json_value(build_standard_panel(recipe, split, validate=gym_validator("panels", recipe)))
    if split != "selection":
        raise ValueError("M10 recipes have only the selection development panel")
    from .stream_runner import build_development_panel
    return build_development_panel(recipe)


def summarize_panel_reports(panel: Mapping[str, object], reports: Mapping[str, object]) -> Mapping[str, object]:
    """Aggregate per-case reports for either panel kind."""
    from .stream_runner import summarize_development_reports, summarize_standard_reports
    if panel.get("schema") == STANDARD_PANEL_SCHEMA:
        return summarize_standard_reports(panel, reports)
    return summarize_development_reports(panel, reports)


__all__ = ["GymUnavailable", "Recipe", "bind_validator", "build_panel", "generate_shard", "gym_validator",
           "is_standard_recipe_file", "load_recipe", "summarize_panel_reports"]
