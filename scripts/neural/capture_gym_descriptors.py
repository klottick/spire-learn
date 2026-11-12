"""Capture authenticated public Gym descriptors as test fixtures; no gameplay.

Run inside the offline runtime container with the verified release overlay on
``sys.path``. Constructs environments, reads their public specs, verifies each
descriptor fingerprint against its Gym domain, writes canonical JSON and
closes every environment. ``reset`` and ``step`` are patched to fail.
"""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

# Gym fingerprint domains (spire-env combat_policy.rs, run_schema_standard.rs).
DOMAINS = {
    "common": b"spire-gym/combat-policy-view/v1\0",
    "standard-combat": b"spire-gym/combat-view/standard/v1\0",
    "standard-run": b"spire-gym/run-schema/standard/v1\0",
}
IRONCLAD_ENTRY = {
    "id": "capture-ironclad-a0", "character": "ironclad", "ascension": 0,
    "encounter": "cultist", "hp": 80, "max_hp": 80, "gold": 99,
    "deck": [{"card": card, "upgrades": 0} for card in ["strike"] * 5 + ["defend"] * 4 + ["bash"]],
    "relics": ["burning_blood"], "potions": [None] * 3,
}


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def verify(name: str, descriptor: dict) -> dict:
    body = {key: value for key, value in descriptor.items() if key != "fingerprint"}
    actual = hashlib.sha256(DOMAINS[name] + canonical(body)).hexdigest()
    if actual != descriptor["fingerprint"]:
        raise SystemExit(f"{name}: descriptor body does not match its fingerprint")
    return descriptor


def _refuse(*args, **kwargs):
    raise AssertionError("descriptor capture must not reset or step")


def capture(output: Path, revision: str) -> dict:
    from spire_lab import _native
    from spire_lab.gymnasium import (
        IroncladCombatVectorEnv, RunEnv, RunVectorEnv, StandardCombatVectorEnv,
    )

    info = dict(_native.build_info())
    if info.get("revision") != revision or info.get("dirty") is not False:
        raise SystemExit(f"unexpected build identity: {info}")
    classes = (IroncladCombatVectorEnv, RunEnv, RunVectorEnv, StandardCombatVectorEnv)
    patches = [patch.object(cls, method, _refuse) for cls in classes for method in ("reset", "step")]
    for item in patches:
        item.start()
    files: dict[str, dict] = {}
    try:
        env = StandardCombatVectorEnv({
            "catalog_schema_id": "spire-gym-standard-combat-catalog-v1",
            "profile_id": "standard-combat-v1", "entries": [IRONCLAD_ENTRY],
        }, 1)
        try:
            files["standard-combat"] = verify("standard-combat", env.observation_spec())
            files["standard-combat-environment"] = env.environment_spec()
            files["standard-combat-catalog"] = env.catalog_descriptor()
        finally:
            env.close()
        env = RunVectorEnv([{"character": "ironclad", "ascension": 0}], profile="standard")
        try:
            files["standard-run"] = verify("standard-run", env.observation_spec())
            files["standard-run-environment"] = env.environment_spec()
        finally:
            env.close()
        # RunVectorEnv exposes no combat-view spec; the scalar Run does, without a reset.
        env = RunEnv("ironclad", 0, profile="standard")
        try:
            if verify("standard-run", env.observation_spec()) != files["standard-run"]:
                raise SystemExit("scalar and vector Standard Run descriptors differ")
            view = verify("standard-combat", env.combat_observation_spec())
            if view != files["standard-combat"]:
                raise SystemExit("Standard Run combat view differs from the standalone view")
        finally:
            env.close()
        env = IroncladCombatVectorEnv(1, {"entries": [{
            "id": "capture-common", "encounter": "cultist", "hp": 80, "max_hp": 80,
            "deck": [{"card": "strike", "upgrades": 0}], "relics": [], "potions": [None] * 3}]})
        try:
            files["common"] = verify("common", env.observation_spec())
        finally:
            env.close()
    finally:
        for item in reversed(patches):
            item.stop()
    short = revision[:7]
    output.mkdir(parents=True, exist_ok=True)
    for name, value in files.items():
        (output / f"{name}-{short}.json").write_bytes(canonical(value) + b"\n")
    return {"build_info": info,
            "fingerprints": {name: files[name]["fingerprint"]
                             for name in ("common", "standard-combat", "standard-run")}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    print(json.dumps(capture(args.output, args.revision), sort_keys=True))


if __name__ == "__main__":
    main()
