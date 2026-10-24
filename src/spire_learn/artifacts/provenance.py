"""Read-only source and installed-distribution provenance collection."""

from importlib import metadata
from pathlib import Path
import subprocess


def _git(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    try:
        top = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if Path(top).resolve() != root:
            return {"revision": None, "head_state": "unavailable", "dirty": None}
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            capture_output=True, text=True,
        )
        if head.returncode:
            return {"revision": None, "head_state": "unborn", "dirty": None}
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout
        return {"revision": head.stdout.strip(), "head_state": "present", "dirty": bool(status)}
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "head_state": "unavailable", "dirty": None}


def collect_provenance(*, learner_root: Path, gym_checkout: Path | None) -> dict[str, object]:
    learner = _git(learner_root)
    gym = None
    if gym_checkout is not None:
        state = _git(gym_checkout)
        gym = {"revision": state["revision"], "dirty": state["dirty"]}
    distributions = {}
    for name in ("spire-learn", "spire-lab", "numpy", "gymnasium", "torch", "torchrl"):
        try:
            distributions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            distributions[name] = None
    return {"learner": learner, "gym_checkout": gym, "distributions": distributions, "native_build_info": None}
