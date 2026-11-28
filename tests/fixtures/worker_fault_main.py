"""Run as a script (``__main__``) to reproduce production: spawn workers re-run this file as ``__mp_main__``.

The entry script is executed under the name ``__mp_main__`` in every process, as a spawn worker sees it, with its
inner job patched to raise an exception that cannot pickle. Usage: worker_fault_main.py ENTRY_PATH KIND
(KIND is ``label`` or ``run``). Prints one line per outcome for the test to assert.
"""

import multiprocessing
import runpy
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
ENTRY, KIND = Path(sys.argv[1]), sys.argv[2]
NS = runpy.run_path(str(ENTRY), run_name="__mp_main__")


class UnpicklableFault(Exception):
    def __reduce__(self):
        raise TypeError("import of module '_native' failed")


def _inner(item):
    raise UnpicklableFault("illegal action 7 accepted twice")


def pool_job(item):
    guard = NS["_guarded_label_job" if KIND == "label" else "_guarded_play_job"]
    guard.__globals__["_label_job" if KIND == "label" else "_play_job"] = _inner
    return guard(item)


if __name__ == "__main__":
    job = (SimpleNamespace(shard=3, local=5, run_id="run-abc") if KIND == "label"
           else SimpleNamespace(index=9, run_id="run-xyz"))
    pool = ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"))
    try:
        list(pool.map(pool_job, [(job, None), (job, None)]))
    except Exception as error:
        print(f"RAISED {type(error).__module__}.{type(error).__name__}")
        print(str(error))
    finally:
        pool.shutdown(cancel_futures=True)
