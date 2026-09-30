"""Precompute every simulation the app shows, so pages load without running regressions.

Run at build time (Render's build command runs it after installing requirements):
    python precompute.py
The output file is git-ignored. If it's missing or was built from different source files,
the app ignores it and computes results on demand.
"""
from __future__ import annotations

import hashlib
import itertools
import pickle
import time
import warnings
from pathlib import Path

from design_lab import DIMENSIONS, Design, simulate
from simulator import SEED, load_demo_data
from targeting_signals import load_signals

HERE = Path(__file__).parent
OUTPUT = HERE / "precomputed.pkl"
SOURCES = ("design_lab.py", "simulator.py", "targeting_signals.py")


def source_fingerprint() -> str:
    """Changes whenever the simulation code changes, so a stale file is never used."""
    digest = hashlib.sha256()
    for name in SOURCES:
        digest.update((HERE / name).read_bytes())
    return digest.hexdigest()


def main() -> None:
    warnings.filterwarnings("ignore")
    start = time.time()
    designs = {combo: simulate(Design(*combo), SEED) for combo in itertools.product(*[list(options) for options in DIMENSIONS.values()])}
    payload = {"fingerprint": source_fingerprint(), "seed": SEED, "designs": designs,
               "demo": load_demo_data(SEED), "signals": load_signals(SEED)}
    OUTPUT.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    print(f"Precomputed {len(designs)} designs, worked examples and shopper signals in {time.time() - start:.0f}s -> {OUTPUT.name} ({OUTPUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
