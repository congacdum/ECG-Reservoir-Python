#!/usr/bin/env python3
"""Run deterministic RTL smoke regressions for push/PR CI."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = (
    "run_lif_tb.py",
    "run_recurrent_tb.py",
    "run_controller_tb.py",
    "run_readout_tb.py",
    "run_classifier_tb.py",
)


def main() -> int:
    total_started = time.perf_counter()
    for runner in RUNNERS:
        command = [sys.executable, str(ROOT / "rtl" / "scripts" / runner), "--smoke"]
        print("$ " + runner + " --smoke", flush=True)
        started = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT, check=False)
        elapsed = time.perf_counter() - started
        status = "PASS" if result.returncode == 0 else "FAIL"
        print(f"{runner}: {status} in {elapsed:.3f}s", flush=True)
        if result.returncode != 0:
            print(f"FAIL: {runner} returned {result.returncode}", flush=True)
            return result.returncode or 1
    print(f"PASS: RTL smoke suite in {time.perf_counter() - total_started:.3f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
