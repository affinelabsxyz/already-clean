"""Run the frozen RLLEAK probe (frozen/rlleak_probe.py) unchanged, with your own adapter folder.

The probe keeps the adapter folder as a constant, RUNS_DIR, which pointed at the original machine. This wrapper sets it
from the RUNS_DIR environment variable and then calls the probe's own main(). The probe still records its own sha256,
so results from this wrapper carry the frozen hash 0fbfc4d7...
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frozen"))
import rlleak_probe  # noqa: E402

rlleak_probe.RUNS_DIR = os.environ.get("RUNS_DIR", "runs_rl")
rlleak_probe.main()
