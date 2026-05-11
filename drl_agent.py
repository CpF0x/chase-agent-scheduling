"""Compatibility wrapper for FL-DRL inference classes."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from after_project.algorithms.fldrl import DDQNNetwork, FLDRLInference

__all__ = ["DDQNNetwork", "FLDRLInference"]
