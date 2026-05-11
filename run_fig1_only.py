"""Compatibility wrapper for the Fig. 5-1 revenue experiment."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from project.experiments.fig1 import main


if __name__ == "__main__":
    main()
