#!/usr/bin/env python3
"""N-CMAPSS DS03 frozen prospective two-phase CLI."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pp_extrapolation.ds03_prospective import main


if __name__ == "__main__":
    main()
