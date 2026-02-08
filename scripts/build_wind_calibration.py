#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.wind_calibration import build_and_save_calibration_table


def main():
    out_path = build_and_save_calibration_table()
    print(f"Wrote wind calibration table: {out_path}")


if __name__ == "__main__":
    main()
