#!/usr/bin/env python3
"""Run the frozen v4 unit-route gate unchanged on prospective Na-ion group 5."""
from pathlib import Path
import naion_unit_route_gate_v4_eval as frozen

ROOT=Path(__file__).resolve().parents[1]
frozen.DATA=ROOT/'data/naion_external_replication'
frozen.OUT=ROOT/'results/naion_unit_route_gate_replication'
frozen.EXPECTED=('270040-5-1-39.csv','270040-5-2-38.csv','270040-5-3-37.csv','270040-5-4-36.csv',
                 '270040-5-5-35.csv','270040-5-6-34.csv','270040-5-7-33.csv','270040-5-8-32.csv')

if __name__=='__main__':
    frozen.main()
