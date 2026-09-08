#!/usr/bin/env python3
"""Final locked official Zn-ion test cohort for deterministic RBF-regime PP."""
from pathlib import Path
import znion_rbf_regime_pp_confirmation_v2 as frozen

ROOT=Path(__file__).resolve().parents[1]
frozen.TEST_DIR=ROOT/'data/znion_rbf_regime_confirmation_v3'
frozen.OUT=ROOT/'results/znion_rbf_regime_confirmation_v3'
frozen.PRIOR_DIRS=(*frozen.PRIOR_DIRS,'znion_rbf_regime_confirmation_v2')
frozen.EXPECTED=(
 'ZN-coin_205-1_20231205230230_07_4',
 'ZN-coin_209-1_20231205230248_07_7',
 'ZN-coin_402-3_20231209225844_01_3',
 'ZN-coin_406-2_20231209231604_08_6',
 'ZN-coin_406-3_20231209231637_08_7',
 'ZN-coin_412-1_20231209232958_09_7',
 'ZN-coin_413-1_20231209233202_06_2',
 'ZN-coin_446-1_20240104212538_07_2',
)

if __name__=='__main__':frozen.main()
