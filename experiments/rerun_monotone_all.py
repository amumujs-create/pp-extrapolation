#!/usr/bin/env python3
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT/'.benchmark_deps'),str(ROOT.parent/'ca-css-ncmapss')]
import torch
import extrapolation_competitors_matr as bench
from extrapolation_competitors_all import datasets,OUT
from pp_extrapolation import select_affine_initialization
bench.OUT=OUT
torch.set_num_threads(2);path=OUT/'results.json';payload=json.load(open(path))
for name,parts in datasets().items():
 aff=select_affine_initialization(parts[0],parts[1]);payload['datasets'][name]['monotone']=bench.neural_model('monotone',parts,aff,name+'_');path.write_text(json.dumps(payload,indent=2)+'\n');print('UPDATED',name,flush=True)
# NASA and N-CMAPSS are reconstructed by the special runner helpers only when needed;
# their first coordinate already has positive train correlation in the saved audit.
