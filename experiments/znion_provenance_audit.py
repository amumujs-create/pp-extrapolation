#!/usr/bin/env python3
"""Audit exact-file and normalized-trajectory duplicates across local Zn-ion cohorts."""
import hashlib,json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import znion_bq_confirmatory as data
OUT=ROOT/'results/znion_provenance_audit'
DIRECTORIES=('znion_bq_confirmatory/train','znion_bq_confirmatory/validation','znion_bq_confirmatory/test','znion_regime_scale_confirmation_v1','znion_rbf_regime_confirmation_v2','znion_rbf_regime_confirmation_v3')

def grouped(values):
 groups=defaultdict(list)
 for path,digest in values:groups[digest].append(path)
 return [paths for paths in groups.values() if len(paths)>1]

def main():
 paths=sorted({p for name in DIRECTORIES for p in (ROOT/'data'/name).glob('*.xlsx')});files=[];trajectories=[];failures={}
 for path in paths:
  label=str(path.relative_to(ROOT));files.append((label,hashlib.sha256(path.read_bytes()).hexdigest()))
  try:
   cell=data.read_cell(path)
   if cell is None:failures[label]='no observed EOL or missing cycle-10 nominal';continue
   packed=np.column_stack((cell['cycle'],np.round(cell['capacity'],10))).astype('<f8').tobytes();trajectories.append((label,hashlib.sha256(packed).hexdigest()))
  except Exception as error:failures[label]=f'{type(error).__name__}: {error}'
 result={'status':'retrospective provenance audit','n_files':len(paths),'n_eligible_trajectories':len(trajectories),'exact_file_duplicate_groups':grouped(files),'normalized_eol_trajectory_duplicate_groups':grouped(trajectories),'read_failures':failures,'fingerprint':'SHA-256 of little-endian float64 [reset cycle, SOH rounded to 10 decimals] through first observed 80%-SOH crossing','limitation':'A matching fingerprint establishes processed trajectory duplication; a non-match does not establish independent experimental provenance.'}
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
