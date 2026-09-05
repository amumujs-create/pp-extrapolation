"""Local adapter for the original NASA archive; does not redistribute source data."""
import argparse
import sys
from pathlib import Path
import numpy as np
from pp_extrapolation.temporal import DegradationContract,causal_history

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--research-root',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();sys.path.insert(0,str(Path(args.research_root).resolve()))
    from run_affine_tail_external_nasa_health_v2 import prepare_folds
    from run_affine_tail_external_three import nasa_cells
    cells=nasa_cells();folds,audit=prepare_folds()
    histories={c:causal_history(f.cycle.to_numpy(),f.capacity.to_numpy(),DegradationContract(1.4,'decreasing')) for c,f in cells.items()}
    archive={}
    for i,f in enumerate(folds):
        for part in ('train','validation','test'):
            values={k:[] for k in ('x','prior','reliability','q','disagreement')}
            for y,c,old_x in zip(f[part]['y'],f[part]['groups'],f[part]['x']):
                # Labels used only to recover identity of already fixed split rows.
                # They do not enter causal_history, prior construction or features.
                idx=np.flatnonzero(cells[c].RUL.to_numpy()==y)
                assert len(idx)==1
                j=int(idx[0]);assert np.isclose(histories[c]['q'][j],old_x[0])
                for k in values:values[k].append(histories[c][k][j])
            for k,v in values.items():archive[f'f{i}_{part}_{k}']=np.asarray(v,dtype='float32')
            archive[f'f{i}_{part}_y']=f[part]['y']
            archive[f'f{i}_{part}_groups']=f[part]['groups'].astype(str)
    np.savez_compressed(args.output,**archive)
    print('Exact original test rows:',audit['n_test_rows'])
if __name__=='__main__':main()
