import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from function_prior_train_diagnosis import ridge_predict,score
from component_transfer_screen import source_episode_indices


def test_signed_health_cutoff_excludes_later_rows():
    rows={'x':np.column_stack((np.tile(np.linspace(.1,1.,10),6),np.zeros((60,10)))),
          'groups':np.repeat(np.arange(6),10)}
    donor,query,cut,_=source_episode_indices(rows,'mich',0,.6)
    early=rows['x'][:,0]>=-cut
    assert not np.any(early&query)
    assert rows['x'][early,0].min()>rows['x'][query,0].max()
    assert not set(rows['groups'][donor])&set(rows['groups'][query])


def test_ridge_query_batch_does_not_change_normalization():
    x=np.arange(20,dtype=float).reshape(10,2);y=np.column_stack((x[:,0],x[:,0]**2))
    one=ridge_predict(x,y,np.array([[3.,4.]]))
    both=ridge_predict(x,y,np.array([[3.,4.],[1e8,1e8]]))
    np.testing.assert_allclose(one,both[:1])


def test_unit_score_not_row_weighted():
    y=np.zeros(4);p=np.array([1.,1.,1.,3.]);g=np.array([0,0,0,1])
    assert score(y,p,g)['unit_mse']==5.
