"""Extract official EOS summary. C=capacitance loss percent (Celaya et al.2012)."""
from pathlib import Path
import io,json,zipfile
import scipy.io
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data/nonbattery_external'
d=scipy.io.loadmat(io.BytesIO(zipfile.ZipFile(D/'cap2_tail.bin').read('EOS_DataSet.mat')))
s={f'CAP2_{j+1}':{'time':d['aging_time'].ravel().astype(float).tolist(),'health':(1-d['C'][:,j]/100).tolist()} for j in range(d['C'].shape[1])}
(D/'capacitor_stress2_screening.json').write_text(json.dumps(s,indent=2))
