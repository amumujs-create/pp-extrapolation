"""Audit raw non-battery source structure without evaluating model outcomes."""
from pathlib import Path
import json,zipfile,io
import openpyxl
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'data/nonbattery_external'
z=zipfile.ZipFile(D/'PHMDC2019_Data.zip');out={}
# Inspect source training only; T7/T8 remain unused.
for i in range(1,7):
 name=next(n for n in z.namelist() if n.endswith(f'Description_T{i}.xlsx') and not n.startswith('__MACOSX'))
 ws=openpyxl.load_workbook(io.BytesIO(z.read(name)),data_only=True).active
 out[f'T{i}']={'sheet_rows':[list(r) for r in ws.values]}
(D/'phm2019_training_audit.json').write_text(json.dumps(out,indent=2,default=str))
print('Saved source training inspection tables; no test outcome evaluation.')
