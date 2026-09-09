from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments"))
from femto_sensor_adapter_v2 import RAW, OFFICIAL, load


def test_correct_channels_and_endpoint_labels():
    d=load()
    raw=np.loadtxt(RAW/"Learning_set/Bearing1_1/acc_00001.csv",delimiter=",")
    row=np.flatnonzero((d["bearing"]=="Bearing1_1") & (d["recording_index"]==1))[0]
    assert np.isclose(d["sensor"][row,0],np.sqrt(np.mean(raw[:,4]**2)),rtol=1e-5)
    assert np.isclose(d["sensor"][row,7],np.sqrt(np.mean(raw[:,5]**2)),rtol=1e-5)
    for bearing,target in OFFICIAL.items():
        ix=np.flatnonzero(d["bearing"]==bearing)
        endpoint=ix[np.argmax(d["recording_index"][ix])]
        assert d["y"][endpoint]==target


def test_no_forbidden_full_test_units():
    d=load()
    assert set(np.unique(d["bearing"][d["role"]=="test"]))==set(OFFICIAL)
