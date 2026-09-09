import numpy as np
from xjtu_spectrum_adapter_v2 import log_power_bins,spectrum
def test_xjtu_spectrum_shape_and_peak():
 t=np.arange(32768)/25600.;x=np.sin(2*np.pi*1000*t);value=log_power_bins(x,12800,32)
 assert value.shape==(32,) and 2<=int(np.argmax(value))<=3
 assert spectrum(x).shape==(48,)
