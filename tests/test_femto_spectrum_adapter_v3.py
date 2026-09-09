import numpy as np

from femto_spectrum_adapter_v3 import ENVELOPE_BINS, SPECTRUM_BINS, binned_log_power, spectral_features


def test_spectrum_dimension_and_frequency_peak():
    sampling = 25600.0
    time = np.arange(2560) / sampling
    signal = np.sin(2*np.pi*1000*time)
    value = binned_log_power(signal, sampling/2, SPECTRUM_BINS)
    assert value.shape == (SPECTRUM_BINS,)
    assert 2 <= int(np.argmax(value)) <= 3
    assert spectral_features(signal).shape == (SPECTRUM_BINS+ENVELOPE_BINS,)
