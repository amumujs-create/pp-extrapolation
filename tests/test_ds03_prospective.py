import hashlib
import json

import h5py
import numpy as np
import pytest

from pp_extrapolation.ds03_prospective import (
    TEST_UNITS,
    causal_features,
    load_development,
    prior_admissibility_from_train,
    reveal_score,
)
from pp_extrapolation.paper_ppx import PPXCandidateEvidence, PPXContract, select_paper_ppx


def _small_h5(path):
    a = np.asarray([[1, 1, 0, 0], [1, 1, 0, 0], [1, 2, 0, 0]], float)
    with h5py.File(path, "w") as handle:
        handle["A_dev"] = a
        handle["W_dev"] = np.arange(12, dtype=float).reshape(3, 4)
        handle["X_s_dev"] = np.arange(42, dtype=float).reshape(3, 14)
        handle["Y_dev"] = np.asarray([[2], [2], [1]])
        # A sentinel with an incompatible type makes accidental reads obvious.
        handle["Y_test"] = np.asarray([b"DO_NOT_READ"])


def test_development_loader_never_opens_test_outcome(tmp_path, monkeypatch):
    path = tmp_path / "tiny.h5"
    _small_h5(path)
    original = h5py.File

    class GuardedFile:
        def __init__(self, *args, **kwargs):
            self.handle = original(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def __getitem__(self, key):
            assert key != "Y_test"
            assert not key.endswith("_test")
            return self.handle[key]

    monkeypatch.setattr(h5py, "File", GuardedFile)
    batch = load_development(path)
    assert batch.units.tolist() == [1, 1]
    assert batch.target.tolist() == [2, 1]


def test_causal_adapter_does_not_use_future_cycles(tmp_path):
    path = tmp_path / "tiny.h5"
    _small_h5(path)
    batch = load_development(path)
    before = causal_features(batch, "multiscale")
    batch.values[-1] += 10_000
    after = causal_features(batch, "multiscale")
    np.testing.assert_array_equal(before["x"][0], after["x"][0])
    assert not np.array_equal(before["x"][-1], after["x"][-1])


def test_reveal_rejects_selection_hash_before_hdf5_open(tmp_path, monkeypatch):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"not": "valid"}))
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen")
    h5 = tmp_path / "data.h5"
    h5.write_bytes(b"not hdf5")

    def forbidden(*args, **kwargs):
        raise AssertionError("HDF5 must not open before selection verification")

    monkeypatch.setattr(h5py, "File", forbidden)
    with pytest.raises(ValueError, match="selection JSON SHA-256 mismatch"):
        reveal_score(h5, protocol, selection, "0" * 64, tmp_path / "out.json")


def test_frozen_test_units_are_disjoint():
    assert TEST_UNITS == (10, 11, 12, 13, 14, 15)
    assert not set(TEST_UNITS) & set(range(1, 10))


def test_ds03_unknown_boundary_insufficient_prior_evidence_forces_fallback():
    train = {
        "y": np.asarray([2, 0, 3, 0], np.float32),
        "groups": np.asarray(["1", "1", "2", "2"]),
    }
    prior, audit = prior_admissibility_from_train(train)
    assert prior.known_boundary is False
    assert prior.complete_groups == 2
    assert prior.minimum_complete_groups_per_regime == 0
    assert prior.oof_prior_regret is None
    assert prior.oof_mode_stability is None
    assert audit["validation_units_used"] is False

    decision = select_paper_ppx(
        PPXContract(False, True, True, False, True, "direct_fallback"),
        prior,
        (
            PPXCandidateEvidence("direct_fallback", 10.0, 0.0, 1.0),
            PPXCandidateEvidence("unbounded", 1.0, 1.0, 0.1),
        ),
    )
    assert decision.executor == "direct_fallback"
    assert decision.approved is False
    assert decision.reason.startswith("prior rejected:")
