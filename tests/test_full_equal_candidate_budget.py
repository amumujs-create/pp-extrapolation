import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "full_equal_candidate_budget.py"
SPEC = importlib.util.spec_from_file_location("full_equal_candidate_budget", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def test_all_tunable_models_have_30_distinct_candidates():
    assert set(runner.CANDIDATES) == set(runner.MODELS)
    for candidates in runner.CANDIDATES.values():
        assert len(candidates) == 30
        assert len({json.dumps(row, sort_keys=True) for row in candidates}) == 30


def test_protocol_records_ppx_as_heterogeneous_reference():
    assert set(runner.SETTINGS) == {
        "hust", "virkler", "nasa", "sunwoda", "rwth", "matr",
        "matr_batch2", "ncmapss", "mich",
    }
    assert set(runner.PPX_ARTIFACTS) == set(runner.SETTINGS)


def test_split_hash_is_row_order_sensitive():
    parts = runner.synthetic_parts()
    hashes = runner.split_hashes(parts)
    changed = tuple({k: np.asarray(v).copy() for k, v in part.items()} for part in parts)
    changed[2]["y"] = changed[2]["y"][::-1].copy()
    assert hashes["test"]["y"] != runner.split_hashes(changed)["test"]["y"]


def test_import_rejects_non_30_candidate_artifact(tmp_path):
    source = tmp_path / "results.json"
    source.write_text(json.dumps({"datasets": {"smoke": {"vrex": {"search": [{}]}}}}))
    with pytest.raises(ValueError, match="30-candidate"):
        runner.import_completed("smoke", "vrex", runner.synthetic_parts(), source, tmp_path / "out")


def test_short_smoke_plain_mlp(tmp_path):
    row = runner.run_one("smoke", "plain_mlp", runner.synthetic_parts(),
                         tmp_path / "smoke", limit_candidates=2)
    assert row["status"] == "smoke"
    assert row["candidate_count"] == 2
    assert row["contract_candidate_count"] == 30
    assert row["search_seed"] == 42
    assert row["refit_seeds"] == [42, 43, 44, 45, 46]
    artifact = np.load(tmp_path / "smoke" / "predictions.npz")
    assert artifact["prediction"].shape == (5, 16)
    assert np.array_equal(artifact["y"], runner.synthetic_parts()[2]["y"])
