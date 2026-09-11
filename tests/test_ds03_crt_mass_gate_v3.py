import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "ncmapss_ds03_crt_mass_gate_v3.py"
)
SPEC = importlib.util.spec_from_file_location("ncmapss_ds03_crt_mass_gate_v3", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
mass_gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mass_gate)


def _three_unit_rows():
    groups = np.repeat(np.asarray(["7", "8", "9"]), 20)
    cycles = np.tile(np.arange(20), 3)
    y = np.zeros(60)
    direct = np.ones(60)
    crt = np.concatenate([np.zeros(20), np.zeros(20), np.full(20, 1.1)])
    return groups, cycles, y, direct, crt


def test_continuous_gate_selects_largest_improving_safe_grid_mass():
    groups, cycles, y, direct, crt = _three_unit_rows()
    gate = mass_gate.fit_continuous_mass_gate(
        groups,
        cycles,
        y,
        direct,
        crt,
        shell_edges=(0.5, 1.0),
        alpha_grid=(0.0, 0.05, 0.1, 0.2, 0.3, 1.0),
    )

    shell = gate["shells"][0]
    assert shell["selected_alpha"] == 0.2
    evidence = {row["alpha"]: row for row in shell["validation_evidence"]}
    assert evidence[0.2]["safe"] is True
    assert evidence[0.3]["safe"] is False
    assert evidence[0.2]["unit_count"] == 3


def test_nonpositive_improvement_forces_exact_zero_mass():
    groups, cycles, y, direct, _ = _three_unit_rows()
    gate = mass_gate.fit_continuous_mass_gate(
        groups,
        cycles,
        y,
        direct,
        np.full_like(direct, 2.0),
        shell_edges=(0.5, 1.0),
        alpha_grid=(0.0, 0.05, 0.1),
    )

    assert gate["shells"][0]["selected_alpha"] == 0.0
    assert (
        gate["shells"][0]["selection_reason"]
        == "nonpositive_improvement_exact_fallback"
    )


def test_application_is_label_free_and_zero_mass_is_bitwise_direct():
    frozen_gate = {
        "shells": [
            {"lower_grade": 0.5, "upper_grade": 0.75, "selected_alpha": 0.0},
            {"lower_grade": 0.75, "upper_grade": 1.0, "selected_alpha": 0.2},
        ]
    }
    grades = np.asarray([0.1, 0.6, 0.8, 1.0])
    alpha = mass_gate.alpha_by_grade(grades, frozen_gate)
    direct = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    crt = direct + 10.0
    mixed = mass_gate.mix_exact_fallback(direct, crt, alpha)

    assert alpha.tolist() == [0.0, 0.0, 0.2, 0.2]
    assert np.array_equal(mixed[:2], direct[:2])
    np.testing.assert_allclose(mixed[2:], direct[2:] + 2.0)
