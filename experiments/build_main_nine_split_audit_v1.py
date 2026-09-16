#!/usr/bin/env python3
"""Build Main-9 split audit JSON with unified coordinate units and cutoff values.

Writes:
  - results/main_nine_split_audit.json (local, gitignored)
  - reproducibility/main_nine_split_audit.json (Git-tracked)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "experiments"),
    str(ROOT.parent / "ca-css-ncmapss"),
]

OUT_LOCAL = ROOT / "results/main_nine_split_audit.json"
OUT_TRACKED = ROOT / "reproducibility/main_nine_split_audit.json"


def _part(units, rows: int, condition: str) -> dict[str, Any]:
    return {"units": units, "rows": int(rows), "condition": condition}


def _battery_row(name: str) -> dict[str, Any]:
    from distance_uncertainty_pp import prepare_battery
    from pae_boundary_realdata import prepare_dataset

    raw, audit = prepare_dataset(name)
    tr, va, te = prepare_battery(raw)
    ep_tr = raw["train"]["x"][:, -1, 0]
    q25 = float(np.quantile(ep_tr, 0.25))
    test_cut = float(audit["early_health_min"])
    unit = "mAh" if name == "sunwoda" else "normalized health"
    q_label = "train-q25 (train endpoint health)"
    test_label = "early_health_min (support cutoff on source cells)"

    if name == "sunwoda":
        unit_ids = {
            "train": [1025, 2025, 3025, 4025, 5025, 6025, 7025],
            "val": [8025, 9025],
            "test": [
                10035,
                11035,
                12035,
                13035,
                14035,
                15035,
                16035,
                17035,
                18035,
            ],
        }
    elif name == "rwth":
        unit_ids = {
            "train": list(range(2, 25)),
            "val": list(range(25, 33)),
            "test": list(range(33, 41)),
        }
    else:
        unit_ids = {
            "train": list(range(1, 19)),
            "val": list(range(19, 25)),
            "test": list(range(25, 33)),
        }

    fmt = lambda v: f"{v:.4f} mAh" if name == "sunwoda" else f"{v:.4f}"
    train_c = f"endpoint health > {fmt(q25)} ({q_label})"
    val_c = f"endpoint health < {fmt(q25)} ({q_label})"
    test_c = f"point health < {fmt(test_cut)} ({test_label})"

    return {
        "name": {"sunwoda": "Sunwoda", "rwth": "RWTH", "mich": "MICH"}[name],
        "coordinate": {
            "hull_axis": "8-step window endpoint health (feature x[:,−1,0])",
            "unit": unit,
            "note": "1D hull-out uses the same endpoint coordinate as PP benchmark adapter",
        },
        "cutoffs": {
            "train_val_endpoint_q25": {
                "quantile": 0.25,
                "population": "train split window endpoints",
                "value": q25,
                "unit": unit,
                "train": f"> {q25}",
                "val": f"< {q25}",
            },
            "test_support_early_health_min": {
                "value": test_cut,
                "unit": unit,
                "test": f"< {test_cut}",
                "derivation": (
                    "min health among eligible early-life windows on train∪val units; "
                    f"boundary={audit['boundary']}, threshold={audit['threshold']:.4f}"
                ),
            },
        },
        "train": _part(len(unit_ids["train"]), len(tr["y"]), train_c),
        "val": _part(len(unit_ids["val"]), len(va["y"]), val_c),
        "test": _part(len(unit_ids["test"]), len(te["y"]), test_c),
        "rationale": {
            "sunwoda": (
                "25°C train/val cells vs 35°C test cells. Train/val split by train-q25 on "
                "8-step endpoint health; test cells use early_health_min (980.7 mAh) for "
                "deeper tail + temperature shift."
            ),
            "rwth": (
                "Unit IDs 2–24 / 25–32 / 33–40. Train/val use train-q25 on normalized "
                "endpoint health; test uses early_health_min 0.9000 on unseen IDs."
            ),
            "mich": (
                "Unit IDs 1–18 / 19–24 / 25–32 with published MICH life labels. Same q25 vs "
                "early_health_min split as RWTH; test cutoff 0.8708."
            ),
        }[name],
        "extra": {
            "boundary": float(audit["boundary"]),
            "threshold": float(audit["threshold"]),
            "unit_ids": unit_ids,
        },
    }


def _hust() -> dict[str, Any]:
    from run_affine_tail_external_three import prepare_hust

    tr, va, te, meta = prepare_hust()
    audit = meta["audit"]
    cutoff = float(audit["train_capacity_min_ah"])
    return {
        "name": "HUST",
        "coordinate": {"hull_axis": "capacity_ah", "unit": "Ah"},
        "cutoffs": {
            "train_support_min": {
                "value": cutoff,
                "unit": "Ah",
                "train": f"≥ {cutoff:.4f}",
                "val": f"< {cutoff:.4f}",
                "test": f"< {cutoff:.4f}",
                "note": "train_min = min capacity among train rows after protocol filter",
            }
        },
        "train": _part(
            audit["n_train_units"],
            audit["n_train_rows"],
            f"protocol 1–6, capacity ≥ 1.05 Ah; hull axis capacity ≥ {cutoff:.4f} Ah",
        ),
        "val": _part(
            audit["n_validation_units"],
            audit["n_validation_rows"],
            f"protocol 7–8, capacity < {cutoff:.4f} Ah",
        ),
        "test": _part(
            audit["n_test_units"],
            audit["n_test_rows"],
            f"protocol 9–10, capacity < {cutoff:.4f} Ah",
        ),
        "rationale": (
            "Physical units are charging protocols. Train keeps high-capacity regimes; "
            "val/test use later protocols with capacity below the train support floor "
            f"({cutoff:.4f} Ah), giving 100% 1D hull-out on capacity_ah."
        ),
        "extra": {},
    }


def _virkler() -> dict[str, Any]:
    from run_affine_tail_external_three import prepare_virkler

    tr, va, te, _ = prepare_virkler()
    cutoff = 33.0
    return {
        "name": "Virkler",
        "coordinate": {"hull_axis": "crack_length_mm", "unit": "mm"},
        "cutoffs": {
            "train_support_max": {
                "value": cutoff,
                "unit": "mm",
                "train": f"≤ {cutoff:.1f}",
                "val": f"> {cutoff:.1f}",
                "test": f"> {cutoff:.1f}",
            }
        },
        "train": _part(48, len(tr["y"]), f"crack length ≤ {cutoff:.1f} mm"),
        "val": _part(10, len(va["y"]), f"crack length > {cutoff:.1f} mm"),
        "test": _part(10, len(te["y"]), f"crack length > {cutoff:.1f} mm"),
        "rationale": (
            "Specimens are unit-disjoint (seed 42). Train stays inside the declared crack "
            "support; val/test specimens extrapolate beyond 33 mm, the train maximum."
        ),
        "extra": {},
    }


def _nasa() -> dict[str, Any]:
    from run_affine_tail_external_nasa_health_v2 import prepare_folds

    folds, _ = prepare_folds()
    train_rows = sum(len(f["train"]["y"]) for f in folds)
    val_rows = sum(len(f["validation"]["y"]) for f in folds)
    test_rows = sum(len(f["test"]["y"]) for f in folds)
    fold_cutoffs = []
    for fold in folds:
        train_min = float(fold["train"]["x"][:, 0].min())
        fold_cutoffs.append(
            {
                "test_cell": fold["test_cell"],
                "validation_cell": fold["validation_cell"],
                "fold_train_min_health_phi": train_min,
            }
        )
    return {
        "name": "NASA PCoE battery",
        "coordinate": {
            "hull_axis": "health_phi = (capacity−1.4)/(initial−1.4)",
            "unit": "dimensionless",
        },
        "cutoffs": {
            "train_floor": {
                "value": 0.5,
                "unit": "health_phi",
                "train": "≥ 0.5",
            },
            "fold_train_min": {
                "unit": "health_phi",
                "per_fold": fold_cutoffs,
                "val": "health_phi < fold_train_min",
                "test": "health_phi < fold_train_min",
            },
        },
        "train": _part("2/fold×4", train_rows, "health_phi ≥ 0.5"),
        "val": _part("1/fold×4", val_rows, "health_phi < fold_train_min (see cutoffs.fold_train_min)"),
        "test": _part("1/fold×4", test_rows, "health_phi < fold_train_min (4 folds pooled)"),
        "rationale": (
            "Four-fold LOOCV over cells B0005–B0018. Train keeps phi≥0.5; val/test cells "
            "use health below that fold's train minimum on the 1D phi axis."
        ),
        "extra": {"cells": ["B0005", "B0006", "B0007", "B0018"]},
    }


def _matr2019() -> dict[str, Any]:
    audit = json.loads(
        (ROOT / "results/matr_2019_latent_confirmatory/results.json").read_text()
    )["pretest"]
    boundary = float(audit["actual_boundary"])
    return {
        "name": "MATR 2019",
        "coordinate": {"hull_axis": "Q_last (8-step window endpoint)", "unit": "normalized Q"},
        "cutoffs": {
            "train_q25_boundary": {
                "quantile": 0.25,
                "population": "train cells 0–25 endpoint Q",
                "value": boundary,
                "unit": "Q_last",
                "train": f"> {boundary:.4f}",
                "val": f"< {boundary:.4f}",
                "test": f"< {boundary:.4f}",
            }
        },
        "train": _part(26, 17003, f"Q_last > {boundary:.4f} (train-q25)"),
        "val": _part(8, 1306, f"Q_last < {boundary:.4f}"),
        "test": _part(10, 2235, f"Q_last < {boundary:.4f}"),
        "rationale": "Locked 45-cell batch: q25 on train endpoints defines one boundary for all splits.",
        "extra": {"boundary": boundary},
    }


def _matr_batch2() -> dict[str, Any]:
    from matr_batch2_confirmatory import endpoints, load_cells, make_rows

    cells = load_cells()
    q25 = float(np.quantile(endpoints(cells, range(30)), 0.25))
    tr0 = make_rows(cells, range(30), q25, train=True)
    boundary = float(tr0["coordinate"].min())
    tr, va, te = (
        make_rows(cells, range(30), boundary, train=True),
        make_rows(cells, range(30, 39), boundary),
        make_rows(cells, range(39, 48), boundary),
    )
    return {
        "name": "MATR batch 2",
        "coordinate": {"hull_axis": "Q_last", "unit": "normalized Q"},
        "cutoffs": {
            "train_q25_boundary": {
                "quantile": 0.25,
                "value": boundary,
                "unit": "Q_last",
                "train": f"> {boundary:.4f}",
                "val": f"< {boundary:.4f}",
                "test": f"< {boundary:.4f}",
                "note": f"q25 raw={q25:.4f}; boundary=min train coordinate={boundary:.4f}",
            }
        },
        "train": _part(30, len(tr["y"]), f"Q_last > {boundary:.4f} (train-q25)"),
        "val": _part(9, len(va["y"]), f"Q_last < {boundary:.4f}"),
        "test": _part(9, len(te["y"]), f"Q_last < {boundary:.4f}"),
        "rationale": "Same q25 boundary logic as MATR 2019 on the locked batch-2 cohort.",
        "extra": {"q25": q25, "boundary": boundary},
    }


def _ncmapss() -> dict[str, Any]:
    from ncmapss_tra_quantile_split import make_tra_hard_split

    split = make_tra_hard_split(
        (ROOT / "data/N-CMAPSS_DS02-006.h5").resolve(),
        max_windows_per_unit=1500,
        random_seed=42,
    )
    th = split.thresholds
    q70, q90 = float(th.q70), float(th.q90)
    return {
        "name": "N-CMAPSS DS02-006",
        "coordinate": {"hull_axis": "TRA at window end", "unit": "percent throttle"},
        "cutoffs": {
            "train_q70": {"value": q70, "train": f"TRA ≤ {q70:.2f}"},
            "val_band": {"low": q70, "high": q90, "val": f"{q70:.2f} < TRA ≤ {q90:.2f}"},
            "test_q90": {"value": q90, "test": f"TRA > {q90:.2f}"},
        },
        "train": _part([2, 5, 10, 16, 18], len(split.train.y), f"TRA ≤ {q70:.2f}"),
        "val": _part(
            [2, 5, 10, 16, 18, 20],
            len(split.val.y),
            f"{q70:.2f} < TRA ≤ {q90:.2f}",
        ),
        "test": _part([11, 14, 15], len(split.test.y), f"TRA > {q90:.2f}"),
        "rationale": (
            "Unseen engines at high TRA; quantiles q70/q90 fit on train-pool TRA only. "
            "Main test is hard_extrap (159 rows), not RUL≤50 contrast band."
        ),
        "extra": {
            "note": "RUL≤50 band is contrast only",
            "q70": q70,
            "q90": q90,
        },
    }


def build() -> list[dict[str, Any]]:
    return [
        _hust(),
        _virkler(),
        _nasa(),
        _battery_row("sunwoda"),
        _battery_row("rwth"),
        _battery_row("mich"),
        _matr2019(),
        _matr_batch2(),
        _ncmapss(),
    ]


def write_markdown(rows: list[dict[str, Any]]) -> None:
    path = ROOT / "MAIN_NINE_SPLIT_KO.md"
    text = path.read_text(encoding="utf-8")
    marker = "## Cutoff map (Git audit, unified units)"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n"
    block = [
        marker,
        "",
        "작성: 박진서 · 재생성: `python experiments/build_main_nine_split_audit_v1.py`",
        "",
        "배터리 3종은 **train/val = train-q25 (endpoint)** 와 **test = early_health_min** 이 "
        "다른 두 cutoff를 쓴다. 표에는 둘 다 실수로 적었다.",
        "",
        "| Dataset | Coordinate | Train | Val | Test | q / label → value |",
        "|---------|------------|-------|-----|------|-------------------|",
    ]
    for row in rows:
        c = row["coordinate"]
        axis = c.get("hull_axis", c.get("label", ""))
        unit = c.get("unit", "")
        cut = row.get("cutoffs", {})
        mapping = "; ".join(
            f"{k}={json.dumps(v, ensure_ascii=False)}"
            for k, v in cut.items()
        )
        block.append(
            f"| {row['name']} | {axis} ({unit}) | {row['train']['condition']} | "
            f"{row['val']['condition']} | {row['test']['condition']} | {mapping} |"
        )
    block.extend(["", "### Split rationale (one line each)", ""])
    for row in rows:
        block.append(f"- **{row['name']}:** {row['rationale']}")
    block.append("")
    path.write_text(text + "\n".join(block) + "\n", encoding="utf-8")


def main() -> None:
    rows = build()
    payload = json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
    OUT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    OUT_TRACKED.parent.mkdir(parents=True, exist_ok=True)
    OUT_LOCAL.write_text(payload, encoding="utf-8")
    OUT_TRACKED.write_text(payload, encoding="utf-8")
    write_markdown(rows)
    print(f"Wrote {OUT_TRACKED}")


if __name__ == "__main__":
    main()
