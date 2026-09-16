#!/usr/bin/env python3
"""Export Main-9 extrapolation appendix table and tracked hull audit JSON.

Reads ignored `results/all_dataset_hull_audit_v1/results.json` (regenerate via
`experiments/all_dataset_hull_audit.py`) and `results/main_nine_split_audit.json`.
Writes git-tracked copies under `reproducibility/` plus `MAIN_NINE_EXTRAPOLATION_APPENDIX_KO.md`.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HULL_SRC = ROOT / "results/all_dataset_hull_audit_v1/results.json"
SPLIT_SRC = ROOT / "results/main_nine_split_audit.json"
HULL_DST = ROOT / "reproducibility/all_dataset_hull_audit_v1/results.json"
SPLIT_DST = ROOT / "reproducibility/main_nine_split_audit.json"
REPORT = ROOT / "MAIN_NINE_EXTRAPOLATION_APPENDIX_KO.md"

# hull JSON key, split audit `name`, domain label for appendix
MAIN_NINE = (
    ("hust", "HUST", "Battery (protocol shift)"),
    ("virkler", "Virkler", "Fatigue crack growth"),
    ("nasa", "NASA PCoE battery", "Battery (cell LOO)"),
    ("sunwoda", "Sunwoda", "Battery (unseen cell)"),
    ("rwth", "RWTH", "Battery (unseen cell)"),
    ("mich", "MICH", "Battery (unseen cell)"),
    ("matr", "MATR 2019", "Battery (capacity fade)"),
    ("matr_batch2", "MATR batch 2", "Battery (capacity fade)"),
    ("ncmapss", "N-CMAPSS DS02-006", "Aero engine (unseen engine × TRA)"),
)


def _load_split_audit() -> dict[str, dict]:
    rows = json.loads(SPLIT_SRC.read_text(encoding="utf-8"))
    return {row["name"]: row for row in rows}


def _test_block(hull_key: str, datasets: dict) -> dict:
    if hull_key == "nasa":
        folds = datasets["nasa"]["folds"]
        weights = [f["n"]["test"] for f in folds]
        total = sum(weights)

        def wavg(field: str) -> float:
            return sum(f["test"][field] * w for f, w in zip(folds, weights)) / total

        def wavg_nn() -> float:
            return (
                sum(f["test"]["full_feature_nn"]["median"] * w for f, w in zip(folds, weights))
                / total
            )

        return {
            "outside_fraction": wavg("outside_fraction"),
            "distance_median": wavg("distance_median"),
            "distance_max": max(f["test"]["distance_max"] for f in folds),
            "full_feature_nn_median": wavg_nn(),
            "n_test": total,
            "n_folds": len(folds),
        }
    entry = datasets[hull_key]
    test = entry["test"]
    return {
        "outside_fraction": test["outside_fraction"],
        "distance_median": test["distance_median"],
        "distance_max": test["distance_max"],
        "full_feature_nn_median": test["full_feature_nn"]["median"],
        "n_test": entry["n"]["test"],
        "n_folds": 1,
    }


def _split_cell(part: dict) -> str:
    units = part.get("units", "?")
    rows = part.get("rows", "?")
    cond = part.get("condition", "")
    return f"{units} units · {rows} rows · {cond}"


def build_rows(hull: dict, splits: dict[str, dict]) -> list[dict]:
    out = []
    for hull_key, split_name, domain in MAIN_NINE:
        split = splits[split_name]
        test = _test_block(hull_key, hull["datasets"])
        out.append(
            {
                "domain": domain,
                "dataset": split_name,
                "axis": split["coord"],
                "train": _split_cell(split["train"]),
                "val": _split_cell(split["val"]),
                "test": _split_cell(split["test"]),
                "outside_pct": round(100.0 * test["outside_fraction"], 1),
                "hull_distance_median": round(test["distance_median"], 3),
                "hull_distance_max": round(test["distance_max"], 3),
                "full_feature_nn_median": round(test["full_feature_nn_median"], 3),
                "n_test_rows": test["n_test"],
            }
        )
    return out


def write_report(hull: dict, rows: list[dict]) -> None:
    lines = [
        "# Main 9 — Extrapolation distance appendix",
        "",
        "작성: 박진서",
        "",
        "PPT Fig. 7 schematic의 수치 근거. **Test split** 기준.",
        "",
        "- **Outside-support %** = `outside_fraction` × 100 (1D 선언 좌표, train convex hull 밖)",
        "- **Extrapolation distance** = `distance_median` (train SD 단위 hull 밖 거리)",
        "- **Full-feature distance** = train 표준화 full feature 공간 NN median 거리",
        "",
        f"좌표 정의: `{hull['coordinate']}`",
        "",
        "재생성:",
        "",
        "```bash",
        "cd pp-extrapolation",
        "PYTHONPATH=src:experiments:../ca-css-ncmapss python experiments/all_dataset_hull_audit.py",
        "PYTHONPATH=src:experiments:../ca-css-ncmapss python experiments/export_main_nine_extrapolation_appendix_v1.py",
        "```",
        "",
        "Git-tracked JSON: `reproducibility/all_dataset_hull_audit_v1/results.json`",
        "",
        "| Domain | Dataset | Axis (1D hull) | Train | Val | Test | Outside % | Hull dist. (med.) | Full-NN dist. |",
        "|--------|---------|----------------|-------|-----|------|----------:|------------------:|--------------:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['domain']} | {row['dataset']} | {row['axis']} | "
            f"{row['train']} | {row['val']} | {row['test']} | "
            f"{row['outside_pct']:.1f} | {row['hull_distance_median']:.3f} | "
            f"{row['full_feature_nn_median']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Compact (PPT right panel)",
            "",
            "| Dataset | Outside % | Extrapolation distance | Full-feature distance |",
            "|---------|----------:|-----------------------:|----------------------:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['outside_pct']:.1f} | "
            f"{row['hull_distance_median']:.3f} | {row['full_feature_nn_median']:.3f} |"
        )
    lines.append("")
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not HULL_SRC.exists():
        raise FileNotFoundError(
            f"Run all_dataset_hull_audit.py first; missing {HULL_SRC}"
        )
    if not SPLIT_SRC.exists():
        raise FileNotFoundError(f"Missing split audit {SPLIT_SRC}")

    hull = json.loads(HULL_SRC.read_text(encoding="utf-8"))
    splits = _load_split_audit()
    rows = build_rows(hull, splits)

    HULL_DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HULL_SRC, HULL_DST)
    shutil.copy2(SPLIT_SRC, SPLIT_DST)
    write_report(hull, rows)
    payload = {
        "owner": "박진서",
        "source_hull_audit": str(HULL_SRC.relative_to(ROOT)),
        "coordinate": hull["coordinate"],
        "main_nine": rows,
    }
    (HULL_DST.parent / "main_nine_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {REPORT}")
    print(f"Wrote {HULL_DST}")
    print(f"Wrote {SPLIT_DST}")


if __name__ == "__main__":
    main()
