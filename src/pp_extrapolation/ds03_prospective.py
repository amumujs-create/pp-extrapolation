"""Two-phase, test-outcome-blind N-CMAPSS DS03 prospective runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch import nn

from .model import equal_group_weights, fit_feature_scale
from .paper_ppx import PPXCandidateEvidence, PPXContract, select_paper_ppx
from .regime_mixture import fit_latent_regime_pp, predict_latent_regime
from .transferability_gate import PriorEvidence

EXPECTED_SHA256 = "f67cc4bd0cf927f09eb8e0198bd62c777e1c6213177cfc46d81cc43bba52c333"
EXPECTED_SIZE = 3_693_395_776
TRAIN_UNITS = (1, 2, 3, 4, 5, 6)
VALIDATION_UNITS = (7, 8, 9)
TEST_UNITS = (10, 11, 12, 13, 14, 15)
SEEDS = (42, 43, 44, 45, 46)
OBSERVABLES = (
    "alt", "Mach", "TRA", "T2", "T24", "T30", "T48", "T50", "P15",
    "P2", "P21", "P24", "Ps30", "P40", "P50", "Nf", "Nc", "Wf",
)
ROUTES = ("direct_fallback", "basic", "multiscale")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unit_slices(a: np.ndarray) -> dict[int, tuple[int, int]]:
    units, starts = np.unique(a.astype(np.int64), return_index=True)
    ends = np.r_[starts[1:], len(a)]
    return {int(u): (int(s), int(e)) for u, s, e in zip(units, starts, ends)}


@dataclass
class CycleBatch:
    values: np.ndarray
    target: np.ndarray
    units: np.ndarray
    cycles: np.ndarray


def _aggregate_units(
    a: np.ndarray, w: np.ndarray, xs: np.ndarray, y: np.ndarray, units: tuple[int, ...]
) -> CycleBatch:
    parts: list[tuple[np.ndarray, float, int, float]] = []
    slices = _unit_slices(a[:, 0])
    for unit in units:
        if unit not in slices:
            continue
        start, end = slices[unit]
        unit_cycles = a[start:end, 1]
        raw = np.hstack((w[start:end], xs[start:end])).astype(np.float64)
        unit_y = y[start:end].reshape(-1)
        unique, cycle_starts = np.unique(unit_cycles, return_index=True)
        cycle_ends = np.r_[cycle_starts[1:], len(unit_cycles)]
        for cycle, lo, hi in zip(unique, cycle_starts, cycle_ends):
            block = raw[lo:hi]
            # Current-cycle aggregation only; temporal history is added later.
            values = np.r_[block.mean(axis=0), block.std(axis=0)]
            parts.append((values.astype(np.float32), float(unit_y[hi - 1]), unit, float(cycle)))
    return CycleBatch(
        np.stack([p[0] for p in parts]),
        np.asarray([p[1] for p in parts], np.float32),
        np.asarray([p[2] for p in parts], np.int32),
        np.asarray([p[3] for p in parts], np.float32),
    )


def load_development(path: Path) -> CycleBatch:
    """Load only development arrays; no ``*_test`` key is touched."""
    with h5py.File(path, "r") as handle:
        a = handle["A_dev"][:]
        w = handle["W_dev"][:]
        xs = handle["X_s_dev"][:]
        y = handle["Y_dev"][:]
    return _aggregate_units(a, w, xs, y, TRAIN_UNITS + VALIDATION_UNITS)


def load_revealed_test(path: Path) -> CycleBatch:
    """Load test observations and outcome together, only in reveal phase."""
    with h5py.File(path, "r") as handle:
        a = handle["A_test"][:]
        w = handle["W_test"][:]
        xs = handle["X_s_test"][:]
        y = handle["Y_test"][:]
    return _aggregate_units(a, w, xs, y, TEST_UNITS)


def causal_features(batch: CycleBatch, preset: str, window: int = 10) -> dict[str, np.ndarray]:
    rows, targets, groups, cycles = [], [], [], []
    for unit in np.unique(batch.units):
        idx = np.flatnonzero(batch.units == unit)
        for position, current in enumerate(idx):
            lo = max(0, position - window + 1)
            history = batch.values[idx[lo : position + 1]]
            coordinate = np.asarray([batch.cycles[current]], np.float32)
            if preset == "direct":
                blocks = [coordinate, batch.values[current]]
            elif preset == "basic":
                blocks = [coordinate, batch.values[current], history.mean(0),
                          (history[-1] - history[0]) / max(len(history) - 1, 1)]
            elif preset == "multiscale":
                blocks = [coordinate, batch.values[current]]
                for horizon in (3, 5, 10):
                    tail = history[-min(horizon, len(history)):]
                    blocks.extend((tail.mean(0), tail.std(0),
                                   (tail[-1] - tail[0]) / max(len(tail) - 1, 1)))
            else:
                raise ValueError(preset)
            rows.append(np.concatenate(blocks))
            targets.append(batch.target[current])
            groups.append(str(unit))
            cycles.append(batch.cycles[current])
    return {
        "x": np.asarray(rows, np.float32),
        "y": np.asarray(targets, np.float32),
        "groups": np.asarray(groups),
        "cycles": np.asarray(cycles, np.float32),
    }


class DirectNet(nn.Module):
    def __init__(self, width: int, dimension: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, width), nn.Tanh(),
                                 nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def _fit_direct(train: dict, validation: dict, seed: int, width: int = 32) -> dict:
    center, scale = fit_feature_scale(train["x"])
    cap = float(max(np.max(train["y"]), 1.0))
    tx = torch.tensor((train["x"] - center) / scale, dtype=torch.float32)
    ty = torch.tensor(train["y"] / cap, dtype=torch.float32)
    vx = torch.tensor((validation["x"] - center) / scale, dtype=torch.float32)
    torch.manual_seed(seed)
    model = DirectNet(width, tx.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    weights = torch.tensor(equal_group_weights(train["groups"]), dtype=torch.float32)
    rng, best, stale, state = np.random.default_rng(seed), math.inf, 0, None
    for _ in range(300):
        order = rng.permutation(len(tx))
        model.train()
        for start in range(0, len(tx), 256):
            ix = torch.tensor(order[start:start + 256])
            loss = torch.mean(weights[ix] * (model(tx[ix]) - ty[ix]).square())
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        prediction = _predict_direct({"model": model, "center": center, "scale": scale, "cap": cap}, validation["x"])
        mse = float(np.mean((prediction - validation["y"]) ** 2))
        if mse < best - 1e-9:
            best, stale = mse, 0
            state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if stale > 50:
            break
    assert state is not None
    model.load_state_dict(state)
    return {"model": model, "center": center, "scale": scale, "cap": cap, "validation_mse": best}


def _predict_direct(fit: dict, x: np.ndarray) -> np.ndarray:
    fit["model"].eval()
    z = torch.tensor((x - fit["center"]) / fit["scale"], dtype=torch.float32)
    with torch.no_grad():
        return np.clip(fit["model"](z).numpy() * fit["cap"], 0, fit["cap"])


def _unit_evidence(y: np.ndarray, prediction: np.ndarray, fallback: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    ratios = []
    for group in np.unique(groups):
        mask = groups == group
        rmse = math.sqrt(float(np.mean((y[mask] - prediction[mask]) ** 2)))
        base = math.sqrt(float(np.mean((y[mask] - fallback[mask]) ** 2)))
        ratios.append(rmse / max(base, 1e-12))
    return float(np.mean(np.asarray(ratios) < 1.0)), float(max(ratios))


def _manifest(rows: dict[str, dict], protocol: Path) -> dict:
    row_parts = []
    for split in ("train", "validation"):
        row_parts.append(np.column_stack((rows[split]["groups"], rows[split]["cycles"])).astype("U"))
    row_hash = sha256_bytes("\n".join(",".join(row) for part in row_parts for row in part).encode())
    return {
        "adapter": "cycle_mean_std_then_causal_window_v1",
        "observable_columns": list(OBSERVABLES),
        "train_units": list(TRAIN_UNITS),
        "validation_units": list(VALIDATION_UNITS),
        "test_units": list(TEST_UNITS),
        "rows": {key: int(len(value["y"])) for key, value in rows.items()},
        "row_manifest_sha256": row_hash,
        "protocol_sha256": sha256_file(protocol),
        "raw_expected_sha256": EXPECTED_SHA256,
        "raw_expected_size": EXPECTED_SIZE,
        "seeds": list(SEEDS),
    }


def prior_admissibility_from_train(train: dict) -> tuple[PriorEvidence, dict[str, Any]]:
    """Build prior evidence independently of held-out validation evidence.

    DS03 declares neither a failure boundary nor an observed failure-mode
    regime.  A train group is counted as complete only when its source RUL
    reaches zero.  Because per-regime completeness and group-disjoint latent
    mode stability cannot be established under the frozen information
    contract, those OOF fields remain unavailable and the paper gate must
    select the neural fallback.
    """
    complete = 0
    for group in np.unique(train["groups"]):
        target = np.asarray(train["y"])[train["groups"] == group]
        complete += int(len(target) > 0 and float(np.min(target)) <= 0.0)
    evidence = PriorEvidence(
        known_boundary=False,
        complete_groups=complete,
        minimum_complete_groups_per_regime=0,
        oof_prior_regret=None,
        oof_mode_stability=None,
    )
    audit = {
        **asdict(evidence),
        "source": "train units only",
        "validation_units_used": False,
        "status": "insufficient evidence",
        "reason": (
            "no declared boundary or observed regime; group-disjoint OOF "
            "per-regime completeness and latent-mode stability unavailable"
        ),
    }
    return evidence, audit


def prepare_select(h5_path: Path, protocol: Path, selection_path: Path) -> dict:
    if selection_path.exists():
        raise FileExistsError(f"immutable selection already exists: {selection_path}")
    if h5_path.stat().st_size != EXPECTED_SIZE:
        raise ValueError("raw file size differs from frozen protocol")
    dev = load_development(h5_path)
    prepared = {}
    for preset in ("direct", "basic", "multiscale"):
        all_rows = causal_features(dev, preset)
        prepared[preset] = {
            "train": {k: v[np.isin(all_rows["groups"], np.asarray(TRAIN_UNITS, str))]
                      for k, v in all_rows.items()},
            "validation": {k: v[np.isin(all_rows["groups"], np.asarray(VALIDATION_UNITS, str))]
                           for k, v in all_rows.items()},
        }
    fits, predictions, evidence = {}, {}, []
    for route in ROUTES:
        preset = "direct" if route == "direct_fallback" else route
        train, validation = prepared[preset]["train"], prepared[preset]["validation"]
        route_fits, route_predictions = [], []
        for seed in SEEDS:
            if route == "direct_fallback":
                fit = _fit_direct(train, validation, seed)
                pred = _predict_direct(fit, validation["x"])
            else:
                fit = fit_latent_regime_pp(
                    train, validation, seed=seed, max_epochs=300, patience=70,
                    separation_weight=0.0 if route == "basic" else 0.01,
                    gate_weight=0.0, width=24,
                )
                pred = predict_latent_regime(fit, validation["x"])
            route_fits.append(fit); route_predictions.append(pred)
        fits[route] = route_fits
        predictions[route] = np.mean(route_predictions, axis=0)
    yv, gv = prepared["direct"]["validation"]["y"], prepared["direct"]["validation"]["groups"]
    # Every preset preserves identical cycle row order.
    fallback = predictions["direct_fallback"]
    for route in ROUTES:
        wins, worst = _unit_evidence(yv, predictions[route], fallback, gv)
        evidence.append(PPXCandidateEvidence(
            route if route == "direct_fallback" else ("unbounded" if route == "basic" else "dual_scale"),
            float(np.mean((yv - predictions[route]) ** 2)), wins, worst,
        ))
    prior_evidence, prior_audit = prior_admissibility_from_train(prepared["direct"]["train"])
    decision = select_paper_ppx(
        PPXContract(False, True, True, False, True, "direct_fallback"),
        prior_evidence,
        tuple(evidence),
    )
    selected_route = {"direct_fallback": "direct_fallback", "unbounded": "basic", "dual_scale": "multiscale"}[decision.executor]
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = selection_path.with_suffix(".models.pt")
    torch.save({"fits": fits, "feature_presets": {r: ("direct" if r == "direct_fallback" else r) for r in ROUTES}}, model_path)
    manifest = _manifest(
        {"train": prepared["direct"]["train"], "validation": prepared["direct"]["validation"]},
        protocol,
    )
    manifest_hash = sha256_bytes(canonical_bytes(manifest))
    payload = {
        "schema": "ncmapss-ds03-ppx-prospective-selection-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "prepare-select",
        "test_outcome_read": False,
        "selected_route": selected_route,
        "selected_executor": decision.executor,
        "approved": decision.approved,
        "reason": decision.reason,
        "configuration": {
            "window_cycles": 10, "cycle_aggregate": ["mean", "std"],
            "seeds": list(SEEDS), "basic_separation": 0.0, "multiscale_separation": 0.01,
        },
        "prior_admissibility_evidence": prior_audit,
        "validation_evidence": [asdict(item) for item in evidence],
        "manifest": manifest,
        "manifest_sha256": manifest_hash,
        "model_artifact": model_path.name,
        "model_artifact_sha256": sha256_file(model_path),
    }
    selection_path.write_bytes(canonical_bytes(payload))
    return {**payload, "selection_sha256": sha256_file(selection_path)}


def reveal_score(h5_path: Path, protocol: Path, selection_path: Path, expected_selection_sha256: str,
                 output_path: Path) -> dict:
    actual_selection_hash = sha256_file(selection_path)
    if actual_selection_hash != expected_selection_sha256:
        raise ValueError("selection JSON SHA-256 mismatch; reveal refused")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["manifest_sha256"] != sha256_bytes(canonical_bytes(selection["manifest"])):
        raise ValueError("embedded manifest hash mismatch; reveal refused")
    if selection["manifest"]["protocol_sha256"] != sha256_file(protocol):
        raise ValueError("protocol changed after selection; reveal refused")
    model_path = selection_path.with_name(selection["model_artifact"])
    if selection["model_artifact_sha256"] != sha256_file(model_path):
        raise ValueError("model artifact hash mismatch; reveal refused")
    # Integrity is checked only after the immutable selection has been verified.
    if h5_path.stat().st_size != EXPECTED_SIZE or sha256_file(h5_path) != EXPECTED_SHA256:
        raise ValueError("raw file differs from frozen protocol")
    revealed_at = datetime.now(timezone.utc).isoformat()
    test = load_revealed_test(h5_path)  # First semantic Y_test read.
    route = selection["selected_route"]
    preset = "direct" if route == "direct_fallback" else route
    rows = causal_features(test, preset)
    bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    predictions = []
    for fit in bundle["fits"][route]:
        predictions.append(_predict_direct(fit, rows["x"]) if route == "direct_fallback"
                           else predict_latent_regime(fit, rows["x"]))
    prediction = np.mean(predictions, axis=0)
    unit_rmse = {}
    for unit in np.unique(rows["groups"]):
        mask = rows["groups"] == unit
        unit_rmse[str(unit)] = math.sqrt(float(np.mean((rows["y"][mask] - prediction[mask]) ** 2)))
    y = rows["y"]
    result = {
        "schema": "ncmapss-ds03-ppx-prospective-reveal-v1",
        "phase": "reveal-score",
        "reveal_timestamp_utc": revealed_at,
        "selection_sha256": actual_selection_hash,
        "selected_route": route,
        "n_rows": int(len(y)),
        "pooled_rmse": math.sqrt(float(np.mean((y - prediction) ** 2))),
        "pooled_r2": float(1 - np.sum((y - prediction) ** 2) / np.sum((y - np.mean(y)) ** 2)),
        "physical_unit_macro_rmse": float(np.mean(list(unit_rmse.values()))),
        "physical_unit_rmse": unit_rmse,
    }
    if output_path.exists():
        raise FileExistsError(f"reveal artifact already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_bytes(result))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="phase", required=True)
    prepare = sub.add_parser("prepare-select")
    reveal = sub.add_parser("reveal-score")
    for item in (prepare, reveal):
        item.add_argument("--h5", type=Path, required=True)
        item.add_argument("--protocol", type=Path, required=True)
        item.add_argument("--selection", type=Path, required=True)
    reveal.add_argument("--selection-sha256", required=True)
    reveal.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.phase == "prepare-select":
        result = prepare_select(args.h5, args.protocol, args.selection)
    else:
        result = reveal_score(args.h5, args.protocol, args.selection, args.selection_sha256, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))

