"""Infer PP-X contract flags from train-only dataset structure.

The inferred contract restricts which executors may enter validation. It does
not use validation or test labels, test-batch statistics, or test performance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

from .paper_ppx import PPXContract

_TIME_KEYS = ("cycles", "cycle", "times", "time", "progress", "coordinate")
_REGIME_KEYS = ("regime", "regimes", "regime_id", "condition", "protocol")
_GROUP_KEYS = ("groups", "units", "unit", "engine", "cell")


@dataclass(frozen=True)
class DatasetStructure:
    """Train-only structural facts used to infer a typed PP-X contract."""

    boundary_value: float | None = None
    unit_ids: np.ndarray | None = None
    progression_coordinate: np.ndarray | None = None
    regime_ids: np.ndarray | None = None
    support_coordinate: np.ndarray | None = None
    min_history_points: int = 2
    min_units_for_support: int = 2
    dual_scale_min_heterogeneity: float = 0.5
    fallback: Literal["direct_fallback", "persistence_fallback"] = "direct_fallback"


@dataclass(frozen=True)
class ContractInference:
    contract: PPXContract
    support_heterogeneity: float
    reasons: dict[str, str]


@dataclass(frozen=True)
class OptionalExecutorDetection:
    """Train-only on/off for optional residual routes."""

    history: bool
    dual_scale: bool
    transport: bool
    support_heterogeneity: float
    reasons: dict[str, str]


def _as_1d(values: np.ndarray | None, name: str) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a nonempty 1D array")
    return array


def _ordered_progression_audit(
    unit_ids: np.ndarray,
    coordinate: np.ndarray,
) -> tuple[bool, str]:
    unique_units = np.unique(unit_ids)
    if unique_units.size == 0:
        return False, "unit id가 없다"
    multi = 0
    monotonic = 0
    for unit in unique_units:
        mask = unit_ids == unit
        count = int(mask.sum())
        if count < 2:
            continue
        multi += 1
        coords = coordinate[mask]
        diffs = np.diff(coords.astype(np.float64))
        if np.all(diffs > 0) or np.all(diffs < 0):
            monotonic += 1
    if multi == 0:
        return False, "unit마다 시점이 2개 미만이라 순서 열화 좌표를 만들 수 없다"
    if monotonic < multi:
        return False, "일부 unit에서 진행 좌표가 단조가 아니다"
    return True, f"{multi}개 unit에서 단조 진행 좌표 확인"


def _history_eligible_units(
    unit_ids: np.ndarray,
    *,
    min_history_points: int,
) -> tuple[int, int]:
    counts = {unit: int((unit_ids == unit).sum()) for unit in np.unique(unit_ids)}
    eligible = sum(count >= min_history_points for count in counts.values())
    return eligible, len(counts)


def _support_heterogeneity_score(
    unit_ids: np.ndarray,
    coordinate: np.ndarray,
) -> float:
    unique_units = np.unique(unit_ids)
    if unique_units.size < 2:
        return 0.0
    unit_means = np.asarray(
        [float(np.mean(coordinate[unit_ids == unit])) for unit in unique_units],
        dtype=np.float64,
    )
    total = float(np.std(coordinate.astype(np.float64)))
    if total <= 1e-12:
        return 0.0
    return float(np.clip(np.std(unit_means) / total, 0.0, 1.0))


def _unit_counts(unit_ids: np.ndarray) -> np.ndarray:
    return np.asarray(
        [int((unit_ids == unit).sum()) for unit in np.unique(unit_ids)],
        dtype=np.float64,
    )


def _time_varying_within_unit(
    unit_ids: np.ndarray,
    regime_ids: np.ndarray,
    *,
    max_purity: float = 0.8,
) -> bool:
    for unit in np.unique(unit_ids):
        labels = regime_ids[unit_ids == unit]
        _, counts = np.unique(labels, return_counts=True)
        if float(counts.max()) / float(len(labels)) < max_purity:
            return True
    return False


def detect_history_executor(
    unit_ids: np.ndarray | None,
    coordinate: np.ndarray | None,
    *,
    regime_ids: np.ndarray | None = None,
    min_history_points: int = 2,
    min_length_ratio: float = 1.5,
) -> tuple[bool, str]:
    """Turn the history executor on only when one time scale is not enough.

    Equal-length regular trajectories keep history as a feature, not as the
    selected executor. That matches Final: NASA/N-CMAPSS on, Sunwoda/RWTH off.
    """
    if unit_ids is None or coordinate is None:
        return False, "unit id 또는 진행 좌표가 없어 history를 끈다"
    if unit_ids.shape[0] != coordinate.shape[0]:
        return False, "unit id와 진행 좌표 길이가 달라 history를 끈다"
    ordered, reason = _ordered_progression_audit(unit_ids, coordinate)
    if not ordered:
        return False, f"history 불가: {reason}"
    eligible, total = _history_eligible_units(
        unit_ids, min_history_points=min_history_points
    )
    if eligible <= 0:
        return False, f"과거 이력을 만들 unit이 없다 (최소 {min_history_points}개 시점 필요)"
    counts = _unit_counts(unit_ids)
    variable_length = bool(counts.min() > 0 and counts.max() / counts.min() >= min_length_ratio)
    time_varying = False
    if regime_ids is not None and regime_ids.shape[0] == unit_ids.shape[0]:
        time_varying = _time_varying_within_unit(unit_ids, regime_ids)
    if not variable_length and not time_varying:
        return False, (
            "unit마다 시점 수가 같고 운전조건이 고정이라 "
            "history executor가 아니라 일반 궤적 feature다"
        )
    why = []
    if variable_length:
        why.append(f"시점 수 비 {counts.max():.0f}/{counts.min():.0f}")
    if time_varying:
        why.append("unit 안에서 운전조건이 바뀜")
    return True, f"{eligible}/{total}개 unit, {', '.join(why)}"


def detect_dual_scale_executor(
    unit_ids: np.ndarray | None,
    support_coordinate: np.ndarray | None,
    *,
    min_units: int = 2,
    min_heterogeneity: float = 0.5,
) -> tuple[bool, float, str]:
    """Turn dual-scale on only when train units sit in different support regions."""
    if unit_ids is None or support_coordinate is None:
        return False, 0.0, "support 좌표가 없어 dual-scale을 끈다"
    support_array = np.asarray(support_coordinate, dtype=np.float64)
    if support_array.ndim == 2:
        if support_array.shape[0] != unit_ids.shape[0]:
            return False, 0.0, "support 좌표 길이가 row 수와 달라 dual-scale을 끈다"
        support_array = support_array[:, 0]
    elif support_array.ndim != 1 or support_array.shape[0] != unit_ids.shape[0]:
        return False, 0.0, "support 좌표 shape가 맞지 않아 dual-scale을 끈다"
    unique_units = np.unique(unit_ids)
    if unique_units.size < min_units:
        return False, 0.0, f"unit이 {min_units}개 미만이라 dual-scale을 끈다"
    score = _support_heterogeneity_score(unit_ids, support_array)
    if score < min_heterogeneity:
        return (
            False,
            score,
            f"unit 위치 이질성 {score:.3f} < {min_heterogeneity:.2f} 이라 dual-scale을 끈다",
        )
    return (
        True,
        score,
        f"unit {unique_units.size}개, 이질성 {score:.3f} ≥ {min_heterogeneity:.2f}",
    )


def detect_regime_transport_executor(
    unit_ids: np.ndarray | None,
    regime_ids: np.ndarray | None,
    *,
    min_regimes: int = 2,
    min_units_per_regime: int = 2,
    min_unit_purity: float = 0.8,
) -> tuple[bool, str]:
    """Turn transport on only for a unit-stable regime label.

    Time-varying operating conditions such as TRA fail the purity check.
    A charging protocol or batch label that stays with the unit can pass.
    """
    if unit_ids is None or regime_ids is None:
        return False, "unit별 regime 식별자가 없어 transport를 끈다"
    if unit_ids.shape[0] != regime_ids.shape[0]:
        return False, "regime 길이가 row 수와 달라 transport를 끈다"
    labels = np.asarray(regime_ids)
    if np.unique(labels).size < min_regimes:
        return False, "train regime가 1개뿐이라 transport를 끈다"
    majority: dict[object, object] = {}
    for unit in np.unique(unit_ids):
        unit_labels = labels[unit_ids == unit]
        values, counts = np.unique(unit_labels, return_counts=True)
        purity = float(counts.max()) / float(len(unit_labels))
        if purity < min_unit_purity:
            return False, (
                f"unit {unit}의 regime이 시점마다 바뀌어 transport를 끈다 "
                f"(순도 {purity:.2f} < {min_unit_purity:.2f})"
            )
        majority[unit] = values[int(np.argmax(counts))]
    units_per_regime: dict[object, int] = {}
    for regime in majority.values():
        units_per_regime[regime] = units_per_regime.get(regime, 0) + 1
    qualified = sum(count >= min_units_per_regime for count in units_per_regime.values())
    if qualified < min_regimes:
        return False, (
            f"unit이 {min_units_per_regime}개 이상인 regime이 {qualified}개뿐이라 "
            "group-LOO transport를 끈다"
        )
    return True, (
        f"unit 고정 regime {len(units_per_regime)}개, "
        f"각 regime에 unit ≥{min_units_per_regime}"
    )


def _is_discrete_label(values: np.ndarray, *, max_unique: int = 16) -> bool:
    if values.ndim != 1 or values.size == 0:
        return False
    unique = np.unique(values)
    if unique.size < 2 or unique.size > max_unique:
        return False
    if np.issubdtype(values.dtype, np.floating):
        rounded = np.round(values)
        if not np.allclose(values, rounded, atol=1e-6):
            return False
    return True


def _column_is_progression(unit_ids: np.ndarray, values: np.ndarray) -> bool:
    ordered, _ = _ordered_progression_audit(unit_ids, values)
    return ordered


def _optional_requested_key(
    train: Mapping[str, object],
    requested: str | None,
    *,
    kind: str,
) -> str | None:
    """Normalize an optional user column name; blank means auto-detect."""
    if requested is None or not requested.strip():
        return None
    key = requested.strip()
    if key not in train:
        raise ValueError(f"requested {kind} column {key!r} is not present in train")
    return key


def _infer_row_count(train: Mapping[str, object]) -> int | None:
    """Infer row count from target or feature containers, not candidate IDs."""
    for key in ("y", "target"):
        if key in train:
            array = np.asarray(train[key])
            if array.ndim >= 1 and array.shape[0] > 0:
                return int(array.shape[0])
    if "x" in train:
        array = np.asarray(train["x"])
        if array.ndim >= 1 and array.shape[0] > 0:
            return int(array.shape[0])
    lengths = [
        int(np.asarray(value).shape[0])
        for value in train.values()
        if np.asarray(value).ndim == 1 and np.asarray(value).size > 0
    ]
    if lengths and len(set(lengths)) == 1:
        return lengths[0]
    return None


def _contiguous_run_count(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    return int(1 + np.sum(values[1:] != values[:-1]))


def infer_group_ids(
    train: Mapping[str, object],
    *,
    group_key: str | None = None,
) -> tuple[np.ndarray, str, str]:
    """Resolve physical-unit IDs from user input, standard names, or data.

    Automatic inference only considers top-level 1D repeated-label columns.
    Each label must form one contiguous row block; this avoids mistaking a
    repeated cycle/time column for a unit ID. If the best candidate is not
    unique, callers must provide ``group_key`` instead of accepting a guess.
    """
    explicit = _optional_requested_key(train, group_key, kind="group")
    if explicit is not None:
        return np.asarray(train[explicit]), explicit, f"사용자 지정 unit 열 {explicit}"

    named = _pick_first_key(train, _GROUP_KEYS)
    if named is not None:
        return np.asarray(train[named]), named, f"표준 이름 unit 열 {named}"

    row_count = _infer_row_count(train)
    if row_count is None:
        raise ValueError("cannot infer row count; provide group_key and train rows")

    skip = set(_TIME_KEYS) | set(_REGIME_KEYS) | {"x", "y", "target"}
    candidates: list[tuple[str, np.ndarray, int]] = []
    for key, value in train.items():
        if key in skip:
            continue
        array = np.asarray(value)
        if array.ndim != 1 or array.shape[0] != row_count:
            continue
        unique, counts = np.unique(array, return_counts=True)
        if unique.size < 2 or unique.size >= row_count or counts.min() < 2:
            continue
        if np.issubdtype(array.dtype, np.floating):
            if not np.allclose(array, np.round(array), atol=1e-6):
                continue
        if _contiguous_run_count(array) != unique.size:
            continue
        candidates.append((key, array, int(unique.size)))

    if not candidates:
        raise ValueError(
            "unit/group column could not be inferred; provide group_key explicitly"
        )
    candidates.sort(key=lambda value: (-value[2], value[0]))
    if len(candidates) > 1 and candidates[0][2] == candidates[1][2]:
        names = ", ".join(key for key, _, _ in candidates)
        raise ValueError(
            f"unit/group column is ambiguous ({names}); provide group_key explicitly"
        )
    key, array, _ = candidates[0]
    return array, key, f"반복 ID·연속 블록 검사로 unit 열 {key} 자동 선택"


def infer_time_coordinate(
    train: Mapping[str, object],
    unit_ids: np.ndarray,
    *,
    time_key: str | None = None,
    group_key: str | None = None,
    regime_key: str | None = None,
) -> tuple[np.ndarray | None, str | None, str]:
    """Resolve time from user input, semantic names, or one unambiguous column.

    ``x`` feature positions are deliberately excluded.  When unnamed train
    columns contain zero or multiple monotonic candidates, time remains
    unresolved and temporal executors stay closed.
    """
    explicit = _optional_requested_key(train, time_key, kind="time")
    if explicit is not None:
        return np.asarray(train[explicit]), explicit, f"사용자 지정 시간 열 {explicit}"

    named = _pick_first_key(train, _TIME_KEYS)
    if named is not None:
        return np.asarray(train[named]), named, f"표준 이름 시간 열 {named}"

    skip = {
        key
        for key in (group_key, regime_key, "x", "y", "target")
        if key is not None
    }
    candidates: list[tuple[str, np.ndarray]] = []
    for key, value in train.items():
        if key in skip:
            continue
        array = np.asarray(value)
        if array.ndim != 1 or array.shape[0] != unit_ids.shape[0]:
            continue
        if not np.issubdtype(array.dtype, np.number):
            continue
        numeric = array.astype(np.float64, copy=False)
        if not np.all(np.isfinite(numeric)):
            continue
        if _column_is_progression(unit_ids, numeric):
            candidates.append((key, array))

    if len(candidates) == 1:
        key, array = candidates[0]
        return array, key, f"train 단조성 검사로 시간 열 {key} 자동 선택"
    if len(candidates) > 1:
        names = ", ".join(key for key, _ in candidates)
        return None, None, f"단조 시간 후보가 여러 개라 미확정: {names}"
    return None, None, "사용자/표준 시간 열이 없고 단일 단조 후보도 없음"


def infer_regime_ids(
    train: Mapping[str, object],
    unit_ids: np.ndarray,
    *,
    regime_key: str | None = None,
    time_key: str | None = None,
    group_key: str | None = None,
) -> tuple[np.ndarray | None, str]:
    """Find a regime label even when no ``regime`` column is declared.

    Named keys win. Otherwise a discrete column that stays with the unit
    (protocol/batch) is used for transport. A discrete column that changes
    inside the unit (TRA-like) is returned only as a time-varying signal.
    Cycle/health progressions are ignored.
    """
    regime_key = _optional_requested_key(train, regime_key, kind="regime")
    if regime_key is not None:
        return np.asarray(train[regime_key]), f"이름 있는 열 {regime_key}"

    skip = {key for key in (group_key, time_key, "y", "target") if key}
    candidates: list[tuple[str, np.ndarray]] = []
    for key, value in train.items():
        if key in skip or key == "x":
            continue
        array = np.asarray(value)
        if array.ndim == 1 and array.shape[0] == unit_ids.shape[0] and _is_discrete_label(array):
            if time_key is None or not np.array_equal(array, np.asarray(train[time_key])):
                is_progression = False
                if np.issubdtype(array.dtype, np.number):
                    is_progression = _column_is_progression(
                        unit_ids, array.astype(np.float64, copy=False)
                    )
                if not is_progression:
                    candidates.append((key, array))
    if "x" in train:
        features = np.asarray(train["x"])
        if features.ndim == 2 and features.shape[0] == unit_ids.shape[0]:
            progression = None
            if time_key is not None:
                progression = np.asarray(train[time_key])
            for index in range(features.shape[1]):
                column = features[:, index]
                if not _is_discrete_label(column):
                    continue
                if progression is not None and np.allclose(column, progression.astype(np.float64)):
                    continue
                if _column_is_progression(unit_ids, column):
                    continue
                candidates.append((f"x[{index}]", column))

    for name, array in candidates:
        ok, _ = detect_regime_transport_executor(unit_ids, array)
        if ok:
            return array, f"unit 고정 이산열 {name}에서 찾음"
    for name, array in candidates:
        if _time_varying_within_unit(unit_ids, array):
            return array, f"unit 안 변동 이산열 {name}에서 찾음"
    return None, "레짐 열도 없고, unit 고정/변동 이산 특징도 없음"


def detect_optional_executors(
    train: Mapping[str, object],
    *,
    time_key: str | None = None,
    regime_key: str | None = None,
    group_key: str | None = None,
    min_history_points: int = 2,
    min_units_for_support: int = 2,
    dual_scale_min_heterogeneity: float = 0.5,
) -> OptionalExecutorDetection:
    """Detect history, dual-scale, and transport from train rows only.

    User-supplied time/regime keys win. Blank values trigger train-only
    detection; time is accepted only from a semantic name or one unambiguous
    monotonic top-level column. Feature position is never interpreted as time.
    The first feature remains only the default support coordinate used by the
    dual-scale audit.
    """
    unit_ids, group_key, group_source = infer_group_ids(
        train, group_key=group_key
    )
    regime_key = _optional_requested_key(train, regime_key, kind="regime")
    regime_key = regime_key or _pick_first_key(train, _REGIME_KEYS)
    progression, resolved_time_key, time_source = infer_time_coordinate(
        train,
        unit_ids,
        time_key=time_key,
        group_key=group_key,
        regime_key=regime_key,
    )
    features = None
    if "x" in train:
        features = np.asarray(train["x"], dtype=np.float64)
        if features.ndim != 2 or features.shape[0] != unit_ids.shape[0]:
            features = None
    support = None if features is None else features[:, 0]
    if support is None:
        support = progression
    regime_ids, regime_source = infer_regime_ids(
        train,
        unit_ids,
        regime_key=regime_key,
        time_key=resolved_time_key,
        group_key=group_key,
    )
    history, history_reason = detect_history_executor(
        unit_ids,
        progression,
        regime_ids=regime_ids,
        min_history_points=min_history_points,
    )
    dual, score, dual_reason = detect_dual_scale_executor(
        unit_ids,
        support,
        min_units=min_units_for_support,
        min_heterogeneity=dual_scale_min_heterogeneity,
    )
    transport, transport_reason = detect_regime_transport_executor(
        unit_ids,
        regime_ids,
    )
    return OptionalExecutorDetection(
        history=history,
        dual_scale=dual,
        transport=transport,
        support_heterogeneity=score,
        reasons={
            "group": group_source,
            "time": time_source,
            "history": history_reason,
            "dual_scale": dual_reason,
            "transport": f"{transport_reason} ({regime_source})",
        },
    )


def _pick_first_key(data: Mapping[str, object], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in data:
            return key
    return None


def infer_ppx_contract(structure: DatasetStructure) -> ContractInference:
    """Infer a PPXContract from train-only dataset structure."""
    reasons: dict[str, str] = {}

    known_boundary = (
        structure.boundary_value is not None
        and np.isfinite(structure.boundary_value)
    )
    reasons["known_boundary"] = (
        f"고장 경계값 {structure.boundary_value} 선언"
        if known_boundary
        else "고장 경계값이 없거나 유한하지 않다"
    )

    unit_ids = _as_1d(structure.unit_ids, "unit_ids")
    coordinate = _as_1d(structure.progression_coordinate, "progression_coordinate")
    ordered_progression = False
    if unit_ids is None or coordinate is None:
        reasons["ordered_progression"] = "unit id 또는 진행 좌표가 없다"
    elif unit_ids.shape[0] != coordinate.shape[0]:
        reasons["ordered_progression"] = "unit id와 진행 좌표 길이가 다르다"
    else:
        ordered_progression, reason = _ordered_progression_audit(unit_ids, coordinate)
        reasons["ordered_progression"] = reason

    causal_history, history_reason = detect_history_executor(
        unit_ids,
        coordinate,
        regime_ids=(
            None if structure.regime_ids is None
            else np.asarray(structure.regime_ids)
        ),
        min_history_points=structure.min_history_points,
    )
    reasons["causal_history"] = history_reason

    regime_ids = None
    if structure.regime_ids is not None:
        regime_ids = _as_1d(structure.regime_ids, "regime_ids")
    observed_regime, regime_reason = detect_regime_transport_executor(
        unit_ids, regime_ids
    )
    reasons["observed_regime"] = regime_reason

    support_coord = structure.support_coordinate
    if support_coord is None:
        support_coord = coordinate
    support_heterogeneity_available, support_heterogeneity, dual_reason = (
        detect_dual_scale_executor(
            unit_ids,
            support_coord,
            min_units=structure.min_units_for_support,
            min_heterogeneity=structure.dual_scale_min_heterogeneity,
        )
    )
    reasons["support_heterogeneity_available"] = dual_reason

    contract = PPXContract(
        known_boundary=known_boundary,
        ordered_progression=ordered_progression,
        causal_history=causal_history,
        observed_regime=observed_regime,
        support_heterogeneity_available=support_heterogeneity_available,
        fallback=structure.fallback,
    )
    return ContractInference(
        contract=contract,
        support_heterogeneity=support_heterogeneity,
        reasons=reasons,
    )


def infer_ppx_contract_from_train_rows(
    train: Mapping[str, object],
    *,
    boundary_value: float | None = None,
    time_key: str | None = None,
    regime_key: str | None = None,
    group_key: str | None = None,
    min_history_points: int = 2,
    min_units_for_support: int = 2,
    dual_scale_min_heterogeneity: float = 0.5,
    fallback: Literal["direct_fallback", "persistence_fallback"] = "direct_fallback",
) -> ContractInference:
    """Infer a contract from a standard train row dictionary.

    User-supplied keys take precedence; blank keys request train-only
    detection. Unit IDs use standard names first, then repeated contiguous
    top-level ID columns. If unit inference is absent or ambiguous, the caller
    must provide ``group_key``. If time cannot be resolved uniquely, ordered
    progression and causal history are disabled; ``x[:, 0]`` is not a temporal
    fallback.
    """
    unit_ids, group_key, group_source = infer_group_ids(
        train, group_key=group_key
    )
    regime_key = _optional_requested_key(train, regime_key, kind="regime")
    regime_key = regime_key or _pick_first_key(train, _REGIME_KEYS)

    progression, resolved_time_key, time_source = infer_time_coordinate(
        train,
        unit_ids,
        time_key=time_key,
        group_key=group_key,
        regime_key=regime_key,
    )
    features = None
    if "x" in train:
        features = np.asarray(train["x"], dtype=np.float64)
        if features.ndim != 2 or features.shape[0] != unit_ids.shape[0]:
            features = None
    support_coordinate = None if features is None else features[:, 0]
    regime_ids, regime_source = infer_regime_ids(
        train,
        unit_ids,
        regime_key=regime_key,
        time_key=resolved_time_key,
        group_key=group_key,
    )

    structure = DatasetStructure(
        boundary_value=boundary_value,
        unit_ids=unit_ids,
        progression_coordinate=progression,
        regime_ids=regime_ids,
        support_coordinate=support_coordinate,
        min_history_points=min_history_points,
        min_units_for_support=min_units_for_support,
        dual_scale_min_heterogeneity=dual_scale_min_heterogeneity,
        fallback=fallback,
    )
    inferred = infer_ppx_contract(structure)
    reasons = dict(inferred.reasons)
    reasons["group_source"] = group_source
    reasons["time_source"] = time_source
    reasons["regime_source"] = regime_source
    return ContractInference(
        contract=inferred.contract,
        support_heterogeneity=inferred.support_heterogeneity,
        reasons=reasons,
    )
