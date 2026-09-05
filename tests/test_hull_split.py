import numpy as np
import pandas as pd

from pp_extrapolation.hull import audit_convex_hull_support
from pp_extrapolation.split import strict_hull_split_1d


def test_hull_detects_inside_and_outside_points():
    audit = audit_convex_hull_support(
        np.asarray([[0.0], [1.0]]), np.asarray([[0.5], [-0.1], [1.1]])
    )
    assert audit.outside_mask.tolist() == [False, True, True]


def test_split_is_unit_disjoint_and_strict_low_extrapolation():
    frame = pd.DataFrame(
        [(unit, q) for unit in range(20) for q in np.linspace(1.0, 0.0, 21)],
        columns=["unit", "health"],
    )
    train, validation, test, _ = strict_hull_split_1d(
        frame,
        unit_column="unit",
        coordinate="health",
        train_cutoff=0.6,
        direction="low",
    )
    assert set(train.unit).isdisjoint(validation.unit)
    assert set(train.unit).isdisjoint(test.unit)
    assert set(validation.unit).isdisjoint(test.unit)
    assert validation.health.max() < train.health.min()
    assert test.health.max() < train.health.min()

