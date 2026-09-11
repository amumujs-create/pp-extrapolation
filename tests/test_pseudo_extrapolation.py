import numpy as np

from pp_extrapolation.pseudo_extrapolation import (
    build_nested_pseudo_extrapolation_episodes,
    fit_support_grade,
    transform_support_grade,
    transform_training_support_grades,
)


def _source():
    groups = np.repeat(np.array(["a", "b", "c", "d"], dtype=object), 10)
    coordinate = np.tile(np.arange(10, dtype=float), 4)
    offsets = np.repeat(np.array([0.0, 0.2, -0.1, 0.1]), 10)
    x = np.column_stack((coordinate + offsets, np.sin(coordinate) + offsets))
    return x, groups, coordinate


def test_training_grades_exclude_the_rows_own_unit_centroid():
    x, groups, coordinate = _source()
    fit = fit_support_grade(x, groups, coordinate)
    grades = transform_training_support_grades(fit, x, groups, coordinate)

    assert grades.shape == (len(x),)
    assert np.isfinite(grades).all()
    assert fit.unit_centroids.shape[0] == len(set(groups))


def test_application_grade_is_pointwise_and_test_batch_invariant():
    x, groups, coordinate = _source()
    fit = fit_support_grade(x, groups, coordinate)
    row_x = np.array([[11.5, -0.25]])
    row_coordinate = np.array([11.5])

    alone = transform_support_grade(fit, row_x, row_coordinate)[0]
    batch = transform_support_grade(
        fit,
        np.vstack((row_x, [[-1_000.0, 1_000.0]])),
        np.array([11.5, -1_000.0]),
    )[0]

    assert alone == batch


def test_nested_plan_excludes_outer_unit_and_never_mixes_units():
    _, groups, coordinate = _source()
    plan = build_nested_pseudo_extrapolation_episodes(groups, coordinate)

    assert plan.audit_passed
    for episode in plan.episodes:
        fit_units = set(groups[episode.fit_indices])
        tune_units = set(groups[episode.tune_indices])
        query_units = set(groups[episode.query_indices])
        assert episode.outer_unit not in set(
            groups[np.concatenate((episode.fit_indices, episode.tune_indices))]
        )
        assert fit_units.isdisjoint(tune_units)
        assert fit_units.isdisjoint(query_units)
        assert tune_units.isdisjoint(query_units)


def test_nested_plan_is_deterministic():
    _, groups, coordinate = _source()
    first = build_nested_pseudo_extrapolation_episodes(groups, coordinate)
    second = build_nested_pseudo_extrapolation_episodes(groups, coordinate)

    assert len(first.episodes) == len(second.episodes)
    for left, right in zip(first.episodes, second.episodes):
        assert left.outer_unit == right.outer_unit
        assert left.inner_unit == right.inner_unit
        assert left.cutoff == right.cutoff
        assert left.threshold == right.threshold
        assert np.array_equal(left.fit_indices, right.fit_indices)
        assert np.array_equal(left.tune_indices, right.tune_indices)
        assert np.array_equal(left.query_indices, right.query_indices)
