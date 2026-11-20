# coding: utf-8
"""Tests of what this fork adds to LightGBM: the lambdarank targets, the precision
metric, the precision_nms metric with its dataset fields, and lambdagap_weight."""

import numpy as np
import pytest

import lightgbm as lgb

TARGETS = [
    "ndcg",
    "lambdaloss-ndcg",
    "lambdaloss-ndcg-plus-plus",
    "bndcg",
    "lambdaloss-bndcg",
    "lambdaloss-bndcg-plus-plus",
    "precision",
    "arpk",
    "lambdaloss-arp1",
    "lambdaloss-arp2",
    "ranknet",
    "bin-ranknet",
    "lambdagap-s",
    "lambdagap-x",
    "lambdagap-s-plus",
    "lambdagap-x-plus",
    "lambdagap-s-plus-plus",
    "lambdagap-x-plus-plus",
]
EVAL_AT = [1, 3, 5]


def ranking_data(seed=0, discrete=False):
    """Queries of 1 to 40 rows: labels 0 to 3, and features whose few distinct values
    with ``discrete`` make many tied scores."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 41, 60)
    n = int(sizes.sum())
    X = rng.integers(0, 3, (n, 4)).astype(float) if discrete else rng.random((n, 4))
    y = rng.integers(0, 4, n).astype(float)
    return X, y, sizes


def nms_fields(n, seed=0):
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, 8, n).astype(np.int32)  # Repeated within a query
    coordinates = np.round(rng.uniform(100, 110, n), 1)  # Often within a tolerance
    tiebreaks = rng.integers(0, 3, n).astype(np.float64)
    return groups, coordinates, tiebreaks


def train(params, X, y, sizes, rounds=5, valid=None, ds=None):
    if ds is None:
        ds = lgb.Dataset(X, y, group=sizes, params=params)
    record = {}
    booster = lgb.train(
        params,
        ds,
        rounds,
        valid_sets=[valid] if valid is not None else None,
        valid_names=["valid"] if valid is not None else None,
        callbacks=[lgb.record_evaluation(record)] if valid is not None else None,
    )
    return booster, record


def base_params(**extra):
    params = {
        "objective": "lambdarank",
        "verbose": -1,
        "num_leaves": 7,
        "min_data_in_leaf": 5,
        "deterministic": True,
        "force_row_wise": True,
        "eval_at": EVAL_AT,
    }
    params.update(extra)
    return params


def train_and_valid(params, X, y, sizes):
    """The training dataset, and a validation one on the same rows that refers to it,
    both built, so fields can be set on the validation one."""
    ds = lgb.Dataset(X, y, group=sizes, params=params).construct()
    return ds, lgb.Dataset(X, y, group=sizes, reference=ds, params=params).construct()


def query_starts(sizes):
    return np.r_[0, np.cumsum(sizes)]


def reference_precision(score, y, sizes, k):
    """Share of relevant rows among the k best scored, equal scores by row order."""
    starts = query_starts(sizes)
    values = []
    for start, end in zip(starts[:-1], starts[1:]):
        order = np.argsort(-score[start:end], kind="stable")
        values.append(np.sum(y[start:end][order][:k] > 0.5) / k)
    return float(np.mean(values))


def reference_precision_nms(score, y, sizes, k, groups, coordinates, tiebreaks, tolerance, relative):
    """Precision at k of the rows kept: by decreasing score, then tie-break, then row
    order; the first row of each group only, and only if its coordinate is farther than
    the tolerance from that of every row kept before."""
    starts = query_starts(sizes)
    values = []
    for start, end in zip(starts[:-1], starts[1:]):
        rows = np.arange(start, end)
        keys = (rows, -tiebreaks[rows]) if tiebreaks is not None else (rows,)
        order = rows[np.lexsort((*keys, -score[rows]))]
        seen, kept, hits = set(), [], 0
        for row in order:
            if groups[row] in seen:
                continue
            seen.add(groups[row])
            x = coordinates[row]
            if kept:
                distance = min(abs(x - c) / max(x, c) if relative else abs(x - c) for c in kept)
                if not distance > tolerance:
                    continue
            kept.append(x)
            hits += y[row] > 0.5
            if len(kept) == k:
                break
        values.append(hits / k)
    return float(np.mean(values))


@pytest.mark.parametrize("target", TARGETS)
def test_target_trains_the_same_model_twice(target):
    X, y, sizes = ranking_data()
    params = base_params(lambdarank_target=target, lambdarank_truncation_level=3)
    first, _ = train(params, X, y, sizes)
    second, _ = train(params, X, y, sizes)
    assert first.num_trees() == 5
    assert first.model_to_string() == second.model_to_string()


@pytest.mark.parametrize("target", ["precision", "lambdagap-s-plus", "lambdagap-x-plus"])
def test_target_with_truncation_beyond_every_query(target):
    # Every query is shorter than the truncation level
    X, y, sizes = ranking_data()
    sizes = np.full(len(y) // 4, 4)
    X, y = X[: sizes.sum()], y[: sizes.sum()]
    booster, _ = train(base_params(lambdarank_target=target, lambdarank_truncation_level=50), X, y, sizes)
    assert np.all(np.isfinite(booster.predict(X)))


@pytest.mark.parametrize("discrete", [False, True])
def test_precision_metric(discrete):
    X, y, sizes = ranking_data(discrete=discrete)
    params = base_params(metric="precision")
    valid = lgb.Dataset(X, y, group=sizes, params=params)
    booster, record = train(params, X, y, sizes, valid=valid)
    score = booster.predict(X)
    for k in EVAL_AT:
        expected = reference_precision(score, y, sizes, k)
        assert record["valid"][f"precision@{k}"][-1] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("with_tiebreak", [False, True])
@pytest.mark.parametrize("discrete", [False, True])
def test_precision_nms_metric(relative, with_tiebreak, discrete):
    X, y, sizes = ranking_data(discrete=discrete)
    groups, coordinates, tiebreaks = nms_fields(len(y))
    tolerance = 0.002 if relative else 0.25
    params = base_params(
        metric="precision_nms",
        nms_tolerance=tolerance,
        nms_tolerance_mode="relative" if relative else "absolute",
    )
    ds, valid = train_and_valid(params, X, y, sizes)
    valid.set_field("nms_group", groups)
    valid.set_field("nms_coordinate", coordinates)
    if with_tiebreak:
        valid.set_field("nms_tiebreak", tiebreaks)
    booster, record = train(params, X, y, sizes, valid=valid, ds=ds)
    score = booster.predict(X)
    for k in EVAL_AT:
        expected = reference_precision_nms(
            score, y, sizes, k, groups, coordinates, tiebreaks if with_tiebreak else None, tolerance, relative
        )
        assert record["valid"][f"precision_nms@{k}"][-1] == pytest.approx(expected, abs=1e-12)


def test_precision_nms_without_its_fields_is_nan():
    X, y, sizes = ranking_data()
    params = base_params(metric="precision_nms")
    valid = lgb.Dataset(X, y, group=sizes, params=params)
    _, record = train(params, X, y, sizes, valid=valid)
    assert all(np.isnan(record["valid"][f"precision_nms@{k}"][-1]) for k in EVAL_AT)


def test_nms_fields_round_trip_and_subset():
    X, y, sizes = ranking_data()
    groups, coordinates, tiebreaks = nms_fields(len(y))
    ds = lgb.Dataset(X, y, group=sizes, params=base_params(), free_raw_data=False).construct()
    ds.set_field("nms_group", groups)
    ds.set_field("nms_coordinate", coordinates)
    ds.set_field("nms_tiebreak", tiebreaks)
    np.testing.assert_array_equal(ds.get_field("nms_group"), groups)
    np.testing.assert_array_equal(ds.get_field("nms_coordinate"), coordinates)
    np.testing.assert_array_equal(ds.get_field("nms_tiebreak"), tiebreaks)
    # The rows of the first three queries
    rows = np.arange(int(np.sum(sizes[:3])))
    subset = ds.subset(rows).construct()
    np.testing.assert_array_equal(subset.get_field("nms_group"), groups[rows])
    np.testing.assert_array_equal(subset.get_field("nms_coordinate"), coordinates[rows])
    np.testing.assert_array_equal(subset.get_field("nms_tiebreak"), tiebreaks[rows])


def test_precision_metrics_are_higher_is_better():
    X, y, sizes = ranking_data()
    groups, coordinates, _ = nms_fields(len(y))
    params = base_params(metric=["precision", "precision_nms"])
    ds, valid = train_and_valid(params, X, y, sizes)
    valid.set_field("nms_group", groups)
    valid.set_field("nms_coordinate", coordinates)
    booster = lgb.train(params, ds, 2, valid_sets=[valid], keep_training_booster=True)
    results = booster.eval_valid()
    assert {name for _, name, _, _ in results} == {
        f"{metric}@{k}" for metric in ("precision", "precision_nms") for k in EVAL_AT
    }
    assert all(higher_better for *_, higher_better in results)


def test_early_stopping_maximizes_precision():
    X, y, sizes = ranking_data()
    params = base_params(metric="precision", eval_at=[3], learning_rate=0.5)
    valid = lgb.Dataset(X, y, group=sizes, params=params)
    record = {}
    booster = lgb.train(
        params,
        lgb.Dataset(X, y, group=sizes, params=params),
        30,
        valid_sets=[valid],
        callbacks=[lgb.early_stopping(5, verbose=False), lgb.record_evaluation(record)],
    )
    values = record["valid_0"]["precision@3"]
    assert values[booster.best_iteration - 1] == max(values)


def test_lambdagap_weight_is_read_back_from_a_saved_model():
    X, y, sizes = ranking_data()
    params = base_params(lambdarank_target="lambdagap-s-plus", lambdagap_weight=0.5)
    booster, _ = train(params, X, y, sizes)
    loaded = lgb.Booster(model_str=booster.model_to_string())
    assert loaded.params["lambdagap_weight"] == 0.5
