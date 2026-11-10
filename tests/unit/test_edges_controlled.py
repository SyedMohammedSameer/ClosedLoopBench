"""
Unit tests for clbench.edges.controlled (surrogate p-values + BH / BY edge selection).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from clbench.edges.controlled import (  # noqa: E402
    STATISTICS,
    auto_spacing,
    benjamini_hochberg,
    benjamini_yekutieli,
    block_bootstrap_index,
    candidate_mask,
    controlled_edges,
    edge_pvalues,
    lattice_offsets,
    shift_index,
)


def _ar1(T: int, p: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    X = np.zeros((T + 300, p))
    E = rng.normal(size=(T + 300, p))
    for t in range(1, T + 300):
        X[t] = phi * X[t - 1] + E[t]
    return X[300:]


# ---------------------------------------------------------------------------
# BH / BY
# ---------------------------------------------------------------------------


def test_bh_matches_statsmodels() -> None:
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(0)
    for _ in range(200):
        m = int(rng.integers(1, 60))
        p = np.concatenate([rng.uniform(size=m // 2), rng.uniform(0, 0.01, size=m - m // 2)])
        rng.shuffle(p)
        for alpha in (0.05, 0.1, 0.2):
            ref = multipletests(p, alpha=alpha, method="fdr_bh")[0]
            assert np.array_equal(benjamini_hochberg(p, alpha), ref)
            ref_by = multipletests(p, alpha=alpha, method="fdr_by")[0]
            assert np.array_equal(benjamini_yekutieli(p, alpha), ref_by)


def test_bh_step_up_and_nan_handling() -> None:
    # Thresholds 0.025, 0.05, 0.075, 0.1: p(3) = 0.07 <= 0.075, so the three smallest are rejected.
    p = np.array([0.01, 0.04, 0.07, 0.9])
    assert benjamini_hochberg(p, 0.1).tolist() == [True, True, True, False]
    # Step-up: p(1) = 0.03 fails its own threshold 0.025 but is rejected because p(2) passes.
    assert benjamini_hochberg(np.array([0.03, 0.04, 0.5, 0.9]), 0.1).tolist() == [True, True, False, False]
    q = np.array([[np.nan, 0.01], [0.04, np.nan]])
    out = benjamini_hochberg(q, 0.05)  # m = 2: thresholds 0.025, 0.05
    assert out.tolist() == [[False, True], [True, False]]
    assert not benjamini_hochberg(np.full((3, 3), np.nan), 0.1).any()


def test_by_is_more_conservative_than_bh() -> None:
    rng = np.random.default_rng(1)
    for _ in range(100):
        p = rng.uniform(0, 0.05, size=20)
        assert np.all(benjamini_yekutieli(p, 0.1) <= benjamini_hochberg(p, 0.1))


# ---------------------------------------------------------------------------
# Surrogates
# ---------------------------------------------------------------------------


def test_lattice_is_a_subgroup() -> None:
    rng = np.random.default_rng(0)
    T, g = 240, 5
    offs = lattice_offsets(T, g, 10_000, rng)
    group = set(np.concatenate([[0], offs]).tolist())
    assert len(group) == T // g
    assert all(((a + b) % T) in group for a in group for b in group)
    with pytest.raises(ValueError):
        lattice_offsets(241, 5, 100, rng)
    sub = lattice_offsets(T, g, 10, rng)
    assert sub.size == 10 and set(sub.tolist()) <= group


def test_shift_index_is_roll() -> None:
    x = np.arange(12.0)
    idx = shift_index(12, np.array([3, 7]))
    assert np.array_equal(x[idx[0]], np.roll(x, 3))
    assert np.array_equal(x[idx[1]], np.roll(x, 7))


def test_block_bootstrap_shape_and_range() -> None:
    idx = block_bootstrap_index(100, 7, 12, np.random.default_rng(0))
    assert idx.shape == (7, 100)
    assert idx.min() >= 0 and idx.max() < 100
    assert np.all(np.diff(idx[:, :12], axis=1) % 100 == 1)  # first block is contiguous


def test_auto_spacing_grows_with_persistence() -> None:
    rng = np.random.default_rng(0)
    assert auto_spacing(rng.normal(size=2000), 4) == 5
    assert auto_spacing(_ar1(2000, 1, 0.95, rng)[:, 0], 4) > 20


# ---------------------------------------------------------------------------
# p-value validity under independence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("statistic", STATISTICS)
@pytest.mark.parametrize("phi", [0.0, 0.9])
def test_shift_pvalues_uniform_under_independence(statistic: str, phi: float) -> None:
    """Pooled p-values of independent series: P(p <= u) <= u + slack, mean ≈ 1/2."""
    rng = np.random.default_rng(123)
    ps = []
    for r in range(40):
        X = _ar1(480, 4, phi, rng)
        res = edge_pvalues(X, statistic=statistic, max_lag=3, seed=r)
        ps.append(res.pvalues_empirical[np.isfinite(res.pvalues_empirical)])
    p = np.concatenate(ps)  # 480 p-values
    assert 0.44 < p.mean() < 0.56
    for u in (0.05, 0.1, 0.2):
        # binomial sd at n=480, u=0.1 is 0.014; allow ~3 sd
        assert np.mean(p <= u) <= u + 3.0 * np.sqrt(u * (1 - u) / p.size)
    # The empirical p-value is never below its floor 1/(B+1).
    assert p.min() >= 1.0 / (res.n_surrogates.max() + 1) - 1e-12


def test_single_pair_exactness_in_the_tail() -> None:
    """The lattice (subgroup) test hits its minimum p-value at the nominal rate."""
    rng = np.random.default_rng(7)
    hits, n_rep = 0, 1500
    floor = None
    for r in range(n_rep):
        X = rng.normal(size=(120, 2))
        res = edge_pvalues(X, statistic="xcorr", max_lag=2, spacing=3, seed=r, candidates=np.array([[False, True], [False, False]]))
        floor = 1.0 / (res.n_surrogates[0] + 1)
        hits += res.pvalues[0, 1] <= floor + 1e-12
    rate = hits / n_rep  # nominal = floor = 1/40
    assert rate <= floor + 3.0 * np.sqrt(floor * (1 - floor) / n_rep)


# ---------------------------------------------------------------------------
# Power, direction, candidates
# ---------------------------------------------------------------------------


def test_planted_edge_is_detected() -> None:
    rng = np.random.default_rng(3)
    T = 2000
    X = _ar1(T, 5, 0.5, rng)
    X[2:, 3] += 0.8 * X[:-2, 1]  # 1 → 3 at lag 2
    ce = controlled_edges(X, alpha=0.1, statistic="pwcgranger", max_lag=3)
    assert ce.adjacency[1, 3]
    assert ce.tests.best_lag[1, 3] == 2
    assert ce.tests.pvalues[1, 3] == pytest.approx(1.0 / (ce.tests.n_surrogates[1] + 1))
    assert (1, 3) in [(a, b) for a, b, _, _ in ce.edges()]


def test_admitted_restricts_candidates() -> None:
    rng = np.random.default_rng(4)
    X = _ar1(600, 5, 0.3, rng)
    adm = [True, True, False, True, False]
    ce = controlled_edges(X, alpha=0.2, admitted=adm, statistic="pwxcorr", max_lag=2)
    mask = candidate_mask(5, adm)
    assert ce.n_candidates == int(mask.sum()) == 6
    assert np.all(np.isnan(ce.tests.pvalues[~mask]))
    assert not ce.adjacency[~mask].any()


def test_constant_series_gets_pvalue_one() -> None:
    rng = np.random.default_rng(5)
    X = _ar1(400, 3, 0.5, rng)
    X[:, 1] = 0.0
    res = edge_pvalues(X, statistic="cgranger", max_lag=2)
    assert res.pvalues[1, 0] == 1.0 and res.pvalues[0, 1] == 1.0
    assert np.isfinite(res.pvalues[0, 2])


def test_gamma_pvalues_are_continuous_and_close_under_null() -> None:
    rng = np.random.default_rng(6)
    X = _ar1(960, 4, 0.5, rng)
    res = edge_pvalues(X, statistic="pwcgranger", pvalue="gamma", max_lag=3)
    assert np.array_equal(res.pvalues, res.pvalues_gamma, equal_nan=True)
    fin = np.isfinite(res.pvalues)
    assert np.all((res.pvalues[fin] > 0) & (res.pvalues[fin] <= 1))
    assert np.corrcoef(res.pvalues_gamma[fin], res.pvalues_empirical[fin])[0, 1] > 0.9


# ---------------------------------------------------------------------------
# Working scale and covariates
# ---------------------------------------------------------------------------


def test_apply_transform_auto() -> None:
    from clbench.edges.controlled import apply_transform

    rng = np.random.default_rng(8)
    probs = rng.uniform(0.01, 0.99, size=50)
    binary = (rng.uniform(size=50) > 0.5).astype(float)
    raw = rng.normal(size=50)
    X = np.column_stack([probs, binary, raw])
    out = apply_transform(X, "auto")
    assert np.allclose(out[:, 0], np.log(probs / (1 - probs)))
    assert np.array_equal(out[:, 1], binary)
    assert np.array_equal(out[:, 2], raw)
    assert np.array_equal(apply_transform(X, "none"), X)


def test_observed_driver_as_covariate_removes_confounded_edge() -> None:
    """A slow shared driver with different delays creates a spurious a→b; conditioning on it fixes it."""
    rng = np.random.default_rng(9)
    T = 2400
    D = _ar1(T + 4, 1, 0.99, rng)[:, 0]
    D /= D.std()
    a = rng.normal(size=T) + 1.5 * D[4:]  # driver at delay 0
    b = rng.normal(size=T) + 1.5 * D[1 : T + 1]  # driver at delay 3 → a "precedes" b
    c = rng.normal(size=T)
    X = np.column_stack([a, b, c])
    plain = edge_pvalues(X, statistic="pwcgranger", max_lag=4, seed=0)
    adj = edge_pvalues(X, statistic="pwcgranger", max_lag=4, seed=0, covariates=D[4:, None])
    assert plain.pvalues[0, 1] <= 0.01
    assert adj.pvalues[0, 1] > 0.05
    with pytest.raises(ValueError):
        edge_pvalues(X, statistic="pwxcorr", covariates=D[4:, None])
