# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the vectorized tigramite graph converter, the CITestAdapter
result cache, and parallel temporal_bootstrap."""

import itertools

import numpy as np
import pandas as pd
import pytest

from causalts.bootstrap import temporal_bootstrap
from causalts.tigramite_discovery import CITestAdapter, tigramite_graph_to_binary

MARKS = ["", "-->", "<--", "o->", "<-o", "o-o", "x-x", "<->", " --> ", "o-o "]


def _loop_graph_to_binary(
    tig_graph, p_matrix=None, alpha_level=None, undirected="drop", conflict="drop"
):
    """Reference: the original per-element loop implementation."""
    symmetric = set()
    if undirected == "bidirected":
        symmetric.add("o-o")
    if conflict == "bidirected":
        symmetric.add("x-x")
    d = tig_graph.shape[0]
    tau_max_plus_1 = tig_graph.shape[2]
    G = np.zeros((d, d, tau_max_plus_1), dtype=np.int8)
    for i in range(d):
        for j in range(d):
            for tau in range(tau_max_plus_1):
                s = str(tig_graph[i, j, tau]).strip()
                if s in ("-->", "o->"):
                    if p_matrix is not None and alpha_level is not None:
                        if p_matrix[i, j, tau] >= alpha_level:
                            continue
                    G[i, j, tau] = 1
                elif s in ("<--", "<-o"):
                    if p_matrix is not None and alpha_level is not None:
                        if p_matrix[i, j, tau] >= alpha_level:
                            continue
                    G[j, i, tau] = 1
                elif s in symmetric:
                    if p_matrix is not None and alpha_level is not None:
                        if p_matrix[i, j, tau] >= alpha_level:
                            continue
                    G[i, j, tau] = 1
                    G[j, i, tau] = 1
    return G


@pytest.mark.parametrize(
    "undirected,conflict",
    list(itertools.product(["drop", "bidirected"], ["drop", "bidirected"])),
)
@pytest.mark.parametrize("use_p", [False, True])
def test_vectorized_converter_matches_loop(undirected, conflict, use_p):
    rng = np.random.default_rng(0)
    for trial in range(25):
        d = int(rng.integers(1, 9))
        L = int(rng.integers(1, 5))
        graph = np.array(MARKS)[rng.integers(0, len(MARKS), size=(d, d, L))]
        if trial % 2:
            graph = graph.astype(object)
        p_matrix = alpha = None
        if use_p:
            p_matrix = rng.random((d, d, L))
            p_matrix[rng.random((d, d, L)) < 0.1] = np.nan
            alpha = float(rng.choice([0.05, 0.3, 0.5]))
        expected = _loop_graph_to_binary(graph, p_matrix, alpha, undirected, conflict)
        got = tigramite_graph_to_binary(
            graph, p_matrix, alpha, undirected=undirected, conflict=conflict
        )
        assert got.dtype == np.int8
        np.testing.assert_array_equal(got, expected)


def test_vectorized_converter_p_matrix_without_alpha_is_ignored():
    graph = np.array([["", "-->"], ["<--", ""]]).reshape(2, 2, 1)
    p = np.ones((2, 2, 1))
    np.testing.assert_array_equal(
        tigramite_graph_to_binary(graph, p_matrix=p),
        _loop_graph_to_binary(graph, p_matrix=p),
    )


# --------------------------------------------------------------------------
# CITestAdapter result cache
# --------------------------------------------------------------------------


def _var_df(T=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((T, 4))
    for t in range(1, T):
        x[t, 1] += 0.6 * x[t - 1, 0]
        x[t, 2] += 0.5 * x[t - 1, 1] + 0.4 * x[t, 0]
        x[t, 3] += 0.5 * x[t - 1, 3]
    return pd.DataFrame(x, columns=[f"X{i}" for i in range(4)])


class _CountingTest:
    """Wraps a CIT_Base test and counts __call__ invocations."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    @property
    def data(self):
        return self.inner.data

    @data.setter
    def data(self, d):
        self.inner.data = d

    def __call__(self, X, Y, Z):
        self.calls += 1
        return self.inner(X, Y, Z)


def _parcorr_gpu():
    from causalts.ci_tests.parcorr_gpu import ParCorrGPU

    return ParCorrGPU(data=np.zeros((10, 2)), device="cpu")


def _attached_adapter(df):
    pytest.importorskip("tigramite")
    from causalts.tigramite_discovery import _to_tigramite_df

    counter = _CountingTest(_parcorr_gpu())
    adapter = CITestAdapter(counter)
    adapter.set_dataframe(_to_tigramite_df(df))
    return adapter, counter


def test_adapter_cache_hit_returns_same_values():
    adapter, counter = _attached_adapter(_var_df())
    args = dict(X=[(0, -1)], Y=[(1, 0)], Z=[(1, -1), (2, -2)], tau_max=2)
    first = adapter.run_test(**args)
    second = adapter.run_test(**args)
    assert second == first
    assert adapter._cached_pval == first[1]
    assert (adapter.n_tests, adapter.n_actual_tests, counter.calls) == (2, 1, 1)

    val, pval, dependent = adapter.run_test(**args, alpha_or_thres=0.05)
    assert (val, pval) == first
    assert dependent == (first[1] <= 0.05)
    assert (adapter.n_tests, adapter.n_actual_tests, counter.calls) == (3, 1, 1)


def test_adapter_cache_cleared_on_new_data():
    pytest.importorskip("tigramite")
    from causalts.tigramite_discovery import _to_tigramite_df

    adapter, counter = _attached_adapter(_var_df(seed=0))
    args = dict(X=[(0, -1)], Y=[(1, 0)], Z=[], tau_max=1)
    adapter.run_test(**args)
    assert adapter._result_cache

    adapter.clear_cache()
    assert not adapter._result_cache
    adapter.run_test(**args)
    assert adapter.n_actual_tests == 2

    adapter.set_dataframe(_to_tigramite_df(_var_df(seed=1)))
    assert not adapter._result_cache
    adapter.run_test(**args)
    adapter.dataframe = _to_tigramite_df(_var_df(seed=2))
    assert not adapter._result_cache
    assert counter.calls == 3


class _UncachedAdapter(CITestAdapter):
    def run_test(self, *args, **kwargs):
        self.clear_cache()
        return super().run_test(*args, **kwargs)


def test_adapter_cache_pcmciplus_same_pvalues_fewer_gpu_calls():
    pytest.importorskip("tigramite")
    from tigramite.pcmci import PCMCI

    from causalts.tigramite_discovery import _to_tigramite_df

    tig_df = _to_tigramite_df(_var_df())
    runs = {}
    for name, cls in [("cached", CITestAdapter), ("uncached", _UncachedAdapter)]:
        pcmci = PCMCI(
            dataframe=tig_df, cond_ind_test=cls(_CountingTest(_parcorr_gpu()))
        )
        res = pcmci.run_pcmciplus(tau_max=2, pc_alpha=0.05)
        runs[name] = (res, pcmci.cond_ind_test)

    (res_c, cit_c), (res_u, cit_u) = runs["cached"], runs["uncached"]
    np.testing.assert_array_equal(res_c["p_matrix"], res_u["p_matrix"])
    np.testing.assert_array_equal(res_c["val_matrix"], res_u["val_matrix"])
    np.testing.assert_array_equal(res_c["graph"], res_u["graph"])
    assert cit_c.n_tests == cit_u.n_tests
    assert cit_u.n_actual_tests == cit_u.n_tests
    assert cit_c.n_actual_tests < cit_u.n_actual_tests
    assert cit_c._gpu_test.calls == cit_c.n_actual_tests
    assert cit_u._gpu_test.calls == cit_u.n_actual_tests


# --------------------------------------------------------------------------
# temporal_bootstrap parallelism
# --------------------------------------------------------------------------


def _lagged_corr_graph(window):
    x = window.to_numpy()
    d = x.shape[1]
    G = np.zeros((d, d, 2))
    G[:, :, 0] = np.abs(np.corrcoef(x, rowvar=False)) > 0.3
    G[:, :, 1] = np.abs(np.corrcoef(x[:-1], x[1:], rowvar=False)[:d, d:]) > 0.3
    return G


def _sometimes_fails(window):
    if window.iloc[0, 0] > 0.5:
        raise ValueError(f"bad window starting at {window.iloc[0, 0]:.6f}")
    return _lagged_corr_graph(window)


@pytest.mark.parametrize("fn", [_lagged_corr_graph, _sometimes_fails])
def test_temporal_bootstrap_parallel_matches_serial(fn):
    df = _var_df(T=300, seed=3)
    kw = dict(n_bootstrap=12, window_frac=0.5, seed=7, show_progress=False)
    serial = temporal_bootstrap(df, fn, n_jobs=1, **kw)
    parallel = temporal_bootstrap(df, fn, n_jobs=2, **kw)

    assert list(parallel) == list(serial)
    np.testing.assert_array_equal(parallel["persistence"], serial["persistence"])
    assert len(parallel["graphs"]) == len(serial["graphs"])
    for gp, gs in zip(parallel["graphs"], serial["graphs"]):
        np.testing.assert_array_equal(gp, gs)
    for key in ("n_success", "n_errors", "window_size"):
        assert parallel[key] == serial[key]
    assert [i for i, _ in parallel["errors"]] == [i for i, _ in serial["errors"]]
    assert [e.strip().splitlines()[-1] for _, e in parallel["errors"]] == [
        e.strip().splitlines()[-1] for _, e in serial["errors"]
    ]
    if fn is _sometimes_fails:
        assert 0 < serial["n_errors"] < kw["n_bootstrap"]
