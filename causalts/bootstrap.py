# Copyright 2025 Bloomberg Finance L.P.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Temporal bootstrap for causal discovery on time series.

The key function is :func:`temporal_bootstrap`, which runs any discovery
callable repeatedly on random contiguous windows of a time series and
aggregates the results into a link-persistence matrix.

Unlike random-index subsampling (as in ``run_stability_selection``), contiguous
windows preserve the temporal autocorrelation structure of the process —
essential for time-series causal discovery.  The pattern mirrors ``boot.strength``
in the bnlearn R package and the bootstrap used in the elbe_river example.
"""

from __future__ import annotations

import traceback
from typing import Callable

import numpy as np
import pandas as pd


def temporal_bootstrap(
    df: pd.DataFrame,
    discovery_fn: Callable[[pd.DataFrame], np.ndarray],
    n_bootstrap: int = 20,
    window_frac: float = 0.6,
    seed: int = 42,
    show_progress: bool = True,
    n_jobs: int = 1,
    backend: str = "loky",
) -> dict:
    """Contiguous-window temporal bootstrap for causal discovery.

    Draws ``n_bootstrap`` random contiguous windows from ``df``, runs
    ``discovery_fn`` on each window, and returns the per-edge mean
    (persistence / frequency) alongside the raw per-run outputs.

    The callable ``discovery_fn`` receives a *reset-index* DataFrame of
    length ``window_size`` and must return a numpy array of shape
    ``(d, d, max_lag+1)`` — binary (e.g. from CDNOTS) or continuous gate
    values (e.g. from GRACE).  The mean is meaningful in both cases.

    Parameters
    ----------
    df : DataFrame of shape (T, d)
        Full time series. Any index is acceptable; windows are extracted
        by integer position.
    discovery_fn : callable (DataFrame) -> ndarray (d, d, L)
        Discovery algorithm to run on each window.  Must return an array
        whose mean across runs gives the desired persistence measure.
        Example wrappers::

            # CDNOTS
            from causalts import run_cdnots
            ci = PartialCorr(data=np.zeros((2, 2)))
            def my_discovery(sub):
                res = run_cdnots(sub, ci, num_lags=2, include_C=False, alpha=0.05)
                return res.cg_tig

            # GRACE (continuous gates)
            def run_grace_gates(sub):
                res = run_cdnots_gated(sub, max_lag=2,
                                      model_seed=0, verbose=False)
                return res.gate_values

    n_bootstrap : int, default 20
        Number of bootstrap windows to draw.
    window_frac : float, default 0.6
        Fraction of T to use as window size.  Must satisfy
        ``window_frac * T >= 10`` (a practical minimum).
    seed : int, default 42
        Seed for the window-start RNG.  Each run's ``discovery_fn`` is
        called with its own internal state; set ``model_seed`` inside the
        closure for full reproducibility of GRACE runs.
    show_progress : bool, default True
        Show a tqdm progress bar.
    n_jobs : int, default 1
        Number of windows to run concurrently (joblib semantics: ``-1``
        uses all cores).  All window starts are drawn from the ``seed`` RNG
        up front, and outputs are collected in window order, so the result
        is identical to the serial run (given a deterministic
        ``discovery_fn``).  With the default ``"loky"`` backend
        ``discovery_fn`` runs in worker processes and must be picklable
        (closures are fine; loky uses cloudpickle).
    backend : str, default "loky"
        joblib backend used when ``n_jobs != 1``.  **GPU CI tests**
        (``ParCorrGPU``, ``KCIGPU``, ...) should use ``n_jobs=1`` or
        ``backend="threading"``: separate loky processes each initialise
        their own CUDA context and compete for device memory.

    Returns
    -------
    dict with keys:

    ``persistence`` : ndarray (d, d, L)
        Element-wise mean of all successful run outputs.  For binary
        ``discovery_fn`` outputs, values ∈ [0, 1] represent the fraction
        of windows in which each edge was discovered.  Apply any threshold
        to obtain a final graph::

            G = (result["persistence"] >= 0.6).astype(int)

    ``graphs`` : list of ndarray
        Raw output of ``discovery_fn`` for each successful run, in order.

    ``n_success`` : int
        Number of runs that completed without error.

    ``n_errors`` : int
        Number of runs that raised an exception (see ``errors``).

    ``window_size`` : int
        Number of rows per window (``floor(T * window_frac)``).

    ``errors`` : list of (run_index, error_message)
        Diagnostic information for failed runs.

    Raises
    ------
    ValueError
        If the window size is too small (< 10 rows) or the dataset is too
        short to draw even one window.

    Examples
    --------
    >>> from causalts import run_cdnots, temporal_bootstrap
    >>> from causalts.ci_tests import PartialCorr
    >>> import numpy as np
    >>>
    >>> ci = PartialCorr(data=np.zeros((2, 2)))
    >>> def my_discovery(sub):
    ...     res = run_cdnots(sub, ci, num_lags=2, include_C=False, alpha=0.05)
    ...     return res.cg_tig
    >>>
    >>> result = temporal_bootstrap(df, my_discovery, n_bootstrap=20,
    ...                             window_frac=0.6, seed=42)
    >>> G_stable = (result["persistence"] >= 0.6).astype(int)
    """
    T = len(df)
    W = int(T * window_frac)

    if W < 10:
        raise ValueError(
            f"window_size = int({T} * {window_frac}) = {W} is too small "
            f"(minimum 10 rows). Reduce window_frac or use a larger dataset."
        )
    if W >= T:
        raise ValueError(f"window_size {W} >= T {T}. Use window_frac < 1.0.")

    rng = np.random.default_rng(seed)
    starts = [int(rng.integers(0, T - W + 1)) for _ in range(n_bootstrap)]
    windows = (
        (i, df.iloc[start : start + W].reset_index(drop=True))
        for i, start in enumerate(starts)
    )

    try:
        from tqdm.auto import tqdm as _tqdm

        _has_tqdm = True
    except ImportError:
        _has_tqdm = False

    if n_jobs == 1:
        outcomes = (_run_window(discovery_fn, i, window) for i, window in windows)
    else:
        from joblib import Parallel, delayed

        outcomes = Parallel(n_jobs=n_jobs, backend=backend, return_as="generator")(
            delayed(_run_window)(discovery_fn, i, window) for i, window in windows
        )
    if show_progress and _has_tqdm:
        outcomes = _tqdm(
            outcomes, total=n_bootstrap, desc="temporal_bootstrap", unit="run"
        )

    graphs: list[np.ndarray] = []
    errors: list[tuple[int, str]] = []
    for i, ok, payload in outcomes:
        if ok:
            graphs.append(payload)
        else:
            errors.append((i, payload))

    if not graphs:
        raise RuntimeError(
            f"All {n_bootstrap} bootstrap runs failed. "
            f"First error:\n{errors[0][1] if errors else 'unknown'}"
        )

    persistence = np.mean(graphs, axis=0)

    return {
        "persistence": persistence,
        "graphs": graphs,
        "n_success": len(graphs),
        "n_errors": len(errors),
        "window_size": W,
        "errors": errors,
    }


def _run_window(discovery_fn, i, window):
    """Run one bootstrap window; returns ``(i, ok, graph_or_traceback)``."""
    try:
        return i, True, np.asarray(discovery_fn(window))
    except Exception:
        return i, False, traceback.format_exc()
