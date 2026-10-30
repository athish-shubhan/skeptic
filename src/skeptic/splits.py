"""Time-series splitters that respect label overlap.

Every sample i has a decision time ``t0[i]`` and a label end time ``t1[i]``
(the last moment whose data the label uses). Two samples overlap if their
[t0, t1] intervals intersect. A training sample is *purged* if it overlaps any
test sample, and *embargoed* if it starts within ``embargo`` after the test
period ends (to absorb serial correlation that the label window does not cover).

Times can be integers or numpy datetime64; ``embargo`` must be the matching
unit (an int, or a numpy timedelta64).

Reference: Lopez de Prado, M. (2018). Advances in Financial Machine Learning, ch. 7 and 12.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    test: np.ndarray
    group: tuple[int, ...] = ()


def _prep(t0: ArrayLike, t1: ArrayLike | None) -> tuple[np.ndarray, np.ndarray]:
    a0 = np.asarray(t0)
    a1 = a0 if t1 is None else np.asarray(t1)
    if a0.shape != a1.shape:
        raise ValueError("t0 and t1 must have the same length")
    if np.any(a0[1:] < a0[:-1]):
        raise ValueError("t0 must be sorted ascending")
    if np.any(a1 < a0):
        raise ValueError("t1 must be >= t0")
    return a0, a1


def purge_train(
    train: np.ndarray, test: np.ndarray, t0: np.ndarray, t1: np.ndarray, embargo: Any = 0
) -> np.ndarray:
    """Remove from ``train`` every sample that overlaps or is embargoed by ``test``.

    ``test`` may consist of several contiguous blocks; each is handled separately.
    """
    if test.size == 0:
        return train
    keep = np.ones(train.size, dtype=bool)
    # split test into contiguous runs of positions
    breaks = np.flatnonzero(np.diff(np.sort(test)) > 1)
    ts = np.sort(test)
    runs = np.split(ts, breaks + 1)
    tr0, tr1 = t0[train], t1[train]
    for run in runs:
        start, end = t0[run[0]], t1[run].max()
        overlap = (tr0 <= end) & (tr1 >= start)
        embargoed = (tr0 > end) & (tr0 <= end + embargo)
        keep &= ~(overlap | embargoed)
    return train[keep]


class WalkForward:
    """Expanding or rolling walk-forward with purging and embargo.

    Parameters
    ----------
    n_splits : number of test folds, taken from the end of the sample.
    min_train : minimum number of training samples before the first test fold.
    window : None for expanding; an int for a rolling window of that many samples.
    test_size : samples per test fold (default: split the post-min_train sample evenly).
    """

    def __init__(self, n_splits: int = 5, min_train: int = 0, window: int | None = None,
                 test_size: int | None = None, embargo: Any = 0) -> None:
        self.n_splits, self.min_train, self.window = n_splits, min_train, window
        self.test_size, self.embargo = test_size, embargo

    def split(self, t0: ArrayLike, t1: ArrayLike | None = None) -> Iterator[Split]:
        a0, a1 = _prep(t0, t1)
        n = a0.size
        size = self.test_size or (n - self.min_train) // self.n_splits
        if size <= 0:
            raise ValueError("not enough samples for the requested folds")
        first = n - size * self.n_splits
        for k in range(self.n_splits):
            lo, hi = first + k * size, first + (k + 1) * size
            test = np.arange(lo, hi)
            start = 0 if self.window is None else max(0, lo - self.window)
            train = np.arange(start, lo)
            train = purge_train(train, test, a0, a1, self.embargo)
            yield Split(train, test, (k,))


class PurgedKFold:
    """K contiguous folds; training data on both sides of the test fold, purged and embargoed."""

    def __init__(self, n_splits: int = 5, embargo: Any = 0) -> None:
        self.n_splits, self.embargo = n_splits, embargo

    def split(self, t0: ArrayLike, t1: ArrayLike | None = None) -> Iterator[Split]:
        a0, a1 = _prep(t0, t1)
        folds = np.array_split(np.arange(a0.size), self.n_splits)
        for k, test in enumerate(folds):
            train = np.setdiff1d(np.arange(a0.size), test, assume_unique=True)
            yield Split(purge_train(train, test, a0, a1, self.embargo), test, (k,))


class CombinatorialPurged:
    """Combinatorial purged cross-validation (CPCV).

    Time is cut into ``n_groups`` contiguous groups. Every combination of
    ``n_test`` groups is used once as the test set, giving C(n_groups, n_test)
    splits and ``n_paths = n_test * C / n_groups`` full out-of-sample backtest paths.
    """

    def __init__(self, n_groups: int = 6, n_test: int = 2, embargo: Any = 0) -> None:
        if not 0 < n_test < n_groups:
            raise ValueError("need 0 < n_test < n_groups")
        self.n_groups, self.n_test, self.embargo = n_groups, n_test, embargo

    @property
    def n_splits(self) -> int:
        return len(list(combinations(range(self.n_groups), self.n_test)))

    @property
    def n_paths(self) -> int:
        return self.n_test * self.n_splits // self.n_groups

    def groups(self, n: int) -> list[np.ndarray]:
        return np.array_split(np.arange(n), self.n_groups)

    def split(self, t0: ArrayLike, t1: ArrayLike | None = None) -> Iterator[Split]:
        a0, a1 = _prep(t0, t1)
        grp = self.groups(a0.size)
        for combo in combinations(range(self.n_groups), self.n_test):
            test = np.concatenate([grp[g] for g in combo])
            train = np.setdiff1d(np.arange(a0.size), test, assume_unique=True)
            yield Split(purge_train(train, test, a0, a1, self.embargo), test, combo)

    def paths(self) -> list[list[tuple[int, int]]]:
        """For each path, the (split_index, group) pairs that make up that path.

        Each group appears in exactly ``n_paths`` splits as a test group; path j
        takes the j-th of those occurrences, so every path covers all groups once.
        """
        combos = list(combinations(range(self.n_groups), self.n_test))
        seen: dict[int, list[int]] = {g: [] for g in range(self.n_groups)}
        for i, c in enumerate(combos):
            for g in c:
                seen[g].append(i)
        return [[(seen[g][j], g) for g in range(self.n_groups)] for j in range(self.n_paths)]

    def assemble_paths(self, n: int, preds: dict[tuple[int, ...], np.ndarray]) -> np.ndarray:
        """Stitch per-split test predictions into an (n_paths, n) array.

        ``preds`` maps each split's group tuple to its prediction vector over that
        split's test indices (in index order).
        """
        grp = self.groups(n)
        combos = list(combinations(range(self.n_groups), self.n_test))
        out = np.full((self.n_paths, n), np.nan)
        for j, path in enumerate(self.paths()):
            for split_i, g in path:
                combo = combos[split_i]
                test_idx = np.concatenate([grp[c] for c in combo])
                p = preds[combo]
                mask = np.isin(test_idx, grp[g])
                out[j, grp[g]] = p[mask]
        return out
