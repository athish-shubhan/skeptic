"""Multiple-testing corrections. All functions return adjusted p-values."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def bonferroni(p: ArrayLike) -> np.ndarray:
    pv = np.asarray(p, dtype=float)
    return np.minimum(pv * pv.size, 1.0)


def holm(p: ArrayLike) -> np.ndarray:
    """Holm (1979) step-down adjusted p-values (controls FWER)."""
    pv = np.asarray(p, dtype=float)
    m = pv.size
    order = np.argsort(pv)
    adj_sorted = np.maximum.accumulate((m - np.arange(m)) * pv[order])
    out = np.empty(m)
    out[order] = np.minimum(adj_sorted, 1.0)
    return out


def benjamini_hochberg(p: ArrayLike) -> np.ndarray:
    """Benjamini-Hochberg (1995) step-up adjusted p-values (controls FDR)."""
    pv = np.asarray(p, dtype=float)
    m = pv.size
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj_sorted, 1.0)
    return out


METHODS = {"none": lambda p: np.asarray(p, dtype=float), "bonferroni": bonferroni,
           "holm": holm, "bh": benjamini_hochberg}


def adjust(p: ArrayLike, method: str = "holm") -> np.ndarray:
    try:
        return METHODS[method](p)
    except KeyError as e:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(METHODS)}") from e
