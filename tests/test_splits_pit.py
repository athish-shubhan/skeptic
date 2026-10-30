import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from skeptic.pit import LookaheadError, PITStore
from skeptic.splits import CombinatorialPurged, PurgedKFold, WalkForward


def _no_overlap(sp, t0, t1, embargo=0):
    """Contiguous test fold: no train label window may touch it or start inside the embargo."""
    tr, te = sp.train, sp.test
    if tr.size == 0 or te.size == 0:
        return True
    start, end = t0[te].min(), t1[te].max()
    overlap = (t0[tr] <= end) & (t1[tr] >= start)
    embargoed = (t0[tr] > end) & (t0[tr] <= end + embargo)
    return not (overlap | embargoed).any()


@settings(max_examples=30, deadline=None)
@given(st.integers(50, 300), st.integers(1, 20), st.integers(0, 10), st.integers(2, 6))
def test_purged_kfold_no_overlap(n, h, emb, k):
    t0 = np.arange(n)
    t1 = t0 + h
    for sp in PurgedKFold(k, embargo=emb).split(t0, t1):
        assert np.intersect1d(sp.train, sp.test).size == 0
        assert _no_overlap(sp, t0, t1, emb)


def test_walk_forward_train_precedes_test():
    t0 = np.arange(1000)
    for sp in WalkForward(n_splits=4, min_train=200, embargo=5).split(t0, t0 + 10):
        assert sp.train.max() + 10 < sp.test.min()


def test_rolling_window_size():
    t0 = np.arange(1000)
    for sp in WalkForward(n_splits=4, min_train=200, window=150).split(t0):
        assert sp.train.size <= 150


def test_cpcv_counts_and_paths():
    cv = CombinatorialPurged(n_groups=6, n_test=2)
    assert cv.n_splits == 15
    assert cv.n_paths == 5
    n = 600
    splits = list(cv.split(np.arange(n)))
    preds = {sp.group: sp.test.astype(float) for sp in splits}  # prediction = own index
    paths = cv.assemble_paths(n, preds)
    assert paths.shape == (5, n)
    assert not np.isnan(paths).any()
    assert np.all(paths == np.arange(n))  # every path covers every index exactly once


def test_pit_store_blocks_unreleased_rows():
    df = pd.DataFrame({
        "month": pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31"], utc=True),
        "release": pd.to_datetime(["2025-02-12", "2025-03-12", "2025-04-10"], utc=True),
        "value": [1.0, 2.0, 3.0],
    })
    s = PITStore()
    s.add("cpi", df, event_time="month", available_at="release")
    v = s.view(pd.Timestamp("2025-03-31", tz="UTC"))
    assert v.get("cpi")["value"].tolist() == [1.0, 2.0]  # March CPI not released yet
    with pytest.raises(LookaheadError):
        s.view(pd.Timestamp("2025-01-01", tz="UTC")).latest("cpi")
    al = s.align("cpi", pd.to_datetime(["2025-03-11", "2025-03-12"], utc=True), ["value"])
    assert al["value"].tolist() == [1.0, 2.0]


def test_pit_requires_explicit_latency():
    df = pd.DataFrame({"t": pd.to_datetime(["2025-01-01"], utc=True), "x": [1]})
    with pytest.raises(ValueError):
        PITStore().add("x", df, event_time="t")
    s = PITStore()
    s.add("x", df, event_time="t", latency=pd.Timedelta("1min"))
    assert s.view(pd.Timestamp("2025-01-01 00:00:30", tz="UTC")).get("x").empty
