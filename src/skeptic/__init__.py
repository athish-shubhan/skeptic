"""Skeptic: a small research library that makes false discoveries hard to produce."""

from .boot import MAX_DD, MEAN, MEDIAN, SHARPE, block_bootstrap_by_group, bootstrap
from .costs import FixedBps, ImpactCost, SpreadCost, backtest
from .ledger import Ledger, optuna_callback
from .multiple import adjust, benjamini_hochberg, bonferroni, holm
from .pbo import pbo
from .pit import LookaheadError, PITStore
from .splits import CombinatorialPurged, PurgedKFold, WalkForward, purge_train
from .stats import (
    annualise,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe,
    sharpe_ci,
)

__version__ = "0.1.0"
__all__ = [
    "MAX_DD", "MEAN", "MEDIAN", "SHARPE", "CombinatorialPurged", "FixedBps", "ImpactCost",
    "Ledger", "LookaheadError", "PITStore", "PurgedKFold", "SpreadCost", "WalkForward",
    "adjust", "annualise", "backtest", "benjamini_hochberg", "block_bootstrap_by_group",
    "bonferroni", "bootstrap", "deflated_sharpe_ratio", "expected_max_sharpe", "holm",
    "min_track_record_length", "optuna_callback", "pbo", "probabilistic_sharpe_ratio",
    "purge_train", "sharpe", "sharpe_ci",
]
