"""Leakage-resistant validation toolkit (López de Prado, AFML).

Public API
----------
- ``PurgedKFold``            : purged + embargoed K-fold splitter.
- ``CombinatorialPurgedCV``  : CPCV splitter with back-test path accounting.
- ``get_train_times`` / ``get_embargo_times`` : purging/embargo primitives.
- ``probabilistic_sharpe_ratio`` / ``deflated_sharpe_ratio`` / ``expected_max_sharpe``.

Nothing here contains strategy logic, features, or tuned parameters — only the
validation machinery described in the methodology notes under ``docs/``.
"""

from .purged_cv import (
    PurgedKFold,
    CombinatorialPurgedCV,
    get_train_times,
    get_embargo_times,
)
from .deflated_sharpe import (
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    expected_max_sharpe,
)

__all__ = [
    "PurgedKFold",
    "CombinatorialPurgedCV",
    "get_train_times",
    "get_embargo_times",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
]
