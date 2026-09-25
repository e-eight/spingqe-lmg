"""SpinGQE extended to the Lipkin-Meshkov-Glick model with individual spin encoding.

Entry point: load a TOML config with ``load_config``, then call ``train``.
"""

__all__ = [
    "EvaluatorProtocol",
    "RunConfig",
    "TrainResult",
    "build_evaluator",
    "load_config",
    "train",
]

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("spingqe-lmg")
except PackageNotFoundError:
    __version__ = "unknown"

from spingqe_lmg.config import RunConfig, load_config
from spingqe_lmg.evaluators import EvaluatorProtocol, build_evaluator
from spingqe_lmg.training import TrainResult, train
