"""Enumerated option types used throughout the configuration system.

Replaces bare-string comparisons with exhaustiveness-checkable values.
Enum members carry the same string values that TOML configs use, so existing
config files continue to work unmodified.
"""

from enum import Enum


class HamiltonianKind(Enum):
    LMG = "lmg"
    HEISENBERG_XXZ = "heisenberg_xxz"


class PoolKind(Enum):
    PAULI_PAIR = "pauli_pair"
    COLLECTIVE = "collective"


class Connectivity(Enum):
    ALL = "all"
    NN = "nn"


class ModelKind(Enum):
    GPT = "gpt"
    MARKOV = "markov"


class EvaluatorKind(Enum):
    PENNYLANE = "pennylane"
    CUDAQ = "cudaq"
    DICKE = "dicke"
    INCREMENTAL = "incremental"
    PARITY_EVEN = "parity_even"
    PARITY_EVEN_GPU = "parity_even_gpu"
    INCREMENTAL_GPU = "incremental_gpu"


class InitState(Enum):
    ZERO = "zero"
    MEAN_FIELD = "mean_field"
    GHZ_X = "ghz_x"


class DeviceKind(Enum):
    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


class EnergyNorm(Enum):
    NONE = "none"
    SHIFT_SCALE = "shift_scale"


class RefineOptimizer(Enum):
    SPSA = "spsa"
    COBYLA = "cobyla"


class RefineGradient(Enum):
    ADJOINT = "adjoint"
    FINITE_DIFF = "finite_diff"


class Variant(Enum):
    """Derived label for a pool / reference / Pauli-word combination.

    Computed by ``variant_column()`` for use as a categorical variable in
    plots and analysis tables.  Members carry the same string values stored
    in aggregate CSVs so existing data files continue to work unmodified.
    """

    COLLECTIVE = "collective"
    PAIRWISE_EXT_MF = "pairwise-ext-mf"
    PAIRWISE_ALL = "pairwise-all"
    PAIRWISE_NN = "pairwise-nn"
