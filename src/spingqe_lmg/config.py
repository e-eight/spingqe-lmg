"""Run configuration: frozen dataclasses + TOML loading + dotted CLI overrides.

A run is fully described by a TOML file with [hamiltonian], [pool], [model],
[train] tables (all keys optional). Values can be overridden on the command
line with repeated ``--set section.key=value`` flags.

All option-valued fields use Python enums; TOML strings and CLI overrides are
coerced automatically. Cross-parameter constraints are validated at parse time;
see _validate_cross_constraints for the full list.

Pool/evaluator compatibility note: the Dicke evaluator requires the collective
pool and the LMG Hamiltonian (it exploits permutation symmetry).  The collective
pool itself is compatible with any evaluator — Dicke is simply the most
efficient choice because it avoids the full 2^N statevector.  Parity-even
evaluators work with pairwise pools whose operators are parity-even (the
default LMG pools satisfy this).
"""

import dataclasses
import enum
import json
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, get_args, get_type_hints

from spingqe_lmg.enums import (
    Connectivity,
    DeviceKind,
    EnergyNorm,
    EvaluatorKind,
    HamiltonianKind,
    InitState,
    ModelKind,
    PoolKind,
    RefineGradient,
    RefineOptimizer,
)
from spingqe_lmg.pools import COLLECTIVE_GENERATORS, DEFAULT_ANGLES


@dataclass(frozen=True)
class HamiltonianConfig:
    kind: HamiltonianKind = HamiltonianKind.LMG
    n_qubits: int = 4
    h: float = 1.0  # transverse field
    lam: float = 0.5  # coupling; QPT at lam = h
    gamma: float = 0.0  # anisotropy (Jy^2 weight)
    j1: float = 10.0  # heisenberg_xxz couplings (reference baseline values)
    j2: float = 10.0

    def __post_init__(self):
        if self.kind is HamiltonianKind.LMG:
            if self.n_qubits < 2 or self.n_qubits % 2 != 0:
                raise ValueError(f"n_qubits must be a positive even integer (got {self.n_qubits})")


@dataclass(frozen=True)
class PoolConfig:
    kind: PoolKind = PoolKind.PAULI_PAIR
    connectivity: Connectivity = Connectivity.ALL
    paulis: tuple[str, ...] = ("ZZ", "XX", "YY")
    include_single_z: bool = True
    angles: tuple[float, ...] = DEFAULT_ANGLES
    generators: tuple[str, ...] = COLLECTIVE_GENERATORS
    angle_scale: float = 0.0


@dataclass(frozen=True)
class ModelConfig:
    kind: ModelKind = ModelKind.GPT
    order: int = 1
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.2
    bias: bool = False


@dataclass(frozen=True)
class PennyLaneConfig:
    """Config for the PennyLane circuit evaluator (shots, device, refinement)."""

    device: str = "default.qubit"
    shots: int = 0
    refine_shots: int = 0
    refine_optimizer: RefineOptimizer = RefineOptimizer.SPSA
    refine_gradient: RefineGradient = RefineGradient.ADJOINT


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 700
    # 0-indexed last epoch. Training iterates epochs 0, 1, ..., epochs inclusive
    # (total: epochs+1 steps). Setting epochs=700 produces 701 training steps.
    seq_gen: int = 10
    seq_len: int = 12
    n_batches: int = 10
    temperature: float = 0.5
    beta: float = 0.3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    eval_iter: int = 50
    eval_sequences: int = 100
    seed: int = 1
    device: DeviceKind = DeviceKind.AUTO
    n_jobs: int = 1
    checkpoint_every: int = 50
    energy_norm: EnergyNorm = EnergyNorm.NONE
    refine: bool = True
    init_state: InitState = InitState.ZERO
    evaluator: EvaluatorKind = EvaluatorKind.PENNYLANE
    pennylane: PennyLaneConfig | None = None
    save_final_checkpoint: bool = True


@dataclass(frozen=True)
class RunConfig:
    run_name: str = "run"
    hamiltonian: HamiltonianConfig = field(default_factory=HamiltonianConfig)
    pool: PoolConfig = field(default_factory=PoolConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


_SECTIONS = {
    "hamiltonian": HamiltonianConfig,
    "pool": PoolConfig,
    "model": ModelConfig,
    "train": TrainConfig,
}


def _coerce_enum(enum_cls: type[enum.Enum], value: Any) -> enum.Enum:
    """Coerce a string or Enum value to the given Enum type."""
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        try:
            return enum_cls(value)
        except ValueError:
            raise ValueError(
                f"unknown {enum_cls.__name__} value {value!r}; valid: {[m.value for m in enum_cls]}"
            ) from None
    raise TypeError(f"cannot coerce {type(value)} to {enum_cls.__name__}")


def _unwrap_optional(field_type):
    """If field_type is T | None, return T; otherwise return field_type."""
    args = get_args(field_type)
    if args:
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return field_type


def _build_section(cls, raw: dict):
    """Construct a dataclass from a raw dict, coercing string -> enum and
    dict -> nested dataclass where the type annotations indicate."""
    hints = get_type_hints(cls)
    valid = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(valid)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    coerced = {}
    for key, value in raw.items():
        if isinstance(value, list):
            value = tuple(value)
        field_type = hints.get(key)
        if isinstance(field_type, type) and issubclass(field_type, enum.Enum):
            value = _coerce_enum(field_type, value)
        elif field_type is bool and isinstance(value, str):
            if value.lower() not in ("true", "false"):
                raise ValueError(f"invalid bool override for {key!r}: {value!r}")
            value = value.lower() == "true"
        coerced[key] = value
    for field_info in dataclasses.fields(cls):
        field_type = hints.get(field_info.name)
        if (
            field_type is not None
            and field_info.name in coerced
            and isinstance(coerced[field_info.name], dict)
        ):
            inner_type = _unwrap_optional(field_type)
            if inner_type is not None and dataclasses.is_dataclass(inner_type):
                coerced[field_info.name] = _build_section(inner_type, coerced[field_info.name])
    return cls(**coerced)


def _validate_cross_constraints(cfg: RunConfig) -> None:
    """Check that config parameters are mutually compatible.

    Enforced constraints (all are errors):
    - evaluator='dicke'   → hamiltonian.kind='lmg'  (Dicke basis requires LMG)
    - evaluator='dicke'   → init_state ≠ 'ghz_x'
    - evaluator='cudaq'   → init_state ≠ 'ghz_x'
    - init_state='mean_field' → hamiltonian.kind='lmg'
    - pennylane.refine_shots > 0 → pennylane.shots > 0

    Note on pool/evaluator direction: the Dicke evaluator requires the
    collective pool, not the other way around.  The collective pool is valid
    with any evaluator; Dicke is simply O(N) instead of O(2^N).
    """
    tc = cfg.train
    hc = cfg.hamiltonian

    if tc.evaluator is EvaluatorKind.DICKE and hc.kind is not HamiltonianKind.LMG:
        raise ValueError("evaluator='dicke' requires hamiltonian.kind='lmg'")

    if tc.evaluator is EvaluatorKind.DICKE and tc.init_state is InitState.GHZ_X:
        raise ValueError("evaluator='dicke' does not support init_state='ghz_x'")

    if tc.evaluator is EvaluatorKind.CUDAQ and tc.init_state is InitState.GHZ_X:
        raise ValueError("evaluator='cudaq' does not support init_state='ghz_x'")

    if tc.init_state is InitState.MEAN_FIELD and hc.kind is not HamiltonianKind.LMG:
        raise ValueError("init_state='mean_field' is only defined for the LMG model")

    if (
        tc.pennylane is not None
        and tc.pennylane.shots == 0
        and tc.evaluator is EvaluatorKind.PENNYLANE
        and tc.pennylane.refine_shots > 0
    ):
        raise ValueError("refine_shots>0 requires shots>0")


# Keys that older run configs (config.json written before the PennyLane options
# moved into [train.pennylane]) carry directly under [train].
_LEGACY_PENNYLANE_KEYS = {
    "qml_device": "device",
    "shots": "shots",
    "refine_shots": "refine_shots",
    "refine_optimizer": "refine_optimizer",
    "refine_gradient": "refine_gradient",
}


def _migrate_legacy_train(train: dict[str, Any]) -> dict[str, Any]:
    """Move pre-[train.pennylane] keys of an old run config into train.pennylane."""
    if not set(train) & set(_LEGACY_PENNYLANE_KEYS):
        return train
    train = dict(train)
    pennylane = dict(train.get("pennylane") or {})
    for old, new in _LEGACY_PENNYLANE_KEYS.items():
        if old in train:
            value = train.pop(old)
            pennylane.setdefault(new, int(value) if isinstance(value, bool) else value)
    train["pennylane"] = pennylane
    return train


def config_from_dict(data: dict[str, Any]) -> RunConfig:
    data = dict(data)
    data.pop("sweep", None)
    if isinstance(data.get("train"), dict):
        data["train"] = _migrate_legacy_train(data["train"])
    kwargs: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        kwargs[name] = _build_section(cls, data.pop(name, {}))
    run_name = data.pop("run_name", "run")
    if data:
        raise ValueError(f"unknown top-level config keys: {sorted(data)}")
    cfg = RunConfig(run_name=run_name, **kwargs)
    _validate_cross_constraints(cfg)
    return cfg


def _parse_override_value(raw: str) -> Any:
    try:
        parsed = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw  # bare string
    if isinstance(parsed, bool):
        return raw  # "true"/"false" stay as strings for enum coercion
    return parsed


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` (or ``run_name=value``) overrides in place."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        value = _parse_override_value(raw.strip())
        parts = dotted.strip().split(".")
        target = data
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return data


def load_config(path: str | Path, overrides: list[str] | None = None) -> RunConfig:
    """Load a run config from TOML, or from a run directory's ``config.json``."""
    with open(path, "rb") as f:
        data = json.load(f) if Path(path).suffix == ".json" else tomllib.load(f)
    for derived in ("vocab_size", "pool_size"):  # written into config.json by train()
        data.pop(derived, None)
    if overrides:
        apply_overrides(data, overrides)
    return config_from_dict(data)


def config_to_dict(cfg: RunConfig) -> dict[str, Any]:
    def _convert(obj):
        if isinstance(obj, enum.Enum):
            return obj.value
        if dataclasses.is_dataclass(obj):
            return {f.name: _convert(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return obj

    return _convert(cfg)


# ---------------------------------------------------------------------------
# CLI manifest utilities (pure data transforms)
# ---------------------------------------------------------------------------

MANIFEST_FIXED_COLUMNS = ("row", "run_id", "config", "out_dir")


def manifest_row_to_overrides(row: dict[str, str]) -> list[str]:
    """Convert a manifest CSV row to TOML override strings.

    Columns named in MANIFEST_FIXED_COLUMNS are skipped; empty values are
    skipped.  Returns list of ``"section.key=value"`` strings.
    """
    return [
        f"{key}={value}"
        for key, value in row.items()
        if key not in MANIFEST_FIXED_COLUMNS and value not in (None, "")
    ]


def toml_literal(value) -> str:
    """Serialize a Python value as a TOML-compatible literal string.

    Uses JSON encoding because JSON literals are valid TOML for strings,
    numbers, booleans, and arrays.  ``repr()`` is not suitable (it produces
    single-quoted strings that TOML rejects).
    """
    if value is None:
        return ""
    return json.dumps(value)
