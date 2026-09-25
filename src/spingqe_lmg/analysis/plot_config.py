"""Shared helpers for plot scripts: config loading, seed-count reporting."""

import sys
import tomllib

import pandas as pd


def resolve_config(
    args,
    required: list[str],
    config_keys: tuple[str, ...] = ("data", "output"),
) -> dict:
    """Merge TOML config with CLI overrides; validate required keys are present.

    ``args`` is an argparse.Namespace carrying --config plus per-field
    CLI flags matching the TOML keys.  Returns a flat dict.
    """
    cfg: dict = {}
    if args.config:
        with open(args.config, "rb") as f:
            raw = tomllib.load(f)
        for section in config_keys:
            cfg.update(raw.get(section, {}))

    # CLI flags override TOML values
    for key in required:
        val = getattr(args, key.replace("-", "_"), None)
        if val is not None:
            cfg[key] = val

    missing = [k for k in required if k not in cfg]
    if missing:
        print(
            f"Error: missing required field(s): {', '.join(missing)}\n"
            f"Supply via --config or individual CLI flags.",
            file=sys.stderr,
        )
        sys.exit(1)

    return cfg


def print_seed_counts(
    label: str,
    df: pd.DataFrame,
    variants: list[str],
) -> None:
    """Print seed-count ranges per variant across (N, lam)."""
    print(f"\n{label}:")
    for v in variants:
        sub = df[df["variant"] == v]
        if sub.empty:
            print(f"  {v}: (no data)")
            continue
        counts = sub.groupby(["n_qubits", "lam"])["seed"].count()
        print(
            f"  {v}: {len(sub)} rows, "
            f"seed range {counts.min()}-{counts.max()}, "
            f"N={sorted(sub.n_qubits.unique())}"
        )
