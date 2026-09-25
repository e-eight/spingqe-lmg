# SpinGQE-LMG

Code and data for the paper *Deciding whether a generative quantum eigensolver
can work before running it* (Soham Pal, NCSA, University of Illinois
Urbana-Champaign).

The package applies the SpinGQE variant of the generative quantum eigensolver
(GQE) to the Lipkin-Meshkov-Glick (LMG) model. It provides three operator pools
(collective, standard pairwise and extended pairwise), two reference states
(zero and mean-field), and several energy evaluators: PennyLane, full-space
incremental statevector (CPU and GPU), parity-even statevector (CPU and GPU),
CUDA-Q, and an (N+1)-dimensional permutation-symmetric evaluator.

All results are classical simulations. Every run used one GPU allocation on
the DeltaAI system at NCSA (NVIDIA GH200: 72-core Grace CPU + H100 GPU).

## Install

```bash
pip install -e ".[dev]"      # add ".[cudaq]" for the CUDA-Q evaluator
python -m pytest             # default suite; see tests/README.md for markers
```

The code needs Python >= 3.11, PennyLane >= 0.39, PyTorch >= 2.0, NumPy,
SciPy, pandas, joblib and matplotlib. `uv.lock` pins the exact versions used.
The GPU evaluators need a CUDA GPU and `numba.cuda`.

## Quickstart

```bash
python examples/quickstart.py                                    # raw API, no training
spingqe-train --config configs/quickstart.toml --out-dir results/quickstart
spingqe-refine --run-dir results/quickstart                      # angle refinement
spingqe-exact --n 4 8 12 16 --out-dir results/                   # exact energies
```

## Repository layout

| Path | Contents |
|---|---|
| `src/spingqe_lmg/` | The library: Hamiltonians, pools, evaluators, GQE model, training, refinement, CLI |
| `tests/` | pytest suite |
| `configs/` | Sweep configs for the paper's training campaigns |
| `scripts/` | The generators for every table, figure and in-text number in the paper |
| `data/tables/` | Derived CSV/JSON files that the tables and figures are built from |
| `data/runs/` | Per-run records of every training campaign and scan the paper uses |
| `figures/` | The paper's two figures, as regenerated from `data/` |

## Reproducing the paper

Run every command from the repository root. None of the commands below
retrain anything: they rebuild each table, figure and number from `data/`,
and all of them run on CPU.

| Paper item | Command | Input |
|---|---|---|
| Table: throughput (tab:throughput) | `python scripts/table-benchmark.py throughput data/tables/bench-merged-2026-07-08.csv` (the paper prints the l.gpu (g), Incr GPU, P.-E. GPU and Symm columns) | `bench-merged-2026-07-08.csv` |
| Peak-RSS figures in the text | `python scripts/table-benchmark.py rss data/tables/bench-merged-2026-07-08.csv` | same |
| Fig.: permutation-symmetric showcase to N=1000 | `python scripts/plot-dicke-showcase.py` | `data/runs/dicke-showcase-{mf,zero}-seeds-2026-08-06` |
| Table: descent test (tab:descent) | `python scripts/descent-test.py --json data/tables/descent-test-n16-lam2.0.json`; the last column is `python scripts/gradient-probe.py` | computed directly, a few minutes on one core |
| Fig.: onset scan at N=16 | `python scripts/plot-reachability-regimes.py` | `data/tables/lreach-2026-07-26.csv` |
| Table: onset (tab:onset) | `python scripts/paper-numbers.py` (first block) | `lreach-2026-07-26.csv`, `lreach-n28-2026-07-28.csv` |
| Table: collective accuracy at L=12 (tab:pool-accuracy) | `python scripts/table-pool-accuracy.py data/tables/pool-accuracy-l12-s1-10-2026-07-29.csv --latex --collective-only --floor 1e-10 --stat median` | `pool-accuracy-l12-s1-10-2026-07-29.csv` |
| Table: pool vs reference (tab:2x2) | the `upper` block of `data/tables/2x2-table-2026-07-29.csv` | same |
| In-text numbers: onset-scan cost, floor lengths at L=26, the N=28 pre-registered arms, pairwise spreads, the uniform-sampler control, the N=1000 endpoint, the refinement-backend check | `python scripts/paper-numbers.py` | `data/` |

Each block of `paper-numbers.py` prints the value the paper states next to
the value it recomputes.

### Rebuilding the derived tables from the run records

```bash
python scripts/make-pool-accuracy-10seed-csv.py   # -> data/tables/pool-accuracy-l12-s1-10-2026-07-29.csv
python scripts/make-2x2-table-csv.py              # -> data/tables/2x2-table-2026-07-29.csv
python scripts/make-lreach-csv.py --sweep-dir data/runs/reachability-depth-grid-2026-07-26 \
    --out data/tables/lreach-2026-07-26.csv
python scripts/make-lreach-csv.py --sweep-dir data/runs/prospective-depth-n28-2026-07-28/scan \
    --out data/tables/lreach-n28-2026-07-28.csv
```

Each command regenerates its shipped file exactly.

### Re-running the campaigns

Training is GPU work: 701 epochs of 10 sequences each, then best-of-M
refinement. On one GH200 a run takes minutes to hours, depending on N and the
evaluator.

- **Training sweeps.** `spingqe-sweep-plan --config configs/<file>.toml --runs-root <dir>`
  expands a config into one run per manifest row; run each row with
  `spingqe-train`. Each config's header names the run directory it produced.
  Every run directory also holds the exact `config.json` it trained with, so
  `spingqe-train --config <run>/config.json --out-dir <new-dir>` repeats a
  single run. This is the only route for `collective-dicke-l26`, whose runs
  came from per-submission edits of one template config. Its configs predate
  the `[train.pennylane]` section: they load unchanged, but they record the
  `default.qubit` device, which the training CLI refuses. Add
  `--set train.pennylane.device=lightning.qubit`. The Dicke evaluator these runs
  use does not touch that device.
- **Refinement.** `spingqe-refine --run-dir <run> --generate M --top M` samples M
  candidate sequences from a trained model and refines their angles with
  L-BFGS-B.
- **Onset scans.** `python scripts/reachability-depth-scan.py --pool <pool> --n-qubits ... --lam ... --seq-len ...`
  runs random restarts refined directly, with no training. Each
  `data/runs/*/manifest.csv` lists the grid that was run.
- **Uniform-sampler control.** Run `scripts/random-sampler-control-grid.py`, which
  calls `scripts/random-sampler-control.py`. It needs the paired trained runs
  (the three-seed campaign behind `data/tables/pool-accuracy-l12-s123-2026-07-25.csv`).
  Those runs are not shipped, but their refined errors are in that CSV and in
  the control's `summary.csv`.
- **Evaluator benchmark.** `scripts/benchmark-scaling.py` measures
  per-evaluator throughput and peak RSS, and `scripts/merge-benchmark-csvs.py`
  merges the per-backend outputs. `scripts/refine-backend-comparison.py`
  compares refinement backends.

## Data

`data/runs/<campaign>/<run>/` holds one directory per training run:

| File | Contents |
|---|---|
| `config.json` | The full configuration of the run |
| `metadata.json` | Exact ground-state energy, reference-state energy, best raw and refined energies, and the software versions |
| `best_sequence.json` | The best token sequence and its refined angles |
| `losses.csv` | Per-epoch training loss |
| `eval.csv` | Periodic diagnostic evaluations |

Onset scans store one `cells/cell-*.csv` per (pool, N, lambda, L), with one row
per random restart. Model checkpoints (`*.pt`, about 180 GB in total) and Slurm
logs are not included. In manifests and metadata, absolute cluster paths were
rewritten as repository-relative ones.

| Campaign | Used for |
|---|---|
| `pool-accuracy-l12-s1-10-2026-07-28`, `pool-accuracy-pairwise-l12-s1-10-2026-07-29` | The full 51-cell grid at L=12, 10 seeds per cell (tab:pool-accuracy and the pairwise prose) |
| `pool-vs-reference-2026-07-26`, `collective-mf-n16-2026-07-30` | tab:2x2 and the collective mean-field contrast |
| `dicke-showcase-mf-seeds-2026-08-06`, `dicke-showcase-zero-seeds-2026-08-06` | Fig. dicke-showcase, N=50 to 1000 |
| `reachability-depth-grid-2026-07-26` | Onset scan, N=8 to 24 (12 restarts per cell) |
| `prospective-depth-n28-2026-07-28` | The pre-registered N=28 test: onset scan (`scan/`) and the L=68/72 training arms (`train/`) |
| `collective-dicke-l26` | Floor length at L=26, N=4 to 20, 6 seeds |
| `random-sampler-control-2026-07-25` | The uniform-sampler control (153 pairs) |

`data/tables/n1000-mf-timing-sacct.csv` records the Slurm accounting wall time
of the ten N=1000 mean-field runs.

Some run settings are recorded only in part:

- **Refinement budget M.** The number of refinement candidates is not stored in
  `config.json`. The `collective-dicke-l26` and `prospective-depth-n28` runs
  were refined with `spingqe-refine --generate 50 --top 50`.
- **Angle vocabulary.** These are set by `pool.angle_scale` in each run's
  `config.json`, where 8 means the 8/N-rescaled vocabulary and 0 means
  unscaled. `collective-dicke-l26` and `prospective-depth-n28` use 8.

## License

MIT (see `LICENSE`). The GQE model, the energy-evaluation structure and the
training loop are adapted from SpinGQE and nanoGPT, both MIT-licensed; see
`THIRD_PARTY_NOTICES.md`.
