# Test Suite

## Running the tests

| Command | What runs | When to use |
|---|---|---|
| `python -m pytest` | Core + integration tests | Default: CI, quick sanity check |
| `python -m pytest -m "not slow"` | Fast core tests only | Before a commit on a login node |
| `python -m pytest -m slow` | Slow diagonalization + training tests | Full validation on a compute node |
| `python -m pytest -m script` | Script and plotting tests | When editing analysis scripts |
| `python -m pytest -m hardware` | CUDA-Q tests | When CUDA-Q is installed |
| `python -m pytest tests/test_peven.py` | Single file | Focused debugging |

## File index

### Core library tests
These test invariants of the physics and algorithms. Run them to verify a code change is
correct.

| File | What it covers |
|---|---|
| `test_adapt.py` | ADAPT-VQE algorithm: pool construction, stall detection, convergence |
| `test_config.py` | Config parsing, TOML roundtrip, enum coercion, CLI overrides |
| `test_dicke.py` | Dicke-sector evaluator: energy correctness, large-N efficiency |
| `test_energy.py` | `EnergyEvaluator` prefix energies, caching, batch evaluation |
| `test_exact_diag.py` | Exact diagonalization vs Dicke-sector, QPT curve, validators |
| `test_ghz_reference.py` | GHZ-X reference state correctness across backends |
| `test_lmg_hamiltonian.py` | LMG and Heisenberg Hamiltonian construction, coefficients |
| `test_markov_model.py` | MarkovGQE: forward shapes, generation, loss, determinism |
| `test_mean_field.py` | Mean-field angle formula, initial-energy product-state formula |
| `test_model.py` | GPT model: shapes, generation, loss, weight tying |
| `test_peven.py` | Parity-even evaluator, subspace tables, Rust/Python agreement |
| `test_peven_gpu.py` | GPU parity-even evaluator: gate-factor signs, energy correctness |
| `test_pools.py` | Pool construction: sizes, unitarity, parity, edge cases |
| `test_refine.py` | Angle refinement: monotone improvement, BOS guard |
| `test_shots.py` | Shot-based energy sampling: statistics, reproducibility |
| `test_state_quality.py` | State entropy, order parameter, doublet lifting |
| `test_statevec.py` | Full-space statevec evaluator and `core/statevec_ops.py` primitives |

### Integration test
| File | What it covers |
|---|---|
| `test_training_integration.py` | Full `train()` loop: artifacts, resume, BOS guard |

### Script tests (marker: `script`)
Excluded from the default run. Test standalone scripts and plotting utilities.

| File | What it covers |
|---|---|
| `test_benchmark_scaling.py` | `benchmark-scaling.py` internal helpers |
| `test_plots.py` | `analysis/plots.py` variant-label assignment |

### Hardware tests (marker: `hardware`)
Excluded from the default run. Require CUDA-Q to be installed.

| File | What it covers |
|---|---|
| `test_cudaq_evaluator.py` | CUDA-Q evaluator energy parity and caching |
