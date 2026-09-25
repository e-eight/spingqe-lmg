# Third-Party Notices

This project ports and adapts code from the following MIT-licensed projects:

## SpinGQE

- Source: https://github.com/Mindbeam-AI/SpinGQE
- License: MIT
- Adapted: the GQE model (`src/spingqe_lmg/model/gqe.py` — cumulative-logit loss,
  Boltzmann sequence generation), the snapshot-based per-step energy evaluation
  (`src/spingqe_lmg/energy.py`), and the overall training-loop structure
  (`src/spingqe_lmg/training.py`).

## nanoGPT

- Source: https://github.com/karpathy/nanoGPT
- License: MIT
- Adapted: the GPT transformer implementation (`src/spingqe_lmg/model/gpt.py`),
  ported via the SpinGQE repository's copy of `model.py`.
