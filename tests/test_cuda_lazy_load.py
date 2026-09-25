"""Importing evaluator modules must not eagerly compile the CUDA extensions.

``torch.utils.cpp_extension.load`` is expensive (ninja invocation, ABI
checks) and pointless on a GPU-less node (e.g. a DeltaAI login node), where
the fused-CUDA path can never actually run. Regression test for the eager
"load-on-import" pattern in ``_statevec_cuda.py`` / ``_peven_cuda.py``.
"""

import subprocess
import sys


def _run_import_and_check_load_called(tmp_path, import_stmt: str) -> bool:
    marker = tmp_path / "load_called"
    code = f"""
import torch.utils.cpp_extension as ext

def _mark(*args, **kwargs):
    open({str(marker)!r}, "w").close()
    raise RuntimeError("blocked for test: cpp_extension.load should be lazy")

ext.load = _mark
{import_stmt}
"""
    subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return marker.exists()


def test_importing_statevec_gpu_does_not_compile_cuda_extension(tmp_path):
    called = _run_import_and_check_load_called(
        tmp_path, "import spingqe_lmg.evaluators.statevec_gpu"
    )
    assert not called, "importing statevec_gpu triggered torch.utils.cpp_extension.load"


def test_importing_spingqe_lmg_does_not_compile_cuda_extension(tmp_path):
    called = _run_import_and_check_load_called(tmp_path, "import spingqe_lmg")
    assert not called, "importing spingqe_lmg triggered torch.utils.cpp_extension.load"
