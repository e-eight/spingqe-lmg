import os
import random
import subprocess

import numpy as np
import pytest


def _gpu_present():
    try:
        subprocess.run(
            ["nvidia-smi", "-L"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        return True
    except Exception:
        return False


if not _gpu_present():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch


@pytest.fixture(autouse=True, scope="function")
def seed_everything():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(7)
