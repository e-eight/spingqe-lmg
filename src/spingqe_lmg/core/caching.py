"""Generic LRU cache and the shared evaluator ``__call__`` mixin.

Used by all five evaluator backends (statevector CPU/GPU, Dicke, CUDA-Q,
PennyLane) -- not statevector-specific, despite historically living in
_statevec_core.py.
"""

from collections import OrderedDict

import numpy as np


class LRUCache(OrderedDict):
    """OrderedDict with optional max-size LRU eviction."""

    def __init__(self, maxsize: int | None):
        super().__init__()
        self.maxsize = maxsize

    def __reduce__(self):
        # OrderedDict.__reduce__ reconstructs via a no-arg cls() call, which fails
        # here since maxsize is required; supply it explicitly and let pickle
        # restore contents via the dictitems iterator (5th element).
        return (self.__class__, (self.maxsize,), None, None, iter(self.items()))

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.maxsize is not None and len(self) > self.maxsize:
            self.popitem(last=False)

    def __getitem__(self, key):
        self.move_to_end(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


class CachedEvaluatorMixin:
    """Shared ``__call__`` for evaluators: cache lookup, compute misses, fill, stack.

    Subclasses must set ``self._cache`` (an ``LRUCache``) and implement
    ``_sequence_energies(key: tuple[int, ...]) -> np.ndarray``. Set
    ``self.n_jobs`` to a value other than 1 to dispatch cache misses through
    ``joblib.Parallel`` instead of a serial loop; evaluators that don't set
    ``self.n_jobs`` always run serially.
    """

    def __call__(self, idx_batch: np.ndarray) -> np.ndarray:
        keys = [tuple(int(i) for i in row) for row in idx_batch]
        missing = [k for k in dict.fromkeys(keys) if k not in self._cache]
        if missing:
            n_jobs = getattr(self, "n_jobs", 1)
            if n_jobs == 1:
                results = [self._sequence_energies(k) for k in missing]
            else:
                from joblib import Parallel, delayed

                results = Parallel(n_jobs=n_jobs)(
                    delayed(self._sequence_energies)(k) for k in missing
                )
            self._cache.update(zip(missing, results, strict=True))
        return np.stack([self._cache[k] for k in keys])
