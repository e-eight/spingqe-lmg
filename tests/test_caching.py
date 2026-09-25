import numpy as np

from spingqe_lmg.core.caching import CachedEvaluatorMixin, LRUCache


class _DummyEvaluator(CachedEvaluatorMixin):
    def __init__(self, n_jobs: int = 1, cache_maxsize=None):
        self.n_jobs = n_jobs
        self._cache = LRUCache(cache_maxsize)
        self.calls: list[tuple[int, ...]] = []

    def _sequence_energies(self, key: tuple[int, ...]) -> np.ndarray:
        self.calls.append(key)
        return np.array([float(sum(key))] * len(key))


class _NoNJobsEvaluator(CachedEvaluatorMixin):
    def __init__(self):
        self._cache = LRUCache(None)
        self.calls: list[tuple[int, ...]] = []

    def _sequence_energies(self, key: tuple[int, ...]) -> np.ndarray:
        self.calls.append(key)
        return np.array([1.0] * len(key))


def test_cached_evaluator_mixin_computes_and_caches_misses():
    ev = _DummyEvaluator()
    idx = np.array([[0, 1], [0, 1]])
    result = ev(idx)
    np.testing.assert_array_equal(result, [[1.0, 1.0], [1.0, 1.0]])
    assert ev.calls == [(0, 1)], "duplicate rows in one batch must compute once"


def test_cached_evaluator_mixin_reuses_cache_across_calls():
    ev = _DummyEvaluator()
    ev(np.array([[0, 1]]))
    ev(np.array([[0, 1]]))
    assert ev.calls == [(0, 1)], "second call must be a cache hit, not a recompute"


def test_cached_evaluator_mixin_parallel_matches_serial():
    idx = np.array([[0, 1], [2, 3], [4, 5]])
    serial = _DummyEvaluator(n_jobs=1)(idx)
    parallel = _DummyEvaluator(n_jobs=2)(idx)
    np.testing.assert_array_equal(serial, parallel)


def test_cached_evaluator_mixin_defaults_to_serial_without_n_jobs_attr():
    ev = _NoNJobsEvaluator()
    ev(np.array([[0, 1]]))
    assert ev.calls == [(0, 1)]
