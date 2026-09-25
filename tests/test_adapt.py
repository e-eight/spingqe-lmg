import pytest

from spingqe_lmg.adapt import AdaptVQE, build_adapt_pool, lmg_repair_words
from spingqe_lmg.enums import PoolKind
from spingqe_lmg.exact import dicke_matrix, ground_state
from spingqe_lmg.hamiltonians import lmg_hamiltonian, lmg_mean_field_angle


def test_adapt_pool_sizes():
    n = 5
    assert len(build_adapt_pool(n)) == 3 * 10 + 5  # 3 words x C(5,2) + single Z
    assert len(build_adapt_pool(n, kind=PoolKind.COLLECTIVE)) == 3


def test_repair_words_phase_dependence():
    assert lmg_repair_words(1.0, 0.5) == ()
    assert set(lmg_repair_words(1.0, 1.5)) == {"YZ", "ZY", "XY", "YX"}


def test_screening_gradient_encodes_first_order_coupling():
    # the unit-level mechanism statement: at the mean-field state, XX/YY/ZZ
    # words have ~zero screening gradient while YZ words have O(1) gradient
    n, h, lam = 4, 1.0, 1.5
    ham = lmg_hamiltonian(n, h=h, lam=lam)
    theta_mf = lmg_mean_field_angle(h, lam)
    std = AdaptVQE(build_adapt_pool(n, include_single_z=False), ham, n, init_angle=theta_mf)
    g_std = std.screening_gradients([], [])
    ext = AdaptVQE(
        build_adapt_pool(n, paulis=("YZ",), include_single_z=False), ham, n, init_angle=theta_mf
    )
    g_ext = ext.screening_gradients([], [])
    assert g_std.max() < 1e-6
    assert g_ext.max() > 1e-2


def test_adapt_stalls_on_standard_pool_at_mean_field():
    # the predicted ADAPT manifestation of the GQE trapping
    n, h, lam = 4, 1.0, 1.5
    ham = lmg_hamiltonian(n, h=h, lam=lam)
    adapt = AdaptVQE(build_adapt_pool(n), ham, n, init_angle=lmg_mean_field_angle(h, lam))
    result = adapt.run(max_iterations=5)
    assert result.stalled
    assert result.labels == []  # stalled immediately: no first-order direction at all


def test_adapt_stalls_even_on_collective_pool_at_real_reference():
    # The odd-Y rule: for real H and a real reference state, dE/dtheta is
    # proportional to Im<psi|P H|psi>, which vanishes for every real Pauli
    # generator — and a Pauli word is real iff it has an even number of Y's.
    # Jz, Jx^2, Jy^2 are all even-Y, so ADAPT's greedy max-|gradient|
    # selection (the rule shared by all standard ADAPT variants; we implement
    # vanilla qubit-ADAPT, Tang et al. 1911.10205) cannot take a single step
    # from |0...0> with the collective pool — the same pool GQE drove to 1e-6
    # by finite-angle sampling. (GQE-vs-ADAPT separation.)
    n, h, lam = 4, 1.0, 1.5
    ham = lmg_hamiltonian(n, h=h, lam=lam)
    adapt = AdaptVQE(build_adapt_pool(n, kind="collective"), ham, n)
    result = adapt.run(max_iterations=8)
    assert result.stalled
    assert result.labels == []
    assert result.gradients[-1] < 1e-10


def test_odd_y_words_are_the_live_directions_from_zero_reference():
    # complementary positive control for the odd-Y rule at the zero reference
    n, h, lam = 4, 1.0, 1.5
    ham = lmg_hamiltonian(n, h=h, lam=lam)
    odd_y = AdaptVQE(build_adapt_pool(n, paulis=("YZ", "XY"), include_single_z=False), ham, n)
    assert odd_y.screening_gradients([], []).max() > 1e-2


def test_stall_repair_recovers_correlation_energy():
    # the Tier-1.5 claim: classical fluctuation-derived repair unsticks ADAPT
    n, h, lam = 4, 1.0, 1.5
    ham = lmg_hamiltonian(n, h=h, lam=lam)
    e0 = ground_state(dicke_matrix(n, h, lam))[0]
    theta_mf = lmg_mean_field_angle(h, lam)
    from spingqe_lmg.evaluators.pennylane import initial_energy

    e_floor = initial_energy(ham, n, theta_mf)
    adapt = AdaptVQE(build_adapt_pool(n), ham, n, init_angle=theta_mf)
    result = adapt.run(
        max_iterations=10, repair=True, repair_words_fn=lambda: lmg_repair_words(h, lam)
    )
    assert result.repaired_at == 0  # stall detected and repaired at the first screening
    assert not result.stalled
    # recovers a strict majority of the missing correlation energy
    assert result.final_energy < e_floor - 0.5 * (e_floor - e0)


@pytest.mark.parametrize("lam", [0.5, 1.5])
def test_adapt_standard_pool_stalls_at_zero_reference_in_both_phases(lam):
    # the odd-Y rule applies in both phases: the standard pool is entirely
    # even-Y, so from the real |0...0> ADAPT stalls at the reference energy
    # regardless of lambda (in the paramagnetic phase that's near-optimal by
    # luck; in the broken phase it's the trap)
    n, h = 4, 1.0
    ham = lmg_hamiltonian(n, h=h, lam=lam)
    e0 = ground_state(dicke_matrix(n, h, lam))[0]
    adapt = AdaptVQE(build_adapt_pool(n), ham, n)
    result = adapt.run(max_iterations=6)
    assert result.stalled
    assert result.initial_energy == pytest.approx(-n * h / 2, abs=1e-9)
    assert result.initial_energy >= e0 - 1e-9
    assert result.final_energy is None  # stalled: zero iterations


def test_adapt_abs_convergence_on_undershoot():
    n, h, lam = 4, 1.0, 1.5
    ham = lmg_hamiltonian(n, h=h, lam=lam)
    exact = ground_state(dicke_matrix(n, h, lam))[0]
    adapt = AdaptVQE(
        build_adapt_pool(n, paulis=("YZ", "XX", "ZZ"), include_single_z=True),
        ham,
        n,
        init_angle=lmg_mean_field_angle(h, lam),
    )
    result = adapt.run(max_iterations=10, target_error=1e-6, exact_energy=exact)
    assert len(result.energies) > 1
    assert result.final_energy < result.energies[0]


def test_repair_words_fn_receives_h_lam():
    """lmg_repair_words(h, lam) takes 2 args; AdaptVQE.run() must pass them."""
    n, h, lam = 4, 1.0, 1.5
    ham = lmg_hamiltonian(n, h=h, lam=lam)
    pool = build_adapt_pool(n, include_single_z=False)
    adapt = AdaptVQE(pool, ham, n, h=h, lam=lam)
    result = adapt.run(max_iterations=1, repair=True, repair_words_fn=lmg_repair_words)
    assert result.repaired_at is not None, "lmg_repair_words(h,lam) was never called"
    assert result.final_energy is not None, "final_energy must be populated when an iteration runs"
    assert result.energies, "energies must be populated when an iteration runs"
    assert isinstance(result.initial_energy, float), "initial_energy must always be populated"


def test_adapt_zero_iterations_returns_none_final_energy():
    """When max_iterations=0, final_energy must be None, initial_energy populated."""
    n, h, lam = 4, 1.0, 0.5
    ham = lmg_hamiltonian(n, h=h, lam=lam)
    adapt = AdaptVQE(build_adapt_pool(n), ham, n)
    result = adapt.run(max_iterations=0)
    assert result.final_energy is None
    assert result.energies == []
    assert isinstance(result.initial_energy, float)
    assert result.initial_energy == pytest.approx(-n * h / 2, abs=1e-9)
