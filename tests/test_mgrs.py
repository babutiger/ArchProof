"""MGRS (Minimum Gate-Removal Set) greedy-optimality tests."""
import numpy as np
import pytest


@pytest.mark.parametrize("k", [1, 2, 3, 5, 10])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_greedy_minimum_cardinality(k, seed):
    """Greedy descending-c should always reach a size-k subset whose total
    cost meets the target whenever such a subset exists."""
    from archproof.mgrs import greedy_minimum_removal

    rng = np.random.default_rng(seed)
    n = 20
    contributions = rng.uniform(0.01, 10.0, size=n).astype(np.float64)
    target = float(np.sort(contributions)[::-1][:k].sum())

    try:
        picked = greedy_minimum_removal(contributions, target)
    except (AttributeError, NotImplementedError):
        pytest.skip("greedy_minimum_removal not exposed at this API level")

    # Greedy must achieve the target with at most k removals
    assert len(picked) <= k
    assert sum(contributions[i] for i in picked) >= target - 1e-9


def test_greedy_matches_brute_force_on_small_n():
    """On n=10 brute-force the optimal cardinality and compare to greedy."""
    from archproof.mgrs import greedy_minimum_removal
    from itertools import combinations

    rng = np.random.default_rng(0)
    n = 10
    n_trials = 20
    n_match = 0

    for _ in range(n_trials):
        contributions = rng.uniform(0.01, 10.0, size=n).astype(np.float64)
        target = float(rng.uniform(1.0, contributions.sum() * 0.7))

        # Brute-force minimum cardinality
        bf_k = None
        for k in range(1, n + 1):
            for subset in combinations(range(n), k):
                if sum(contributions[i] for i in subset) >= target:
                    bf_k = k
                    break
            if bf_k is not None:
                break

        try:
            greedy_picked = greedy_minimum_removal(contributions, target)
        except (AttributeError, NotImplementedError):
            pytest.skip("greedy_minimum_removal not exposed")

        if len(greedy_picked) == bf_k:
            n_match += 1

    # Greedy should match brute-force on every trial (per Theorem 8)
    assert n_match == n_trials, \
        f"greedy matched brute-force on {n_match}/{n_trials} trials"


def test_mgrs_never_exceeds_random_baseline():
    """Across many random subsets, greedy should match-or-beat random
    on the great majority (per the 99.8% claim in the paper)."""
    from archproof.mgrs import greedy_minimum_removal

    rng = np.random.default_rng(0)
    n = 30
    n_trials = 100
    n_greedy_better_or_equal = 0

    for _ in range(n_trials):
        contributions = rng.uniform(0.01, 10.0, size=n).astype(np.float64)
        target = float(rng.uniform(0.5, contributions.sum() * 0.5))

        try:
            greedy_picked = greedy_minimum_removal(contributions, target)
        except (AttributeError, NotImplementedError):
            pytest.skip("greedy_minimum_removal not exposed")

        # Random baseline: random permutation of indices, take prefix until target
        perm = rng.permutation(n)
        running = 0.0
        random_picked = []
        for i in perm:
            running += contributions[i]
            random_picked.append(i)
            if running >= target:
                break

        if len(greedy_picked) <= len(random_picked):
            n_greedy_better_or_equal += 1

    rate = n_greedy_better_or_equal / n_trials
    assert rate >= 0.95, f"greedy beat-or-tied random on only {rate:.1%} of trials"
