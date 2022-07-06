import pytest

from random_mdp.random.simplex import uniform as sample


@pytest.mark.parametrize("n", [1, 2, 10])
def test_is_probability(n):
    p = sample(n)
    assert p.sum() == pytest.approx(1)
