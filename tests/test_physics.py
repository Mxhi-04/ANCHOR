import numpy as np
import pytest

from anchor.physics import fit_gordon_taylor_k, fox_tg, gordon_taylor_tg


def test_homopolymer_returns_its_own_tg():
    tg = np.array([378.0, 230.0])
    assert fox_tg([[1.0, 0.0]], tg)[0] == pytest.approx(378.0)
    assert fox_tg([[0.0, 1.0]], tg)[0] == pytest.approx(230.0)


def test_fox_matches_hand_calculation():
    tg = np.array([378.0, 230.0])
    expected = 1.0 / (0.5 / 378.0 + 0.5 / 230.0)
    assert fox_tg([[0.5, 0.5]], tg)[0] == pytest.approx(expected)


def test_gordon_taylor_reduces_to_fox():
    rng = np.random.default_rng(0)
    tg1, tg2 = 380.0, 215.0
    w1 = rng.uniform(0, 1, 20)
    gt = gordon_taylor_tg(w1, tg1, tg2, k=tg1 / tg2)
    fox = fox_tg(np.column_stack([w1, 1 - w1]), np.array([tg1, tg2]))
    np.testing.assert_allclose(gt, fox, rtol=1e-12)


def test_fit_recovers_known_k():
    w1 = np.linspace(0.1, 0.9, 9)
    y = gordon_taylor_tg(w1, 380.0, 215.0, k=0.8)
    assert fit_gordon_taylor_k(w1, y, 380.0, 215.0) == pytest.approx(0.8, rel=1e-3)


def test_bad_weights_raise():
    with pytest.raises(ValueError):
        fox_tg([[0.6, 0.6]], np.array([378.0, 230.0]))
