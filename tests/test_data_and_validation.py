import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from anchor.data import MONOMERS, load_excluded, load_homopolymers, load_training, weights
from anchor.models import Anchor, ConformalPrediction, PureML, cv_plus
from anchor.physics import fox_tg


def test_training_set_shape_and_systems():
    df = load_training()
    assert len(df) == 39
    # 'ST/EHA' and 'EHA/ST' in the workbook are one system once canonicalised.
    assert sorted(df["system"].unique()) == ["BA/EHA", "BA/MMA", "BA/ST", "EHA/MMA", "EHA/ST", "MMA/ST", "nHA/ST"]
    assert np.allclose(weights(df).sum(axis=1), 1, atol=5e-3)


def test_excluded_rows_are_the_taheri_resins():
    ex = load_excluded()
    assert len(ex) == 12 and set(ex["source_id"]) == {"S01"}
    assert not set(ex["record_id"]) & set(load_training()["record_id"])


def test_homopolymers_cover_every_monomer():
    tg = load_homopolymers()
    assert set(tg) == set(MONOMERS)


def test_leave_one_system_out_never_leaks():
    df = load_training()
    g = df["system"].to_numpy()
    for train, test in LeaveOneGroupOut().split(df, groups=g):
        assert not set(g[train]) & set(g[test])
        assert len(set(g[test])) == 1


def test_anchor_learns_zero_residual_when_data_is_pure_fox():
    tg = np.array([378.0, 230.0, 380.0, 215.0, 216.0])
    rng = np.random.default_rng(1)
    w = rng.dirichlet(np.ones(5), size=40)
    y = fox_tg(w, tg)
    m = Anchor(tg).fit(w, y)
    w_test = rng.dirichlet(np.ones(5), size=20)
    assert np.max(np.abs(m.predict(w_test) - fox_tg(w_test, tg))) < 1.0


def test_cv_plus_interval_covers_on_noisy_synthetic_data():
    tg = np.array([378.0, 230.0, 380.0, 215.0, 216.0])
    rng = np.random.default_rng(2)
    w = rng.dirichlet(np.ones(5), size=80)
    y = fox_tg(w, tg) + rng.normal(0, 4, len(w))
    w_te = rng.dirichlet(np.ones(5), size=200)
    y_te = fox_tg(w_te, tg) + rng.normal(0, 4, len(w_te))
    for cls in (Anchor, PureML):
        pred = cv_plus(cls, tg, w, y, w_te)
        lo, hi = pred.interval(0.9)
        assert 0.8 <= np.mean((y_te >= lo) & (y_te <= hi)) <= 1.0


def test_interval_quantile_indexing():
    # 9 training scores of 1 around a mean of 0 and std 1: the 90% CV+ interval is [-1, 1].
    pred = ConformalPrediction(np.zeros(1), np.ones(1), np.zeros((1, 9)), np.ones((1, 9)), np.ones(9))
    lo, hi = pred.interval(0.9)
    assert lo[0] == -1 and hi[0] == 1
