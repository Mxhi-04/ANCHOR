"""Leave-one-monomer-system-out validation.

Each fold holds out every row of one monomer system. Everything learned
(scaler, GP hyperparameters, conformal scores) is fitted on the other systems
only, so the test is genuine extrapolation to a chemistry the model has not
seen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

from .data import MONOMERS, weights
from .models import Anchor, PureML, cv_plus
from .physics import fit_gordon_taylor_k, fox_tg, gordon_taylor_tg

CONFIDENCE = 0.9
CAL_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def logo_predictions(df: pd.DataFrame, tg_homo: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """Per-row held-out predictions for Fox, Pure ML and ANCHOR."""
    w, y, groups = weights(df), df["Tg_exp_K"].to_numpy(), df["system"].to_numpy()
    out = df[["record_id", "system", "source_id", "Tg_exp_K"]].copy()
    out["Fox_K"] = fox_tg(w, tg_homo)
    calibration = {name: {lvl: np.zeros(len(df), bool) for lvl in CAL_LEVELS} for name in ("Pure ML", "ANCHOR")}

    for train, test in LeaveOneGroupOut().split(w, y, groups):
        assert not set(groups[train]) & set(groups[test]), "held-out system leaked into training"
        for cls in (PureML, Anchor):
            pred = cv_plus(cls, tg_homo, w[train], y[train], w[test])
            lo, hi = pred.interval(CONFIDENCE)
            out.loc[test, f"{cls.name}_K"] = pred.mean
            out.loc[test, f"{cls.name}_lo_K"] = lo
            out.loc[test, f"{cls.name}_hi_K"] = hi
            for lvl in CAL_LEVELS:
                l, h = pred.interval(lvl)
                calibration[cls.name][lvl][test] = (y[test] >= l) & (y[test] <= h)

    curves = {name: {lvl: float(hits.mean()) for lvl, hits in levels.items()} for name, levels in calibration.items()}
    return out, curves


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - y
    ss_res, ss_tot = np.sum(err**2), np.sum((y - y.mean()) ** 2)
    return {
        "MAE_K": float(np.mean(np.abs(err))),
        "RMSE_K": float(np.sqrt(np.mean(err**2))),
        "R2": float(1 - ss_res / ss_tot),
    }


def summarise(pred: pd.DataFrame) -> dict:
    y = pred["Tg_exp_K"].to_numpy()
    summary = {"Fox": _metrics(y, pred["Fox_K"].to_numpy())}
    for name in ("Pure ML", "ANCHOR"):
        m = _metrics(y, pred[f"{name}_K"].to_numpy())
        lo, hi = pred[f"{name}_lo_K"].to_numpy(), pred[f"{name}_hi_K"].to_numpy()
        m["coverage_90"] = float(np.mean((y >= lo) & (y <= hi)))
        m["mean_width_K"] = float(np.mean(hi - lo))
        summary[name] = m
    return summary


def per_system(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system, g in pred.groupby("system"):
        y = g["Tg_exp_K"].to_numpy()
        rows.append({
            "system": system,
            "n": len(g),
            "Fox_MAE_K": float(np.mean(np.abs(g["Fox_K"] - y))),
            "Fox_bias_K": float(np.mean(y - g["Fox_K"])),
            "PureML_MAE_K": float(np.mean(np.abs(g["Pure ML_K"] - y))),
            "ANCHOR_MAE_K": float(np.mean(np.abs(g["ANCHOR_K"] - y))),
            "ANCHOR_coverage_90": float(np.mean((y >= g["ANCHOR_lo_K"]) & (y <= g["ANCHOR_hi_K"]))),
        })
    return pd.DataFrame(rows).sort_values("Fox_MAE_K", ascending=False).reset_index(drop=True)


def gordon_taylor_by_system(df: pd.DataFrame, tg_homo_map: dict) -> pd.DataFrame:
    """Descriptive physics check: Gordon-Taylor k fitted within each binary
    system (in-sample, so not a held-out baseline). k far from Tg1/Tg2 means
    the system departs from Fox."""
    rows = []
    for system, g in df.groupby("system"):
        a, b = system.split("/")
        wa = g[f"{a}_wt_frac"].to_numpy()
        tga, tgb = tg_homo_map[a], tg_homo_map[b]
        k = fit_gordon_taylor_k(wa, g["Tg_exp_K"].to_numpy(), tga, tgb)
        fit = gordon_taylor_tg(wa, tga, tgb, k)
        rows.append({
            "system": system,
            "k_fitted": k,
            "k_fox_equivalent": tga / tgb,
            "GT_in_sample_MAE_K": float(np.mean(np.abs(fit - g["Tg_exp_K"].to_numpy()))),
        })
    return pd.DataFrame(rows)


def external_test(train: pd.DataFrame, external: pd.DataFrame, tg_homo: np.ndarray) -> dict:
    """Train on all 39 rows, predict the excluded Taheri resins."""
    w_tr, y_tr, w_ex, y_ex = weights(train), train["Tg_exp_K"].to_numpy(), weights(external), external["Tg_exp_K"].to_numpy()
    pred = cv_plus(Anchor, tg_homo, w_tr, y_tr, w_ex)
    lo, hi = pred.interval(CONFIDENCE)
    fox = fox_tg(w_ex, tg_homo)
    return {
        "n": int(len(external)),
        "Fox_MAE_K": float(np.mean(np.abs(fox - y_ex))),
        "ANCHOR_MAE_K": float(np.mean(np.abs(pred.mean - y_ex))),
        "ANCHOR_coverage_90": float(np.mean((y_ex >= lo) & (y_ex <= hi))),
        "ANCHOR_mean_width_K": float(np.mean(hi - lo)),
    }


def feature_relevance(df: pd.DataFrame, tg_homo: np.ndarray) -> dict:
    """ARD length scales of the ANCHOR residual GP fitted on all rows
    (shorter = the residual depends more on that feature)."""
    model = Anchor(tg_homo).fit(weights(df), df["Tg_exp_K"].to_numpy())
    names = [f"w_{m}" for m in MONOMERS] + ["Fox_Tg"]
    return dict(zip(names, map(float, model.length_scales())))
