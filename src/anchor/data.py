"""Load Mahi's curated Tg workbook and the homopolymer table.

The workbook (data/ANCHOR_Final_51_With_Training_Status.xlsx) is the single
source of truth for measurements. Nothing here edits values; it only reads,
validates and adds derived columns (canonical system label, source citation).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
WORKBOOK = DATA / "ANCHOR_Final_51_With_Training_Status.xlsx"
HOMOPOLYMERS = DATA / "homopolymer_tg.csv"

MONOMERS = ["MMA", "BA", "ST", "EHA", "nHA"]
WT_COLS = [f"{m}_wt_frac" for m in MONOMERS]
TARGET_C = "Tg_exp_C"

SOURCES = {
    "S01": "Taheri, Jahanfar & Ogino, Des. Monomers Polym. 22, 213 (2019), doi:10.1080/15685551.2019.1699349, Table 4",
    "S02": "Wang et al., Macromol. Rapid Commun. 45, 2400198 (2024), doi:10.1002/marc.202400198, Table 1",
    "S03": "Polym. Chem. 2, 1769 (2011), doi:10.1039/c1py00066g",
    "S04_PENZEL": "Penzel et al., Polymer 38, 325 (1997), doi:10.1016/S0032-3861(96)00521-6, via Bhati et al. 2026, Zenodo 19242815",
}


def canonical_system(row: pd.Series) -> str:
    """Monomers present (wt > 0), alphabetical, joined by '/'.

    The workbook labels the same pair two ways (e.g. 'ST/EHA' and 'EHA/ST'),
    which would split one system into two groups during validation.
    """
    present = [m for m, c in zip(MONOMERS, WT_COLS) if row[c] > 0]
    return "/".join(sorted(present, key=str.lower))


def load_homopolymers(path: Path = HOMOPOLYMERS) -> dict[str, float]:
    table = pd.read_csv(path)
    missing = set(MONOMERS) - set(table["monomer"])
    if missing:
        raise ValueError(f"homopolymer Tg missing for {sorted(missing)}")
    return dict(zip(table["monomer"], table["tg_K"].astype(float)))


def _validate(df: pd.DataFrame) -> None:
    sums = df[WT_COLS].sum(axis=1)
    bad = df.loc[(sums - 1).abs() > 0.005, "record_id"].tolist()
    if bad:
        raise ValueError(f"weight fractions do not sum to 1 for {bad}")
    if (df[WT_COLS] < 0).any().any():
        raise ValueError("negative weight fraction")
    if df["record_id"].duplicated().any():
        raise ValueError("duplicate record_id")
    out_of_range = df.loc[~df[TARGET_C].between(-100, 250), "record_id"].tolist()
    if out_of_range:
        raise ValueError(f"Tg outside -100..250 C for {out_of_range}")


def load_workbook(sheet: str) -> pd.DataFrame:
    df = pd.read_excel(WORKBOOK, sheet_name=sheet)
    _validate(df)
    df = df.copy()
    df["system"] = df.apply(canonical_system, axis=1)
    df["source"] = df["source_id"].map(SOURCES)
    df["Tg_exp_K"] = df[TARGET_C] + 273.15
    return df.reset_index(drop=True)


def load_training() -> pd.DataFrame:
    """The 39 rows Mahi cleared for the five-monomer model."""
    return load_workbook("Training_39")


def load_excluded() -> pd.DataFrame:
    """The 12 Taheri rows: acrylic acid (~1 wt% in feed) was not measured, and
    the resins are low-molecular-weight. Used only as an external stress test."""
    return load_workbook("Excluded_12")


def weights(df: pd.DataFrame) -> np.ndarray:
    return df[WT_COLS].to_numpy(dtype=float)
