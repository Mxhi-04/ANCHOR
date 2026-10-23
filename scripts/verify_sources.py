"""Re-check the workbook against its open sources.

1. Penzel rows (S04_PENZEL) and the five homopolymer Tg values against the
   Zenodo CSV of Bhati et al. 2026 (record 19242815), downloaded fresh.
2. Wang et al. 2024 rows (S02): Tg and the mol% -> wt% conversion, against
   Table 1 of the paper (doi:10.1002/marc.202400198), transcribed below.
3. Taheri et al. 2019 rows (S01): same check against Table 4
   (doi:10.1080/15685551.2019.1699349).

S03 (Polym. Chem. 2011) is paywalled and was not re-checked here.
Run:  python scripts/verify_sources.py
"""
from __future__ import annotations

import csv
import io
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchor.data import load_homopolymers, load_workbook  # noqa: E402

ZENODO = "https://zenodo.org/records/19242815/files/experimental_dataset.csv?download=1"
MOLAR_MASS = {"MMA": 100.12, "BA": 128.17, "ST": 104.15, "EHA": 184.28, "nHA": 156.22}
ZENODO_NAME = {"S": "ST", "Poly(n-hexyl acrylate)": "nHA"}

# Wang et al. 2024, Table 1: styrene mol% / acrylate mol% (1H NMR), Tg (half dCp, 2nd heat, 10 C/min)
WANG = {
    "EHA": [(45, -24), (51, -12), (56, -10), (69, 11), (70, 14), (73, 19)],
    "nHA": [(45, -18), (51, -9), (52, -6), (56, 0), (60, 6), (68, 19), (70, 20)],
}
# Taheri et al. 2019, Table 4: F_MMA, F_BA, F_ST (mol%, NMR), Tg (C)
TAHERI = [
    (39.43, 36.06, 24.51, 17.43), (39.57, 35.82, 24.61, 17.83), (39.12, 36.54, 24.33, 17.10),
    (39.63, 35.72, 24.65, 17.77), (39.75, 35.53, 24.72, 18.21), (36.32, 41.09, 22.59, 13.11),
    (39.71, 35.59, 24.70, 17.92), (39.79, 35.46, 24.75, 18.25), (38.05, 38.28, 23.67, 16.00),
    (39.70, 35.60, 24.70, 17.82), (39.79, 35.45, 24.76, 18.14), (35.06, 43.12, 21.82, 10.97),
]


def mol_to_wt(mol: dict[str, float]) -> dict[str, float]:
    mass = {m: x * MOLAR_MASS[m] for m, x in mol.items()}
    total = sum(mass.values())
    return {m: v / total for m, v in mass.items()}


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'OK' if ok else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")
    return ok


def main() -> int:
    all_ok = True
    df = load_workbook("All_51_With_Status")

    raw = urllib.request.urlopen(ZENODO, timeout=60).read().decode("utf-8-sig")
    rows = {r["ID"]: r for r in csv.DictReader(io.StringIO(raw))}

    # 1a. Penzel rows
    pen = df[df["source_id"] == "S04_PENZEL"]
    bad = []
    for _, r in pen.iterrows():
        z = rows[r["record_id"].split("-")[1]]
        names = [ZENODO_NAME.get(z["Poly1"], z["Poly1"]), ZENODO_NAME.get(z["Poly2"], z["Poly2"])]
        fracs = [float(z["comp_0"]) / 100, float(z["comp_1"]) / 100]
        tg_ok = abs((float(z["Tg(K)"]) - 273.15) - r["Tg_exp_C"]) < 0.01
        comp_ok = all(abs(r[f"{m}_wt_frac"] - f) < 1e-3 for m, f in zip(names, fracs))
        if not (tg_ok and comp_ok):
            bad.append(r["record_id"])
    all_ok &= check(f"{len(pen)} Penzel rows match Zenodo CSV", not bad, ", ".join(bad))

    # 1b. Homopolymer Tg values
    homo = load_homopolymers()
    expected = {"MMA": ("27", 378), "BA": ("7", 230), "ST": ("58", 380), "EHA": ("22", 215), "nHA": ("483", 216)}
    hb = [m for m, (rid, tg) in expected.items() if float(rows[rid]["Tg(K)"]) != homo[m] or homo[m] != tg]
    all_ok &= check("5 homopolymer Tg values match Zenodo CSV", not hb, ", ".join(hb))

    # 2. Wang et al. rows
    s02 = df[df["source_id"] == "S02"]
    bad = []
    for acrylate, table in WANG.items():
        sub = s02[s02[f"{acrylate}_wt_frac"] > 0].sort_values("ST_wt_frac").reset_index(drop=True)
        for (st_mol, tg), (_, r) in zip(table, sub.iterrows()):
            wt = mol_to_wt({"ST": st_mol, acrylate: 100 - st_mol})
            if abs(wt["ST"] - r["ST_wt_frac"]) > 1e-3 or abs(tg - r["Tg_exp_C"]) > 0.01:
                bad.append(r["record_id"])
    all_ok &= check(f"{len(s02)} Wang et al. rows match Table 1 (mol% -> wt%)", not bad and len(s02) == 13, ", ".join(bad))

    # 3. Taheri rows
    s01 = df[df["source_id"] == "S01"].reset_index(drop=True)
    bad = []
    for (f_mma, f_ba, f_st, tg), (_, r) in zip(TAHERI, s01.iterrows()):
        wt = mol_to_wt({"MMA": f_mma, "BA": f_ba, "ST": f_st})
        if any(abs(wt[m] - r[f"{m}_wt_frac"]) > 1e-3 for m in wt) or abs(tg - r["Tg_exp_C"]) > 0.01:
            bad.append(r["record_id"])
    all_ok &= check(f"{len(s01)} Taheri rows match Table 4 (mol% -> wt%)", not bad and len(s01) == 12, ", ".join(bad))

    print("S03 (2 rows, Polym. Chem. 2011): paywalled, not re-checked here.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
