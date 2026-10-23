# ANCHOR: physics-anchored copolymer Tg model (Asian Paints Alchemy 2026, Case 4)

## Data rules
- IMPORTANT: never create, edit or "fix" values in data/. Measurements come only from Mahi's workbook `data/ANCHOR_Final_51_With_Training_Status.xlsx`. If a row looks wrong, report its record_id and stop.
- Never type a number into README, slides or docs by hand. Every reported number comes from `outputs/metrics.json`, written by `python -m anchor.run`.

## Method
- Kelvin internally, °C only in figure labels.
- Prior: Fox with homopolymer Tg from `data/homopolymer_tg.csv`. ANCHOR = Fox + GP on the residual. Pure ML = same GP, no Fox.
- Validation: leave-one-monomer-system-out on the canonical `system` column. Scalers, GP hyperparameters and conformal scores are fitted inside training folds only.
- Intervals: normalised CV+ (K=5), 90%. random_state=42.

## Commands
- Setup: `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`
- Tests: `PYTHONPATH=src .venv/bin/pytest -q`
- Full run: `PYTHONPATH=src .venv/bin/python -m anchor.run`
- Source check: `.venv/bin/python scripts/verify_sources.py`
