# ANCHOR headline numbers

Data: 39 measured copolymers, 7 monomer systems, 3 sources. Validation: leave-one-monomer-system-out.

| Model | MAE (K) | RMSE (K) | R² | 90% coverage | Mean width (K) |
| --- | --- | --- | --- | --- | --- |
| Fox | 4.4 | 6.6 | 0.98 | — | — |
| Pure ML | 30.7 | 43.8 | 0.04 | 92% | 131.8 |
| ANCHOR | 5.1 | 7.2 | 0.97 | 97% | 28.3 |

Trials to hit a target Tg within ±5 K (mean over every row as target): Random 2.82, Fox 1.18, ANCHOR 1.18.
First-try hit rate: Random 31%, Fox 85%, ANCHOR 87%.

External stress test (12 excluded Taheri resins, AA omitted, low Mn): Fox MAE 7.2 K, ANCHOR MAE 5.4 K, ANCHOR 90% coverage 8%.
