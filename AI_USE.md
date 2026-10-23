# AI use log

The Alchemy rules ask teams to label AI-generated content. This is what AI did in ANCHOR, and what the team checked.

| Date | Who | Tool | What the AI did | What the team checked |
| --- | --- | --- | --- | --- |
| Oct 2026 | Mahi | Google Antigravity (Gemini) | Literature search and table extraction for copolymer Tg data | Every row against its source table; mol% to wt% conversions; excluded the 12 Taheri resins because acrylic acid was not measured |
| 23 Oct 2026 | Jatin | Claude Code (Claude Opus 5.5) | Wrote the code in `src/anchor/`, the tests, `scripts/verify_sources.py`, the figures and GIFs, and this README | Ran the tests and the source check; reviewed the method (Fox prior, GP residual, leave-one-system-out, CV+ intervals) and every number against `outputs/metrics.json` |

The ANCHOR concept, the choice of method and the conclusions are the team's own. All Tg values are literature measurements; none were generated.
