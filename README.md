# Proof at the Shelf

A measurement design lab for in-store retail media. It's an interactive portfolio demo showing that in-store video campaigns are proven, or not, by how they're designed. It's built around networks like Quad's In-Store Connect, with screens at the shelf edge, on endcaps, at checkout and at store entrances. It's an independent project with no affiliation to Quad, and all data is simulated.

## What it demonstrates

The main flow follows five simple steps: Quad at a glance → display opportunities → campaign designs → measurement plans → build a test plan. Supporting material sits in an appendix. Business language comes first, and statistical detail sits in "Method details" expanders.

Every design is store-level. In-store screens reach everyone nearby, so the lab doesn't use shopper-level (loyalty-delivered) designs. Comparisons come from holdout stores, staggered waves or play-frequency levels.

**1 · Quad at a glance.** What In-Store Connect offers today (network, model, formats, targeting, buying, measurement), with public sources. Also covers who it serves (retailers and CPG brands) and why in-store matters now (eMarketer figures cited by Quad).

**2 · Display opportunities.** A store floor plan and the four screen placements (endcap, checkout, shelf-edge, entrance). Each shows where it sits, what it's best for, its reach and what it can measure, plus when content plays (all day, by daypart, by frequency).

**3 · Campaign designs.** The four building blocks of a campaign, six common designs that start from a brand goal, and the three rollout patterns:

| Building block | Options | What it changes about measurement |
|---|---|---|
| Placement | Endcap · Checkout · Shelf-edge · Entrance | The exposure proxy. Only shelf-edge defines a neighboring product, so only shelf-edge enables a cannibalization check. |
| Targeting | Blanket · Category-adjacent · Cross-sell · Conquesting · Daypart | Extra outcomes (complement, competitor) and data granularity (daypart needs time-of-day sales). |
| Offer mechanic | Awareness · Coupon · Bundle · New-product trial | Direct vs. inferred outcomes. A new product has no baseline, which forces a cross-sectional design. |
| Rollout | On/off · Staggered · Dose variation | Where the comparison comes from. |

**4 · Measurement plans.** One table matches each campaign design to:

- how it's proven;
- what it's compared against;
- the data needed;
- what can be claimed;
- the retailer stage it requires.

The page also covers the measurement toolkit (holdout, staggered waves, frequency test, launch comparison), three places design adds leverage with live simulated illustrations, and how plans grow with the retailer.

**5 · Build a test plan.** Pick a brand goal and get a matched design. The readout covers:

- how the design will be proven, in plain language;
- what it can't support;
- the analyses it unlocks;
- a simulated result in incremental revenue and iROAS, with editable assumptions.

**Ask the lab (bottom-right corner of every page).** A chat panel powered by Claude (`claude-opus-5`, via the Anthropic Python SDK). It answers questions about Quad's offering, display opportunities, campaign designs, measurement plans and simulated results, and knows which page you're on and which design is loaded in the builder. It stays open, with the conversation kept, as you move between pages. It's grounded in a brief generated from the app's own data (`assistant.py`), streams its answers, caches the brief with prompt caching, and has server-side refusal fallbacks enabled.

**Appendix.**

- **Brand Q&A:** six common brand objections, each with an answer, the design to propose and a live proof point.
- **Retailer roadmap:** three stages in detail. Early-stage is like Homeland, and new deployments such as Smart & Final start there. Established is like Save Mart's location- and time-of-day activation. Instrumented is where aisle traffic and basket data unlock shelf-edge net-lift reads.
- **Shopper signals:** transaction-data targeting signals.
- **Design guardrails:** a network-wide launch with no comparison, and a new-product launch with no baseline.
- **Two worked methods with full statistical output:**
  - holdout DiD with a Welch interval;
  - staggered waves plus dose, with fixed effects, a pre-trend check and a fitted dose curve.

All data is simulated at runtime from fixed seeds, and true injected effects are kept for recovery checks. Across the 192 designs rated valid (of 240 combinations), the 95% confidence interval contains the injected effect about 96% of the time. There are no external data dependencies.

## Run locally

Requires Python 3.11 (3.12 also works).

```powershell
py -3.11 -m venv .venv          # or: python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL printed by Streamlit (usually `http://localhost:8501`).

## Anthropic API key (for "Ask the lab")

The assistant reads `ANTHROPIC_API_KEY` from the environment. Never commit the key: `.env` is git-ignored.

- **Locally:** create a `.env` file next to `app.py` containing `ANTHROPIC_API_KEY=<your key>`. The app loads it on startup.
- **On Render:** open the service → **Environment** → add `ANTHROPIC_API_KEY`. The Blueprint declares it with `sync: false`, so Render asks for the value instead of reading it from the repo.

Without a key the rest of the app works normally, and the chat panel explains how to add one.

## Deploy to Render

1. Push this repository to GitHub.
2. In Render, choose **New → Blueprint** and connect the repository.
3. Render reads `render.yaml`, installs the pinned requirements, and starts Streamlit on `0.0.0.0` using Render's dynamic `$PORT`.

You can also create a Render **Web Service** manually with build command `pip install -r requirements.txt && python precompute.py` and start command `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`.

On Render's free plan the service sleeps when idle. Open it a minute before a live demo so the first page load isn't a cold start.

## Project files

- `app.py`: Streamlit interface, navigation and charts for every page.
- `design_lab.py`: design dimensions, the measurement recommendation engine, partner-maturity mapping, and the design-matched simulator and estimators.
- `precompute.py`: runs every simulation once at build time and saves the results to `precomputed.pkl` (git-ignored), so pages load without running regressions. The file is fingerprinted against the simulation code; if it's missing or stale, the app computes on demand.
- `simulator.py`: fixed-seed generators and analyses for the two appendix methods.
- `assistant.py`: the "Ask the lab" assistant: knowledge brief built from the app's data, and streaming calls to Claude.
- `targeting_signals.py`: a synthetic 40,000-basket transaction log and the targeting signals mined from it (CDI/BDI, basket affinity, hour × category index, brand switching).
- `illustrations.py`: inline SVG illustrations for the opening page and the appendix: measurement flow, store floor plan with screen placements, and rollout patterns. There are no image files to host.
- `requirements.txt`: pinned runtime dependencies.
- `render.yaml`: Render Blueprint configuration.

Statistical results are illustrative and intended for demonstration, not as business guidance. Partner comparisons are illustrative positioning; no partner data is used.
