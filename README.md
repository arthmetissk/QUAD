# Proof at the Shelf

A measurement design lab for in-store retail media. It's an interactive portfolio demo showing that in-store video campaigns are proven, or not, by how they're designed. It's built around networks like Quad's In-Store Connect, with screens at the shelf edge, on endcaps, at checkout and at store entrances. It's an independent project with no affiliation to Quad, and all data is simulated.

## What it demonstrates

The main flow has four steps, written for a sales audience: why proof matters → design the test → answer the brand's questions → scale with the retailer. Supporting material sits in an appendix. Business language comes first, and statistical detail sits in "Method details" expanders.

Every design is store-level. In-store screens reach everyone nearby, so the lab doesn't use shopper-level (loyalty-delivered) designs. Comparisons come from holdout stores, staggered waves or play-frequency levels.

**1 · Why proof matters.** "Every rollout is a measurement opportunity." It summarizes what In-Store Connect does today, with sources. It shows three places design adds leverage (phased rollouts, redemption reporting, time-of-day activation), each with a live simulated illustration, then the design-to-claim flow and the four design dimensions.

**2 · Design the test.** Starts from six brand goals: launch a new product, win share from a competitor, grow the basket, drive offer redemptions, reach shoppers at the right moment, and find the right play frequency. Each goal pre-fills a matched design. The readout covers:

- how the design will be proven, in plain language;
- what it can't support;
- the analyses it unlocks;
- the result in incremental revenue and iROAS, with editable assumptions (stores, weeks, item price, media spend).

The four dimensions can be fine-tuned under "Advanced":

| Dimension | Options | What it changes about measurement |
|---|---|---|
| Placement | Endcap · Checkout · Shelf-edge · Entrance | The exposure proxy. Only shelf-edge defines a neighboring product, so only shelf-edge enables a cannibalization check. |
| Targeting | Blanket · Category-adjacent · Cross-sell · Conquesting · Daypart | Extra outcomes (complement, competitor) and data granularity (daypart needs time-of-day sales). |
| Offer mechanic | Awareness · Coupon · Bundle · New-product trial | Direct vs. inferred outcomes. A new product has no baseline, which forces a cross-sectional design. |
| Rollout | On/off · Staggered · Dose variation | Where the comparison comes from. |

**3 · Answer the brand's questions.** Six common objections, each with a meeting-ready answer, the design to propose and a live proof point:

- "Wouldn't they have bought anyway?"
- "Isn't redemption our ROI?"
- "Holding back stores costs revenue."
- "Can a regional grocer measure this?"
- "Is it just moving sales from our other products?"
- "Can we see when it works?"

**4 · Scale with the retailer.** A measurement roadmap in three stages:

- **Early-stage:** like Homeland; new deployments such as Smart & Final start here.
- **Established network:** like Save Mart's location- and time-of-day activation.
- **Fully instrumented network:** aisle-level traffic and basket data unlock shelf-edge net-lift reads.

Partner references cite public announcements.

**Appendix.**

- Store map and shopper signals: placements, transaction-data targeting signals, offers and rollout patterns.
- Design guardrails: a network-wide launch with no comparison, and a new-product launch with no baseline.
- Two worked methods with full statistical output:
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

## Deploy to Render

1. Push this repository to GitHub.
2. In Render, choose **New → Blueprint** and connect the repository.
3. Render reads `render.yaml`, installs the pinned requirements, and starts Streamlit on `0.0.0.0` using Render's dynamic `$PORT`.

You can also create a Render **Web Service** manually with build command `pip install -r requirements.txt` and start command `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`.

On Render's free plan the service sleeps when idle. Open it a minute before a live demo so the first page load isn't a cold start.

## Project files

- `app.py`: Streamlit interface, navigation and charts for every page.
- `design_lab.py`: design dimensions, the measurement recommendation engine, partner-maturity mapping, and the design-matched simulator and estimators.
- `simulator.py`: fixed-seed generators and analyses for the two appendix methods.
- `targeting_signals.py`: a synthetic 40,000-basket transaction log and the targeting signals mined from it (CDI/BDI, basket affinity, hour × category index, brand switching).
- `illustrations.py`: inline SVG illustrations for the opening page and the appendix: measurement flow, store floor plan with screen placements, and rollout patterns. There are no image files to host.
- `requirements.txt`: pinned runtime dependencies.
- `render.yaml`: Render Blueprint configuration.

Statistical results are illustrative and intended for demonstration, not as business guidance. Partner comparisons are illustrative positioning; no partner data is used.
