"""'Ask the lab' assistant: answers questions about this app with Claude.

The API key is read from the ANTHROPIC_API_KEY environment variable (Render
dashboard in production, a local .env file in development). It is never stored
in the code.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from functools import lru_cache

import anthropic

from design_lab import (
    DIMENSION_TITLES, DIMENSIONS, GOALS, MATURITY, PRESETS, TARGET_DAYPART, WEAK_DESIGNS,
    Design, maturity_required, recommend, simulate,
)

MODEL = "claude-opus-5"
# Server-side refusal fallback: a declined request is re-run on Anthropic's recommended fallback model.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SUGGESTED_QUESTIONS = [
    "Which design would you propose to a regional grocer running its first campaign?",
    "Why isn't a before/after comparison enough to prove lift?",
    "How should we answer a brand that says redemptions are its ROI?",
    "What does a retailer need to reach the instrumented-network stage?",
]

_INSTRUCTIONS = """You are the assistant inside "Proof at the Shelf", a measurement design lab for in-store retail media built around networks like Quad's In-Store Connect. The people asking are sales leaders and retail media teams, so answer in plain business language first and use statistical terms only when asked or when they add something, explaining them in a clause.

Answer from the lab content below. When a question goes beyond it, say what the lab does cover and give general measurement guidance, labelled as general. Never invent facts about Quad, its partners or real campaign results: every number in the lab comes from a simulation with a known injected effect, so present numbers as "in the lab's simulation". All designs are store-level: an in-store screen reaches everyone nearby, so the lab does not use shopper-level or loyalty-delivered designs, and you should not recommend them.

Keep answers short: a direct answer in the first sentence, then two to five bullets or short paragraphs. When useful, point to the page where the user can see it (for example "open Design the test and choose the goal 'Drive offer redemptions'"). Latency-sensitive: begin your visible answer immediately."""

_PAGES = """Pages in the lab (sidebar):
1. Why proof matters: what In-Store Connect does today (from public sources), three places design adds leverage, the design-to-claim flow and the four design dimensions.
2. Design the test: start from one of six brand goals, which pre-fills a matched design; shows how it will be proven, what it can't support, analyses it unlocks, and a simulated readout with incremental revenue and iROAS (editable assumptions: stores, weeks, item price, media spend). The four dimensions can be fine-tuned under "Advanced".
3. Answer the brand's questions: six common brand objections with an answer, a design to propose and a simulated proof point.
4. Scale with the retailer: a three-stage measurement roadmap.
The assistant (you) opens from the 'Ask the lab' button in the bottom-right corner of every page.
Appendix: Store map & signals (placements, transaction-data targeting signals, offers, rollout patterns), Design guardrails, Holdout method (DiD worked example), Waves & dose method (fixed effects, pre-trend check, dose curve)."""

_PUBLIC_FACTS = """What In-Store Connect does today (from Quad's public materials and partner announcements):
- Network: launched in 2024 with The Save Mart Companies (15 stores, plans for 179 more) and Homeland (15 stores); 25 Smart & Final stores in California launch in fall 2026.
- Model: end to end - content, technology, production and in-store execution, with creative from Quad's Betty agency.
- Formats: digital kiosks, endcaps, in-aisle and shelf screens, vertical banners.
- Targeting: location and time of day, ZIP code or retailer footprint, SKU, aisle or store type, creative swaps by audience, data from 117M households.
- Buying: programmatic through Vistar Media (2025).
- Measurement: POS lift at SKU or brand level, coupon redemption rates, matched-market or QR attribution; DiGiorno case: a 23-point sales lift in four weeks.
- Market: eMarketer projects in-store retail media spend growing from $370M (2024) to $1.06B (2028), as cited by Quad."""

_GLOSSARY = """How the lab reads results (plain language first):
- Holdout / difference-in-differences (DiD): compare how sales changed in campaign stores vs. holdout stores, before vs. after launch. Seasonality hits both groups and cancels out.
- Staggered rollout: stores launch in waves; stores that haven't launched yet are the comparison. No store has to miss out on the campaign.
- Dose variation: low, medium and high play frequency trace out how lift grows with frequency (diminishing returns).
- New-product trial: no sales history exists, so the lab compares stores in the same weeks, adjusted for store size (cross-sectional / ANCOVA) instead of before/after.
- Daypart designs are read by daypart, because weekly totals hide when lift happened.
- Shelf-edge placement defines a neighboring product, which allows lift to be reported net of cannibalization; cross-sell adds the complementary product; conquesting adds the competitor product.
- Coupons leave redemption records, but redemptions overstate incrementality because some redeemers would have bought anyway.
- Revenue view: revenue = measured lift per store-week x item price x stores x weeks; iROAS = incremental revenue / media spend.
- Every simulated scenario includes a 6% market-wide seasonal lift starting in the launch week; good designs difference it out, a before/after read credits it to the campaign.
- Statistical details (fixed effects, clustered standard errors, confidence intervals, p-values) sit in "Method details" expanders."""


_simulate = simulate


def use_simulator(simulator) -> None:
    """Let the app share its cached simulation results, so the brief builds without re-running them."""
    global _simulate
    _simulate = simulator


def api_key_configured() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _design_block(label: str, design: Design) -> str:
    rec = recommend(design)
    primary = _simulate(design)["primary"]
    unlocked = ", ".join(name for name, enabled, _ in rec.analyses if enabled) or "core lift read only"
    return (f"- {label}: {design.describe()} Verdict: {rec.status} ({rec.headline}). How it's proven: {rec.plain} "
            f"Method: {rec.method}. Unlocks: {unlocked}. Can't support: {'; '.join(rec.cannot_support[:3])}. "
            f"Retailer stage needed: {MATURITY[maturity_required(design)]['name']}. "
            f"Simulated result: {primary['estimate']:.1f} vs. true {primary['truth']:.1f} (units per store-week), "
            f"{primary['estimate'] / primary['truth']:.0%} of the true lift.")


@lru_cache(maxsize=1)
def knowledge_base() -> str:
    """Stable, deterministic brief of the app (fixed seeds), so it caches across requests."""
    dimensions = "\n".join(f"- {DIMENSION_TITLES[dim]}: {', '.join(options.values())}" for dim, options in DIMENSIONS.items())
    goals = "\n".join(_design_block(f"Goal '{goal}' (we'd propose: {proposal} The brand can say: {outcome})", design)
                      for goal, proposal, outcome, design in GOALS)
    presets = "\n".join(_design_block(f"Preset '{name}'", design) for name, design in PRESETS.items())
    guardrails = []
    for case in WEAK_DESIGNS:
        weak, fix = _simulate(case["design"])["primary"], _simulate(case["fix"])["primary"]
        guardrails.append(f"- {case['guardrail']} ({case['name']}): {case['story']} Fix: {case['fix_note']} "
                          f"Simulated: {weak['estimate'] / weak['truth']:.0%} of the true lift as designed, {fix['estimate'] / fix['truth']:.0%} with the change.")
    stages = "\n".join(f"- Stage {i + 1} · {m['name']} (like {m['comparable']}): {m['summary']} Measurement ceiling: {m['measurement']}"
                       for i, m in enumerate(MATURITY))

    holdout = _simulate(PRESETS["Regional pilot · endcap coupon holdout"])
    blanket = _simulate(PRESETS["Guardrail · Blanket blast"])["primary"]
    waves = _simulate(Design("entrance", "blanket", "awareness", "staggered"))["primary"]
    basket = {item["key"]: item for item in _simulate(PRESETS["Basket builder · shelf-edge cross-sell bundle"])["decomposition"]}
    daypart = {row["daypart"]: row for row in _simulate(PRESETS["Daypart activation · checkout dose test"])["daypart"]["breakdown"]}
    hp, red = holdout["primary"], holdout["redemption"]
    objections = f"""Brand objections and the lab's answers (page "Answer the brand's questions"):
- "How do we know shoppers wouldn't have bought anyway?" Compare against holdout stores in the same weeks. Simulated: a holdout read lands at {hp['estimate'] / hp['truth']:.0%} of the true lift; a before/after read of a network-wide launch reports {blanket['estimate'] / blanket['truth']:.0%}.
- "Redemptions look great. Isn't that our ROI?" Redemptions prove delivery; pair them with a holdout for incrementality. Simulated: redemption-attributed sales run {red['attributed'] / red['truth']:.1f}x the incremental lift.
- "Holding back stores costs us revenue." Roll out in waves; not-yet-launched stores are the comparison for a few weeks. Simulated: a three-wave rollout recovers {waves['estimate'] / waves['truth']:.0%} of the true lift.
- "We're a regional grocer. Can we even measure this?" Yes: 20 campaign + 20 comparison stores with store-week sales. Simulated: {hp['estimate']:.0f} units per store-week, give or take {(hp['ci_high'] - hp['ci_low']) / 2 / hp['estimate']:.0%}.
- "Is the ad just moving sales from our other products?" Shelf-edge screens let us track the neighboring product and report net lift. Simulated: {basket['sales']['estimate']:.0f} units gross, {-basket['neighbor']['estimate']:.0f} shifted from the neighbor, {basket['sales']['estimate'] + basket['neighbor']['estimate']:.0f} net per store-week.
- "Can we see when the ad actually works?" Read daypart-targeted content by daypart. Simulated: {daypart[TARGET_DAYPART]['estimate']:.0f} units of evening lift per store-week, other dayparts within ±{max(abs(r['estimate']) for n, r in daypart.items() if n != TARGET_DAYPART):.0f}."""

    return "\n\n".join([
        _INSTRUCTIONS, _PAGES, _PUBLIC_FACTS,
        f"The four design dimensions (240 combinations):\n{dimensions}",
        _GLOSSARY,
        f"The six brand goals on 'Design the test':\n{goals}",
        f"Builder presets:\n{presets}",
        f"Design guardrails:\n" + "\n".join(guardrails),
        objections,
        f"Retailer stages ('Scale with the retailer'):\n{stages}",
    ])


def context_note(page: str, design: Design) -> dict:
    """Where the user is right now. Sent as a mid-conversation system message after their question,
    so the cached system prompt and earlier history stay unchanged (append-only)."""
    rec = recommend(design)
    return {"role": "system", "content": f"The user is on the page '{page}'. The design currently loaded in 'Design the test': "
                                         f"{design.describe()} Verdict: {rec.status}. How it's proven: {rec.plain}"}


def stream_reply(messages: list[dict]) -> Iterator[str]:
    """Stream the assistant's reply as text chunks. Errors become a short, readable message."""
    client = anthropic.Anthropic()
    try:
        with client.beta.messages.stream(
            model=MODEL,
            max_tokens=16000,
            output_config={"effort": "medium"},
            betas=[FALLBACK_BETA],
            fallbacks="default",
            system=[{"type": "text", "text": knowledge_base(), "cache_control": {"type": "ephemeral"}}],
            cache_control={"type": "ephemeral"},  # also caches the growing conversation
            messages=messages,
        ) as stream:
            yield from stream.text_stream
            final = stream.get_final_message()
        if final.stop_reason == "refusal":
            yield "\n\n_I can't help with that one. Try a question about campaign design or measurement in the lab._"
        elif final.stop_reason == "max_tokens":
            yield "\n\n_(The answer was cut short. Ask me to continue.)_"
    except anthropic.AuthenticationError:
        yield "The API key was rejected. Check that ANTHROPIC_API_KEY holds a valid, active key."
    except anthropic.PermissionDeniedError:
        yield "This API key doesn't have access to the model. Check the key's workspace and permissions."
    except anthropic.RateLimitError:
        yield "The assistant is busy right now (rate limit). Please try again in a minute."
    except anthropic.APIStatusError as error:
        yield f"The assistant hit an API error ({error.status_code}). Please try again."
    except anthropic.APIConnectionError:
        yield "Couldn't reach the Claude API. Check the network connection and try again."
