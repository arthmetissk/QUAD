from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

# Local development: read ANTHROPIC_API_KEY from the .env next to this file (never committed).
# On Render there is no .env; the key comes from the service's environment settings.
load_dotenv(Path(__file__).with_name(".env"))

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from design_lab import (
    DIMENSION_TITLES, DIMENSIONS, GOALS, MATURITY, PRESETS, SECONDARY, TARGET_DAYPART, WEAK_DESIGNS,
    Design, maturity_required, recommend, simulate,
)
from assistant import SUGGESTED_QUESTIONS, api_key_configured, context_note, knowledge_base, stream_reply, use_simulator
from design_lab import TARGETING
from illustrations import PLACEMENT_COLORS, measurement_flow, rollout_patterns, store_map, to_img
from simulator import SEED, load_demo_data
from targeting_signals import ADVERTISED_CATEGORY, load_signals

st.set_page_config(
    page_title="Proof at the Shelf | In-Store Retail Media Measurement Lab",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#172b42; --muted:#65788c; --teal:#087e8b; --mint:#c9f0e6; --paper:#f5f8fa; }
    html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
    .stApp { background:var(--paper); color:var(--ink); }
    [data-testid="stSidebar"] { background:#10243a; }
    [data-testid="stSidebar"] * { color:#e7f0f7 !important; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#a9bdce !important; }
    .block-container { max-width:1440px; padding-top:4.5rem; padding-bottom:4rem; }
    header[data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] code { background:rgba(255,255,255,.08); color:#8ce2d0 !important; }
    .stat { background:white; border:1px solid #e5edf2; border-radius:14px; padding:.95rem 1.05rem .85rem; height:100%; margin-bottom:.6rem; }
    .stat-label { color:#65788c; font-size:.82rem; line-height:1.35; }
    .stat-value { font-family:'Manrope',sans-serif; font-size:1.75rem; font-weight:700; color:#172b42; letter-spacing:-.02em; margin:.2rem 0 .15rem; line-height:1.15; }
    .stat-note { font-size:.8rem; line-height:1.35; }
    table.ltable { width:100%; border-collapse:collapse; background:white; border:1px solid #e2eaf0; border-radius:12px; overflow:hidden; font-size:.88rem; margin:.3rem 0 1rem; }
    table.ltable th { text-align:left; background:#f1f6f8; color:#40566b; font-weight:700; padding:.6rem .8rem; border-bottom:1px solid #e2eaf0; }
    table.ltable td { padding:.6rem .8rem; border-bottom:1px solid #edf1f4; color:#40566b; vertical-align:top; line-height:1.45; }
    table.ltable tr:last-child td { border-bottom:none; }
    table.ltable td:first-child { color:#172b42; font-weight:600; }
    h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:var(--ink); letter-spacing:-.035em; }
    h1 { font-size:2.75rem !important; line-height:1.1 !important; }
    h2 { font-size:1.7rem !important; }
    .eyebrow { text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; color:var(--teal); margin-bottom:.55rem; }
    .hero { background:linear-gradient(115deg,#10243a 0%,#173c53 70%,#087e8b 100%); border-radius:24px; padding:2.5rem 2.7rem; color:white; margin:.2rem 0 1.5rem; }
    .hero h1 { color:white !important; max-width:860px; margin:.35rem 0 .8rem; }
    .hero p { color:#c8d9e5; font-size:1.06rem; max-width:820px; margin:0; line-height:1.65; }
    .hero .eyebrow { color:#8ce2d0; }
    .tier-card { background:white; border:1px solid #e4ebf0; border-radius:18px; padding:1.25rem 1.35rem; min-height:245px; box-shadow:0 7px 20px rgba(16,36,58,.035); }
    .tier-number { color:var(--teal); font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; font-weight:700; }
    .tier-card h3 { margin:.55rem 0 .5rem; font-size:1.25rem; }
    .tier-card p { color:var(--muted); font-size:.91rem; line-height:1.55; margin:.4rem 0; }
    .tier-meta { border-top:1px solid #edf1f4; padding-top:.75rem; margin-top:1rem; color:#40566b; font-size:.78rem; font-weight:600; }
    .insight { border-left:4px solid #087e8b; background:#eaf6f4; padding:.9rem 1.1rem; border-radius:0 12px 12px 0; color:#284958; line-height:1.55; margin:.4rem 0 1.2rem; }
    .assumption { border:1px solid #e2eaf0; background:white; padding:1rem 1.15rem; border-radius:12px; color:#506478; line-height:1.6; }
    .stMetric { background:white; border:1px solid #e5edf2; border-radius:14px; padding:1rem 1rem .65rem; }
    div[data-testid="stExpander"] { background:white; border-radius:12px; border:1px solid #e5edf2; }
    .small-note { color:#748699; font-size:.82rem; }
    .footer { border-top:1px solid #e2eaf0; margin-top:3rem; padding-top:1rem; color:#8292a1; font-size:.78rem; }
    .sentence { background:white; border:1px solid #e2eaf0; border-radius:16px; padding:1.1rem 1.35rem; font-family:'Manrope',sans-serif; font-size:1.2rem; font-weight:600; color:var(--ink); line-height:1.5; margin:.6rem 0 1rem; }
    .verdict { border-radius:16px; padding:1.15rem 1.35rem; margin:.4rem 0 1rem; line-height:1.55; }
    .verdict h3 { margin:.25rem 0 .35rem; font-size:1.3rem; }
    .verdict p { margin:.25rem 0; color:#40566b; }
    .verdict.valid { background:#eaf6f4; border:1px solid #bfe3dc; }
    .verdict.caution { background:#fdf5e6; border:1px solid #f1dcae; }
    .verdict.invalid { background:#fcecea; border:1px solid #f1c2bc; }
    .badge { display:inline-block; font-size:.7rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; padding:.22rem .6rem; border-radius:999px; color:white; }
    .badge.valid { background:#087e8b; } .badge.caution { background:#b7791f; } .badge.invalid { background:#c0392b; }
    .panel { background:white; border:1px solid #e2eaf0; border-radius:14px; padding:1rem 1.15rem; height:100%; }
    .panel h4 { font-family:'Manrope',sans-serif; font-size:.78rem; letter-spacing:.12em; text-transform:uppercase; color:var(--teal); margin:0 0 .55rem; }
    .panel ul { margin:0; padding-left:1.1rem; color:#40566b; font-size:.9rem; line-height:1.55; }
    .panel.cannot h4 { color:#c0392b; }
    .aud { background:white; border:1px solid #e2eaf0; border-top:3px solid #087e8b; border-radius:12px; padding:.8rem 1rem; margin:.2rem 0 1.1rem; }
    .aud h4 { font-family:'Manrope',sans-serif; font-size:.74rem; letter-spacing:.12em; text-transform:uppercase; color:#087e8b; margin:0 0 .3rem; }
    .aud p { margin:0; color:#40566b; font-size:.9rem; line-height:1.5; }
    .grid { display:grid; grid-auto-rows:1fr; column-gap:1rem; row-gap:1rem; margin:.3rem 0 1rem; }
    .grid-2 { grid-template-columns:repeat(2, minmax(0, 1fr)); }
    .grid-3 { grid-template-columns:repeat(3, minmax(0, 1fr)); }
    .grid-4 { grid-template-columns:repeat(4, minmax(0, 1fr)); }
    .grid-5 { grid-template-columns:repeat(5, minmax(0, 1fr)); }
    .panel.selected { border:2px solid #087e8b; box-shadow:0 6px 18px rgba(8,126,139,.12); }
    .grid > * { margin:0 !important; min-height:0 !important; height:auto !important; display:flex; flex-direction:column; }
    .grid .bottom { margin-top:auto !important; }
    /* Stage cards share row tracks (subgrid) so each section starts at the same height in every card. */
    .grid.stages { grid-auto-rows:auto; }
    .grid.stages > * { display:grid !important; grid-template-rows:subgrid; grid-row:span 6; row-gap:.55rem; align-content:start; }
    .grid.stages > * > * { margin:0 !important; }
    @media (max-width: 640px) { .grid.grid-2, .grid.grid-3, .grid.grid-4, .grid.grid-5 { grid-template-columns:minmax(0, 1fr); grid-auto-rows:auto; } }
    .chips { display:flex; flex-wrap:wrap; gap:.5rem; margin:.4rem 0 1rem; }
    .chip { background:white; border:1px solid #bfe3dc; color:#0b5f69; border-radius:10px; padding:.45rem .7rem; font-size:.82rem; font-weight:600; }
    .chip span { display:block; font-weight:400; color:#6b8193; font-size:.74rem; }
    .chip.off { background:#f1f4f6; border-color:#e2e8ed; color:#9aa8b4; text-decoration:line-through; }
    .chip.off span { text-decoration:none; color:#a6b2bc; }
    .opt { display:inline-block; margin:.15rem .25rem .15rem 0; padding:.18rem .5rem; border-radius:8px; font-size:.78rem; background:#e8f5f2; color:#0b5f69; }
    .opt.off { background:#f1f4f6; color:#a6b2bc; text-decoration:line-through; }
    @media (max-width: 640px) { h1 { font-size:2rem !important; } .hero { padding:1.6rem 1.4rem; } .hero h1 { overflow-wrap:normal; } .stat-value { font-size:1.45rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)

STATUS_LABEL = {"valid": "Valid design", "caution": "Valid with caveats", "invalid": "Needs a comparison group"}


@st.cache_data(show_spinner="Generating reproducible synthetic campaign data…")
def demo_data():
    return load_demo_data(SEED)


@st.cache_data(show_spinner="Simulating data for this design and running the recommended method…")
def run_design(placement: str, targeting: str, offer: str, rollout: str):
    design = Design(placement, targeting, offer, rollout)
    return recommend(design), simulate(design, SEED)


use_simulator(lambda design: run_design(*vars(design).values())[1])  # the assistant reuses the app's cached simulations


def money(value: float) -> str:
    return f"{value:,.1f}"


def p_label(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def fmt(value: float, kind: str) -> str:
    if kind == "dollars":
        return f"${value:,.2f}"
    if kind == "pp":
        return f"{value:,.1f} pp"
    return f"{value:,.1f}"


def chart_style(fig, height: int = 390):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white",
        font=dict(family="DM Sans", color="#52677b", size=12),
        margin=dict(l=18, r=18, t=84, b=20),
        title_yref="container", title_y=.975, title_yanchor="top", title_x=0, title_xanchor="left", title_pad=dict(l=6),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(showgrid=False, linecolor="#d9e2e9", zeroline=False),
        yaxis=dict(gridcolor="#edf1f4", zeroline=False),
    )
    fig.update_traces(cliponaxis=False, selector=dict(type="bar"))
    fig.update_traces(cliponaxis=False, selector=dict(type="waterfall"))
    return fig


def header(kicker: str, title: str, description: str):
    st.markdown(f'<div class="eyebrow">{kicker}</div><h1>{title}</h1><p style="color:#64798d;font-size:1.03rem;max-width:900px;line-height:1.65">{description}</p>', unsafe_allow_html=True)


# Grid column-gap (1rem) matches st.columns(gap="small"), so button rows line up under card grids.


def card_grid(cards: list[str], cols: int | None = None, extra_class: str = ""):
    """Render cards as one CSS grid so every card in a row gets the same height."""
    # Column counts are CSS classes: Streamlit strips custom properties from inline styles.
    st.markdown(f"<div class='grid grid-{cols or len(cards)} {extra_class}'>{''.join(cards)}</div>", unsafe_allow_html=True)


def button_row(buttons: list[tuple]):
    """Buttons aligned under a card grid: each item is (label, key, callback, args)."""
    for col, (label, key, callback, args) in zip(st.columns(len(buttons), gap="small"), buttons):
        with col:
            st.button(label, key=key, on_click=callback, args=args, use_container_width=True)


def stat_html(label: str, value: str, note: str | None = None, tone: str = "muted") -> str:
    color = {"good": "#087e8b", "bad": "#c0392b"}.get(tone, "#6b8193")
    return f"<div class='stat'><div class='stat-label'>{label}</div><div class='stat-value'>{value}</div><div class='stat-note bottom' style='color:{color}'>{note or '&nbsp;'}</div></div>"


def result_card(label: str, value: str, note: str | None = None, tone: str = "muted"):
    st.markdown(stat_html(label, value, note, tone), unsafe_allow_html=True)


def html_table(rows: list[dict]):
    head = "".join(f"<th>{column}</th>" for column in rows[0])
    body = "".join("<tr>" + "".join(f"<td>{value}</td>" for value in row.values()) + "</tr>" for row in rows)
    st.markdown(f"<table class='ltable'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>", unsafe_allow_html=True)


def audience(retailer: str, brand: str):
    card_grid([f"<div class='aud'><h4>{who}</h4><p>{text}</p></div>" for who, text in (("For the retailer", retailer), ("For the CPG brand", brand))])


def bullet_panel(title: str, items: list[str], css: str = "") -> str:
    body = "".join(f"<li>{item}</li>" for item in items) or "<li>Nothing flagged.</li>"
    return f"<div class='panel {css}'><h4>{title}</h4><ul>{body}</ul></div>"


# ---------- Session state & navigation ----------

PAGES = ["Why proof matters", "Design the test", "Answer the brand's questions", "Scale with the retailer", "Ask the lab",
         "Appendix · Store map & signals", "Appendix · Design guardrails", "Appendix · Holdout method", "Appendix · Waves & dose method"]


# The chosen design lives in plain session keys. Builder widgets are keyed by a version number
# that changes whenever a design is loaded from elsewhere, so they re-render with the new values
# instead of keeping (or re-sending) stale widget state.

def load_design(design: Design, preset: str = "Custom"):
    st.session_state.design = {dim: getattr(design, dim) for dim in DIMENSIONS}
    st.session_state.preset_name = preset
    st.session_state.design_version = st.session_state.get("design_version", 0) + 1


def open_in_builder(design: Design):
    load_design(design, next((name for name, preset in PRESETS.items() if preset == design), "Custom"))
    st.session_state.page = "Design the test"


if "design" not in st.session_state:
    first = next(iter(PRESETS))
    load_design(PRESETS[first], first)

with st.sidebar:
    st.markdown("<div style='font-family:Manrope;font-weight:800;font-size:1.08rem;color:white'>◉ PROOF AT THE SHELF</div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:.77rem;margin-top:.15rem'>A measurement design lab for in-store retail media</p>", unsafe_allow_html=True)
    page = st.radio("Explore the framework", PAGES, key="page", label_visibility="collapsed")
    if page != "Ask the lab":
        st.session_state.last_content_page = page  # gives the assistant context on what the user was looking at
    # Streamlit keeps the scroll position across reruns, so a newly selected page would open
    # part-way down. Scroll the main pane back to the top whenever the page changes.
    if st.session_state.get("last_page") != page:
        st.session_state.last_page = page
        st.session_state.page_views = st.session_state.get("page_views", 0) + 1
        components.html(
            f"""<script>
            // page view {st.session_state.page_views}: {page}
            const toTop = () => {{
                const doc = window.parent.document;
                [doc.querySelector('[data-testid="stMain"]'), doc.querySelector('[data-testid="stAppViewContainer"]'), doc.scrollingElement]
                    .forEach(el => el && el.scrollTo({{top: 0, left: 0, behavior: "instant"}}));
            }};
            toTop(); setTimeout(toTop, 150); setTimeout(toTop, 600);
            </script>""",
            height=0,
        )
    st.markdown("---")
    st.markdown(f"**Synthetic data**  \nFixed seed · `{SEED}`")
    st.markdown("<p class='small-note'>All stores, shoppers, sales and effects are simulated. No external data is used.</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<p class='small-note'>Campaign design and measurement are chosen together, not bolted on after the fact.</p>", unsafe_allow_html=True)


# ---------- Shared result rendering ----------

def series_chart(result, title: str, height: int = 360):
    series = result["series"]
    palette = ["#087e8b", "#2a9d8f", "#5fb8a9", "#183a50"]
    fig = go.Figure()
    groups = sorted(series.group.unique(), key=lambda g: (g.startswith("Holdout"), g))
    for index, group in enumerate(groups):
        rows = series.loc[series.group == group]
        holdout = group.startswith("Holdout")
        fig.add_trace(go.Scatter(x=rows.week, y=rows.sales, mode="lines+markers", name=group,
                                 line=dict(color="#aab7c3" if holdout else palette[index % len(palette)], width=2.4, dash="dot" if holdout else "solid")))
    for launch in result["launches"]:
        fig.add_vline(x=launch - .5, line_dash="dash", line_color="#e07a5f")
    fig.update_layout(title=title, xaxis_title="Week", yaxis_title=result["series_y"], xaxis=dict(dtick=1))
    return chart_style(fig, height)


def recovery_chart(rows, kind: str, title: str, height: int = 360):
    labels = [row["label"] for row in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[row["estimate"] for row in rows], name="Estimate (95% CI)", marker_color="#087e8b",
                         error_y=dict(type="data", symmetric=False, array=[row["ci_high"] - row["estimate"] for row in rows], arrayminus=[row["estimate"] - row["ci_low"] for row in rows], color="#183a50")))
    fig.add_trace(go.Scatter(x=labels, y=[row["truth"] for row in rows], mode="markers", name="True lift",
                             marker=dict(symbol="line-ew-open", size=46, line=dict(width=3, color="#e07a5f"))))
    fig.update_layout(title=title, yaxis_title=kind, barmode="group")
    return chart_style(fig, height)


def render_result(rec, result):
    kind, primary = result["fmt"], result["primary"]
    if rec.status == "invalid":
        st.markdown("<div class='insight' style='border-left-color:#c0392b;background:#fcecea;color:#6b2a22'><b>Not yet reportable: add a comparison to make it defensible.</b> The number is shown so you can see how far it lands from the injected truth.</div>", unsafe_allow_html=True)
    inside = primary["ci_low"] <= primary["truth"] <= primary["ci_high"]
    ratio = primary["estimate"] / primary["truth"] if primary["truth"] else float("nan")
    stats = []
    stats.append(stat_html(primary["label"], fmt(primary["estimate"], kind), f"Likely range {fmt(primary['ci_low'], kind)} to {fmt(primary['ci_high'], kind)}"))
    stats.append(stat_html("True lift (known in simulation)", fmt(primary["truth"], kind), result["unit"]))
    stats.append(stat_html("Share of the true lift recovered", f"{ratio:.0%}", "Truth inside the likely range" if inside else "Truth outside the likely range", "good" if inside else "bad"))
    reportable = {"valid": ("Yes", "Design supports the claim", "good"), "caution": ("With caveats", "See the warnings above", "muted"), "invalid": ("After one design change", "Add a comparison group", "muted")}[rec.status]
    stats.append(stat_html("Reportable to the brand?", reportable[0], reportable[1], reportable[2]))
    card_grid(stats)
    revenue_view(rec, result)

    left, right = st.columns([1.35, 1])
    with left:
        st.plotly_chart(series_chart(result, "Simulated outcome by week" + (" · dashed lines mark launches" if result["launches"] else "")), use_container_width=True)
    with right:
        st.plotly_chart(recovery_chart(result["rows"], result["unit"], "Measured vs. true lift"), use_container_width=True)
    with st.expander("Method details: statistical output"):
        html_table([{"Term": row["label"], "Estimate": fmt(row["estimate"], kind), "95% CI": f"[{fmt(row['ci_low'], kind)}, {fmt(row['ci_high'], kind)}]",
                     "p-value": p_label(row["p_value"]), "True lift": fmt(row["truth"], kind)} for row in result["rows"]])
        st.markdown(f"<p class='small-note'>{rec.method}. {'Paired t-test on store-level before/after changes.' if primary['term'] == 'prepost' else 'Standard errors clustered by ' + ('shopper' if result['level'] == 'shopper' else 'store') + '.'} Units: {result['unit']}.</p>", unsafe_allow_html=True)

    if result["dose"]:
        dose = result["dose"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dose["grid"], y=dose["truth_curve"], mode="lines", name="True response", line=dict(color="#e07a5f", dash="dash", width=2)))
        points = dose["points"]
        fig.add_trace(go.Scatter(x=[p["intensity"] for p in points], y=[p["estimate"] for p in points], mode="markers+text", name="Estimated (95% CI)",
                                 text=[p["label"].split()[0] for p in points], textposition="top center", marker=dict(size=12, color="#087e8b"),
                                 error_y=dict(type="data", symmetric=False, array=[p["ci_high"] - p["estimate"] for p in points], arrayminus=[p["estimate"] - p["ci_low"] for p in points])))
        fig.update_layout(title="Dose-response" + (" · relative to low dose (no zero-dose stores)" if dose["relative"] else ""), xaxis_title="Play intensity", yaxis_title=result["unit"])
        st.markdown("#### Dose-response curve")
        st.plotly_chart(chart_style(fig, 340), use_container_width=True)

    if result["daypart"]:
        daypart = result["daypart"]
        st.markdown("#### Why daypart designs need daypart data")
        left, right = st.columns([1.2, 1])
        with left:
            st.plotly_chart(recovery_chart([{**row, "label": row["daypart"]} for row in daypart["breakdown"]], result["unit"].split(",")[0].replace(" in the evening daypart", ""), "Estimated lift by daypart"), use_container_width=True)
        with right:
            weekly, targeted = daypart["weekly"], daypart["targeted"]
            weekly_t = weekly["estimate"] / weekly["se"]
            targeted_t = targeted["estimate"] / targeted["se"]
            coarse = "all-trip" if result["level"] == "shopper" else "weekly-total"
            st.markdown(
                f"<div class='assumption'>The content only runs in the <b>{TARGET_DAYPART.lower()}</b>, so lift should appear there and nowhere else, and the chart confirms it.<br><br>"
                f"A {coarse} model estimates <b>{fmt(weekly['estimate'], kind)}</b> (truth {fmt(weekly['truth'], kind)}, t = {weekly_t:.1f}) but can't say <i>when</i> the lift happened. "
                f"The daypart-grain model estimates <b>{fmt(targeted['estimate'], kind)}</b> in the targeted daypart (truth {fmt(targeted['truth'], kind)}, t = {targeted_t:.1f}). "
                "Outcome data must be at least as granular as the design: a daypart test read on weekly totals can't confirm the targeting did anything.</div>",
                unsafe_allow_html=True,
            )

    if result["exposure"]:
        exposure = result["exposure"]
        st.markdown("#### Exposure adjustment")
        stats = []
        stats.append(stat_html("Intent-to-treat · per assigned shopper-week", fmt(exposure["itt"]["estimate"], kind)))
        stats.append(stat_html("Traffic-weighted · per likely exposure", fmt(exposure["adjusted"]["estimate"], kind), f"95% CI [{fmt(exposure['adjusted']['ci_low'], kind)}, {fmt(exposure['adjusted']['ci_high'], kind)}]"))
        stats.append(stat_html("Injected effect per exposure", fmt(exposure["adjusted"]["truth"], kind)))
        card_grid(stats)
        st.markdown(f"<p class='small-note'>Only {exposure['exposure_rate']:.0%} of assigned shopper-weeks were actually exposed. The ITT answers \"what did assigning the campaign do?\"; aisle traffic rescales it to \"what does one impression do?\".</p>", unsafe_allow_html=True)

    if result["decomposition"]:
        items = result["decomposition"]
        own = [item for item in items if item["key"] != "competitor"]
        competitor = next((item for item in items if item["key"] == "competitor"), None)
        money_kind = "dollars" if result["level"] == "shopper" else "units"
        st.markdown("#### Gross vs. net lift")
        left, right = st.columns([1.25, 1])
        with left:
            net = sum(item["estimate"] for item in own)
            waterfall = go.Figure(go.Waterfall(
                measure=["absolute"] + ["relative"] * (len(own) - 1) + ["total"],
                x=[item["label"] for item in own] + ["Net to the brand"],
                y=[item["estimate"] for item in own] + [net],
                text=[fmt(item["estimate"], money_kind) for item in own] + [fmt(net, money_kind)], textposition="outside",
                connector={"line": {"color": "#becbd4"}},
                decreasing={"marker": {"color": "#e07a5f"}}, increasing={"marker": {"color": "#087e8b"}}, totals={"marker": {"color": "#183a50"}},
            ))
            waterfall.update_layout(title="Estimated decomposition", yaxis_title="$ / assigned shopper-week" if money_kind == "dollars" else "Units / store-week", showlegend=False)
            st.plotly_chart(chart_style(waterfall, 360), use_container_width=True)
        with right:
            truth_net = sum(item["truth"] for item in own)
            lines = [f"<b>{item['label']}:</b> {fmt(item['estimate'], money_kind)} (truth {fmt(item['truth'], money_kind)}, p = {p_label(item['p_value'])})" for item in items]
            story = f"<br><br>Net to the brand: <b>{fmt(net, money_kind)}</b> estimated vs. <b>{fmt(truth_net, money_kind)}</b> injected."
            if competitor:
                story += f" The competitor's decline ({fmt(competitor['estimate'], money_kind)}) is the <i>intended</i> share shift for a conquesting campaign, so it's reported separately rather than netted out."
            st.markdown("<div class='assumption'>" + "<br>".join(lines) + story + "</div>", unsafe_allow_html=True)

    if result["redemption"]:
        redemption = result["redemption"]
        money_kind = "dollars" if result["level"] == "shopper" else "units"
        st.markdown("#### Redemptions are not incrementality")
        left, right = st.columns([1.1, 1])
        with left:
            fig = go.Figure(go.Bar(x=["Redemption-attributed", "Measured incremental", "True incremental"],
                                   y=[redemption["attributed"], redemption["incremental"], redemption["truth"]],
                                   marker_color=["#aab7c3", "#087e8b", "#e07a5f"],
                                   text=[fmt(v, money_kind) for v in (redemption["attributed"], redemption["incremental"], redemption["truth"])], textposition="outside"))
            fig.update_layout(title="Attributed vs. incremental", yaxis_title="$ / assigned shopper-week" if money_kind == "dollars" else "Units / treated store-week", showlegend=False)
            st.plotly_chart(chart_style(fig, 330), use_container_width=True)
        with right:
            overstatement = redemption["attributed"] / redemption["truth"] if redemption["truth"] else float("nan")
            st.markdown(f"<div class='assumption'>Redemptions are a direct, trackable outcome, and that's what makes this mechanic measurable. But some redeemers would have bought anyway, so redemption-attributed sales are <b>{overstatement:.1f}×</b> the true incremental effect. Use redemptions to confirm delivery and targeting; use the controlled comparison for lift.</div>", unsafe_allow_html=True)


def revenue_view(rec, result):
    """Translate the measured lift into incremental revenue and iROAS, the numbers a sales team plans around."""
    kind, primary = result["fmt"], result["primary"]
    shopper = result["level"] == "shopper"
    st.markdown("#### In revenue terms")
    with st.expander("Campaign assumptions (edit to match a real proposal)"):
        a1, a2, a3, a4 = st.columns(4)
        reach = a1.number_input("Loyalty shoppers assigned" if shopper else "Campaign stores", min_value=1, value=1200 if shopper else 20, step=100 if shopper else 5, key=f"rev_reach_{result['level']}")
        weeks = a2.number_input("Campaign weeks", min_value=1, max_value=52, value=8, key="rev_weeks")
        price = a3.number_input("Average item price ($)", min_value=0.5, value=4.99, step=0.5, key="rev_price", disabled=kind == "dollars")
        spend = a4.number_input("Media spend ($)", min_value=500, value=25_000, step=1_000, key="rev_spend")
    per_unit = 1.0 if kind == "dollars" else (price / 100 if kind == "pp" else price)
    scale = per_unit * reach * weeks
    measured, true = primary["estimate"] * scale, primary["truth"] * scale
    cards = [
        stat_html("Measured incremental revenue", f"${measured:,.0f}", f"iROAS {measured / spend:.2f} on ${spend:,.0f} of media"),
        stat_html("True incremental revenue", f"${true:,.0f}", f"iROAS {true / spend:.2f} · known because the data is simulated"),
    ]
    if result["redemption"] and not shopper:
        attributed = result["redemption"]["attributed"] * scale
        cards.append(stat_html("Redemption-attributed revenue", f"${attributed:,.0f}", f"{attributed / true:.1f}× the true incremental revenue", "bad"))
    else:
        gap = measured - true
        cards.append(stat_html("Measured vs. true", f"{'+' if gap >= 0 else '−'}${abs(gap):,.0f}", "Within normal sampling range" if rec.status != "invalid" else "The gap a brand's analysts would find at renewal", "good" if rec.status != "invalid" else "bad"))
    card_grid(cards)
    basis = "loyalty shoppers" if shopper else "stores"
    st.markdown(f"<p class='small-note'>Revenue = measured lift × {'$ per shopper-week' if kind == 'dollars' else 'item price'} × {reach:,} {basis} × {weeks} weeks. iROAS = incremental revenue ÷ media spend.</p>", unsafe_allow_html=True)


def render_recommendation(design: Design, rec):
    st.markdown(
        f"<div class='verdict {rec.status}'><span class='badge {rec.status}'>{STATUS_LABEL[rec.status]}</span>"
        f"<h3>{rec.headline}</h3><p><b>How we'll prove it:</b> {rec.plain}</p></div>",
        unsafe_allow_html=True,
    )
    with st.expander("Method details"):
        st.markdown(f"**Method:** {rec.method}  \n{rec.detail}")
    card_grid([
        bullet_panel("Outcomes to track", rec.outcomes),
        bullet_panel("Data & exposure proxy", [f"<b>Grain:</b> {rec.grain}", f"<b>Exposure proxy:</b> {rec.exposure_proxy}", f"<b>Retailer stage needed:</b> {MATURITY[maturity_required(design)]['name']}"]),
        bullet_panel("This design cannot support", rec.cannot_support, "cannot"),
    ])
    unlocked = [(name, note) for name, enabled, note in rec.analyses if enabled]
    locked = len(rec.analyses) - len(unlocked)
    chips = "".join(f"<div class='chip'>{name}<span>{note}</span></div>" for name, note in unlocked)
    more = f"<p class='small-note'>{locked} more analyses open up with other design choices.</p>" if locked else ""
    if unlocked:
        st.markdown(f"<div style='margin-top:1rem' class='eyebrow'>Analyses this design unlocks</div><div class='chips'>{chips}</div>{more}", unsafe_allow_html=True)
    else:
        st.markdown(f"<p class='small-note' style='margin-top:1rem'>This design supports the core lift read only. {locked} further analyses open up with other design choices.</p>", unsafe_allow_html=True)
    for warning in rec.warnings:
        st.warning(warning, icon="⚠️")


# ---------- Pages ----------

PLACEMENT_NOTES = [
    ("endcap", 1, "Endcap / high-traffic display", "High reach where shoppers turn between aisles. Exposure is known only from store traffic, so readouts stay at store level."),
    ("checkout", 2, "Checkout / point-of-sale screen", "Near-universal exposure among buyers and a strong impulse moment. Transaction counts per lane are the exposure proxy."),
    ("shelf", 3, "Shelf-edge / in-aisle", "Next to one product, so aisle traffic becomes a likely-impression dose and the neighboring product is a natural cannibalization control."),
    ("entrance", 4, "Entrance / lobby screen", "Every shopper walks past it, but far from any shelf. Broad reach, low precision, and small, diluted effects that need large samples."),
]

TARGETING_PLAYBOOK = {
    "blanket": {
        "question": "Is there any reason to vary the content at all?",
        "signals": ["<b>Store traffic by hour</b> (transaction counts) → how many plays each slot can reach",
                    "<b>Store-week category sales</b> → matching holdout stores on pre-period sales",
                    "<b>Seasonality index</b> (weekly sales vs. annual average) → when to flight the campaign"],
        "decision": "Run one message everywhere when the goal is awareness or the partner only has store-week POS. Transaction data still sets <i>when</i> to run and <i>which stores</i> to hold out.",
        "measurement": "Blanket <i>content</i> is fine; a blanket <i>rollout</i> with no holdout or waves isn't measurable.",
        "needs": "Store-week POS",
    },
    "category": {
        "question": "Which stores and aisles should carry which category message?",
        "signals": ["<b>Category development index (CDI)</b>: store category penetration ÷ chain average × 100",
                    "<b>Brand development index (BDI)</b>: store brand penetration ÷ chain average × 100",
                    "<b>Category penetration</b>: share of baskets containing the category",
                    "<b>Planogram / aisle map</b>: which screen sits next to which category"],
        "decision": "Stores where the category over-indexes but the brand under-indexes (high CDI, low BDI) are the <b>grow</b> targets: shoppers already buy the category, just not this brand.",
        "measurement": "Outcome is the advertised brand. If targeting uses CDI/BDI, randomize holdouts <i>within</i> each quadrant so the comparison isn't confounded by store type.",
        "needs": "Item-level store-week POS",
    },
    "crosssell": {
        "question": "Which product's shelf should carry an ad for its complement?",
        "signals": ["<b>Support</b>: share of baskets with both items",
                    "<b>Attach rate (confidence)</b>: P(complement | anchor in basket)",
                    "<b>Lift</b>: attach rate ÷ complement's overall penetration (above 1 means real affinity)",
                    "<b>Promo calendar</b>: avoid pairing with an anchor already on deal"],
        "decision": "Pick anchor → complement pairs with lift ≥ 1.5 and enough support to matter, then put the complement's ad at the anchor's shelf.",
        "measurement": "The complement becomes a <b>second outcome variable</b>, and halo on the anchor is reported alongside the advertised lift.",
        "needs": "Basket-level transaction log",
    },
    "conquest": {
        "question": "Where are competitor buyers contestable, and how much share is up for grabs?",
        "signals": ["<b>Brand switching matrix</b>: repeat buyers' brand last quarter vs. this quarter, from retailer loyalty data where it's shared",
                    "<b>Competitor share by store</b>: where the competitor dominates the shelf",
                    "<b>Price gap index</b>: our shelf price ÷ competitor's, by store",
                    "<b>Competitor promo timing</b>: when their buyers are most price-sensitive"],
        "decision": "Target competitor-heavy stores where switching toward the brand is already non-trivial. Loyal buyers of a brand rarely move; switchers do.",
        "measurement": "Track the <b>competitor's sales</b> as an outcome. Their decline is the intended share shift, reported separately from own-brand cannibalization.",
        "needs": "Item POS by brand; retailer loyalty data for switching, if shared",
    },
    "daypart": {
        "question": "At what hours does this category's buying mission happen?",
        "signals": ["<b>Hour × category sales index</b>: category share in an hour ÷ traffic share in that hour × 100",
                    "<b>Basket mission by daypart</b>: breakfast top-up, lunch, dinner-tonight, stock-up",
                    "<b>Traffic curve</b>: transaction counts per hour, which sets how many shoppers each slot reaches"],
        "decision": "Schedule the creative where the category index is above ~120 <i>and</i> traffic is high, e.g. coffee in the morning and pasta sauce in the evening.",
        "measurement": "The outcome must be <b>timestamped</b>: a daypart test read on weekly totals can't confirm the targeting did anything.",
        "needs": "Timestamped transactions",
    },
}


@st.cache_data(show_spinner="Building the synthetic transaction log…")
def signals_data():
    return load_signals(SEED)


def heatmap(frame, title: str, midpoint: float, fmt_text: str, x_title: str, y_title: str, height: int = 430, diverging: bool = True):
    scale = [[0, "#e07a5f"], [.5, "#ffffff"], [1, "#087e8b"]] if diverging else [[0, "#ffffff"], [1, "#087e8b"]]
    values = frame.to_numpy(dtype=float)
    fig = go.Figure(go.Heatmap(
        z=values, x=[str(c) for c in frame.columns], y=[str(i) for i in frame.index], colorscale=scale,
        zmid=midpoint if diverging else None, text=[[("" if np.isnan(v) else fmt_text.format(v)) for v in row] for row in values],
        texttemplate="%{text}", textfont=dict(size=10), hovertemplate=f"{y_title}: %{{y}}<br>{x_title}: %{{x}}<br>Value: %{{z:.2f}}<extra></extra>",
        colorbar=dict(thickness=10, outlinewidth=0),
    ))
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title, yaxis=dict(autorange="reversed"))
    fig = chart_style(fig, height)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def targeting_chart(key: str, signals):
    if key == "blanket":
        traffic = signals["log"].groupby(["hour", "daypart"], as_index=False).size()
        fig = px.bar(traffic, x="hour", y="size", color="daypart", color_discrete_map={"Morning": "#9fd8cf", "Midday": "#3fa99e", "Evening": "#087e8b"},
                     labels={"hour": "Hour of day", "size": "Baskets", "daypart": "Daypart"}, title="Store traffic by hour: plays reach the most shoppers from 5–7pm")
        return chart_style(fig, 400)
    if key == "category":
        stores = signals["stores"]
        fig = px.scatter(stores, x="CDI", y="BDI", size="baskets", text="store_id", size_max=16,
                         labels={"CDI": "Category development index (salsa)", "BDI": "Brand development index (Brand A)"},
                         title="Store prioritization: category vs. brand development")
        fig.update_traces(marker=dict(color="#087e8b", line=dict(color="white", width=1)), textposition="top center", textfont=dict(size=9, color="#65788c"))
        fig.add_hline(y=100, line_dash="dot", line_color="#aab7c3")
        fig.add_vline(x=100, line_dash="dot", line_color="#aab7c3")
        x_hi, y_hi = stores.CDI.max() + 5, stores.BDI.max() + 5
        x_lo, y_lo = stores.CDI.min() - 5, stores.BDI.min() - 5
        for x, y, label, color in [(x_hi, y_lo, "GROW · category sells, brand lags", "#e07a5f"), (x_hi, y_hi, "DEFEND · both strong", "#087e8b"),
                                   (x_lo, y_hi, "BUILD CATEGORY · brand loyal, category small", "#65788c"), (x_lo, y_lo, "LOW PRIORITY", "#aab7c3")]:
            fig.add_annotation(x=x, y=y, text=label, showarrow=False, font=dict(size=10, color=color), xanchor="right" if x == x_hi else "left")
        return chart_style(fig, 430)
    if key == "crosssell":
        return heatmap(signals["affinity"]["lift"].round(2), "Basket co-purchase lift (1.0 = independent)", 1.0, "{:.1f}", "Complement", "Anchor", 470)
    if key == "conquest":
        return heatmap(signals["switching"], "Brand switching among salsa buyers (% of last quarter's buyers)", 0, "{:.0f}%", "Brand this quarter", "Brand last quarter", 400, diverging=False)
    index = signals["hour_index"].round(0)
    index.columns = [f"{h % 12 or 12}{'am' if h < 12 else 'pm'}" for h in index.columns]
    return heatmap(index, "Category sales index by hour (100 = sells in line with traffic)", 100, "{:.0f}", "Hour", "Category", 470)


def render_targeting(signals):
    st.markdown("## Targeting logic: what transaction data tells you")
    st.markdown("<p style='color:#65788c;margin-top:-.6rem;max-width:960px'>Targeting decisions come from transaction data before the campaign ever runs. Each option below shows the question it answers, the signals worth mining, the decision rule and how it changes the measurement plan. The charts are computed from a simulated log of 40,000 baskets across 24 stores.</p>", unsafe_allow_html=True)
    tabs = st.tabs([TARGETING[key] for key in TARGETING_PLAYBOOK])
    for tab, (key, play) in zip(tabs, TARGETING_PLAYBOOK.items()):
        with tab:
            left, right = st.columns([1, 1.25], gap="large")
            with left:
                st.markdown(f"<div class='eyebrow' style='margin-top:.6rem'>The question</div><h3 style='margin-top:0'>{play['question']}</h3>", unsafe_allow_html=True)
                st.markdown(bullet_panel("Transaction signals to mine", play["signals"]), unsafe_allow_html=True)
                st.markdown(f"<div class='insight' style='margin-top:.9rem'><b>Decision rule:</b> {play['decision']}</div>", unsafe_allow_html=True)
                st.markdown(f"<p style='color:#40566b;font-size:.92rem'><b>Measurement implication:</b> {play['measurement']}</p><p class='small-note'>Minimum data: {play['needs']}</p>", unsafe_allow_html=True)
            with right:
                st.plotly_chart(targeting_chart(key, signals), use_container_width=True)
                if key == "crosssell":
                    lift = signals["affinity"]["lift"]
                    best_anchor = lift[ADVERTISED_CATEGORY].drop(ADVERTISED_CATEGORY).idxmax()
                    st.markdown(f"<p class='small-note'>For the advertised salsa brand, the best anchor shelf is <b>{best_anchor}</b> (lift {lift.loc[best_anchor, ADVERTISED_CATEGORY]:.1f}). Top pairs across the store:</p>", unsafe_allow_html=True)
                    rules = signals["affinity"]["rules"].copy()
                    st.dataframe(rules.style.format({"Support": "{:.1%}", "Attach rate": "{:.0%}", "Lift": "{:.2f}"}), use_container_width=True, hide_index=True)
                if key == "conquest":
                    switching = signals["switching"]
                    st.markdown(f"<p class='small-note'>{switching.loc['Competitor B', 'Brand A']:.0f}% of Competitor B buyers already moved to Brand A last quarter vs. {switching.loc['Competitor C', 'Brand A']:.0f}% of Competitor C buyers, so B's shelf is the better conquest target.</p>", unsafe_allow_html=True)
                if key == "category":
                    stores = signals["stores"]
                    grow = stores.loc[(stores.CDI > 100) & (stores.BDI < 100)].sort_values("CDI", ascending=False)
                    st.markdown(f"<p class='small-note'>{len(grow)} of {len(stores)} stores fall in the <b>grow</b> quadrant ({', '.join(grow.store_id.head(5))}{'…' if len(grow) > 5 else ''}).</p>", unsafe_allow_html=True)
    st.markdown("#### Transaction signal library")
    html_table(
        [
            {"Signal": "Category & brand development index (CDI / BDI)", "Built from": "Item sales and basket counts by store", "Guides": "Which stores to prioritize; grow vs. defend", "Data maturity": "Early-stage"},
            {"Signal": "Seasonality index", "Built from": "Weekly category sales history", "Guides": "When to flight the campaign", "Data maturity": "Early-stage"},
            {"Signal": "Basket affinity (support, attach rate, lift)", "Built from": "Item lists per basket", "Guides": "Cross-sell pairs; which shelf hosts which ad", "Data maturity": "Mature"},
            {"Signal": "Hour × category index, traffic curve", "Built from": "Transaction timestamps", "Guides": "Daypart scheduling; play frequency", "Data maturity": "Mature"},
            {"Signal": "Price gap index", "Built from": "Shelf prices vs. competitor, by store", "Guides": "Where conquesting can win on value", "Data maturity": "Mature"},
            {"Signal": "Brand switching matrix", "Built from": "Repeat purchases in retailer loyalty data", "Guides": "Which competitor's shelf to conquest", "Data maturity": "If the retailer shares loyalty data"},
        ],
    )


DIMENSION_IMPLICATIONS = {
    "placement": "Sets the exposure proxy. Only shelf-edge screens define a neighboring product, so only they support a cannibalization check.",
    "targeting": "Adds outcomes. Cross-sell needs the complementary product; conquesting needs the competitor; daypart needs time-of-day data.",
    "offer": "Direct vs. inferred outcome. Coupons leave a redemption trace; awareness doesn't. A new product has no baseline at all.",
    "rollout": "Sets the comparison: holdout stores, staggered waves or play-frequency levels.",
}


def navigate(target: str):
    st.session_state.page = target


TODAY_SOURCES = {
    "brands": "https://www.quad.com/engage/in-store-retail-media-for-brands",
    "product": "https://www.quad.com/solutions/technology/in-store-connect",
    "guide": "https://www.quad.com/resources/marketing-guides/in-store-retail-media-networks-the-revolution-2-0",
    "savemart": "https://www.thesavemartcompanies.com/press-2024/blog-post-title-four-da6ab-2jeg2-2ca5r-268z2-zelrk",
    "homeland": "https://www.quad.com/newsroom/quad-expands-in-store-connect-retail-media-network-with-regional-grocer-homeland-stores",
    "smartfinal": "https://www.quad.com/newsroom/quad-expands-in-store-connect-retail-media-network-with-chedraui-usa",
    "vistar": "https://martech360.com/news/quad-expands-in-store-ads-with-vistar-media-technology/",
}

IN_STORE_CONNECT_TODAY = [
    ("Network", "Launched in 2024 with The Save Mart Companies (15 stores, plans for 179 more) and Homeland (15 stores). 25 Smart & Final stores in California launch this fall.", ("savemart", "homeland", "smartfinal")),
    ("Model", "End to end: content, technology, production and in-store execution, with creative from Quad's Betty agency.", ("smartfinal", "guide")),
    ("Formats", "Digital kiosks, endcaps, in-aisle and shelf screens, vertical banners.", ("savemart", "brands")),
    ("Targeting", "Location and time of day, ZIP code or retailer footprint, SKU, aisle or store type, with creative swaps by audience and data from 117M households.", ("savemart", "brands", "guide")),
    ("Buying", "Programmatic buying through Vistar Media (2025).", ("vistar",)),
    ("Measurement", "POS lift at SKU or brand level, coupon redemption rates, matched-market or QR attribution. DiGiorno case: a 23-point sales lift in four weeks.", ("brands",)),
]


def leverage_cards():
    waves = Design("entrance", "blanket", "awareness", "staggered")
    coupon = PRESETS["Regional pilot · endcap coupon holdout"]
    daypart = PRESETS["Daypart activation · checkout dose test"]
    waves_primary = run_design(*vars(waves).values())[1]["primary"]
    redemption = run_design(*vars(coupon).values())[1]["redemption"]
    breakdown = {row["daypart"]: row for row in run_design(*vars(daypart).values())[1]["daypart"]["breakdown"]}
    others = max(abs(breakdown[name]["estimate"]) for name in breakdown if name != TARGET_DAYPART)
    return [
        ("Use the rollout as the experiment", "Builds on phased network growth",
         "Networks expand in waves, like Save Mart's plan to grow from 15 stores by 179 more. The stores that haven't launched yet are a built-in comparison group, with no screens held back and no extra cost.",
         f"Launching in three waves recovers <b>{waves_primary['estimate'] / waves_primary['truth']:.0%}</b> of the injected lift, even with a seasonal upswing at launch.", waves),
        ("Pair redemptions with a holdout", "Builds on coupon redemption reporting",
         "Redemptions confirm the offer reached shoppers. A holdout sizes how much of it was incremental. Together they tell both the delivered story and the earned story.",
         f"Redemption-attributed sales run <b>{redemption['attributed'] / redemption['truth']:.1f}×</b> the incremental lift, so the holdout keeps the two apart.", coupon),
        ("Read results at the grain of the targeting", "Builds on location and time-of-day activation",
         "When content is scheduled by daypart, reading the result by daypart shows the lift where it happened and confirms the scheduling did its job.",
         f"Evening lift reads at <b>{breakdown[TARGET_DAYPART]['estimate']:.0f} units</b> per store-week (injected {breakdown[TARGET_DAYPART]['truth']:.0f}), with other dayparts within ±{others:.0f}. Weekly totals can't show when it happened.", daypart),
    ]


def page_overview():
    st.markdown(
        "<div class='hero'><div class='eyebrow'>IN-STORE RETAIL MEDIA · MEASUREMENT DESIGN</div><h1>Every rollout is a measurement opportunity.</h1>"
        "<p>In-Store Connect already ties screens to point-of-sale outcomes, from SKU-level lift to coupon redemptions. This lab explores how four design choices, "
        "placement, targeting, offer and rollout, decide which outcomes a campaign can prove, and how that proof grows as each retail partner's network matures.</p>"
        "<p style='margin-top:.9rem;font-size:.9rem;color:#9fc3d4'>An independent exploration built around Quad's In-Store Connect. All campaign data in the lab is simulated.</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown("## What In-Store Connect does today")
    st.markdown("<p style='color:#65788c;margin-top:-.6rem'>From Quad's public materials and partner announcements. eMarketer projects in-store retail media spend growing from $370M in 2024 to $1.06B in 2028, as cited by Quad.</p>", unsafe_allow_html=True)
    cards = []
    for title, text, sources in IN_STORE_CONNECT_TODAY:
        links = " · ".join(f"<a href='{TODAY_SOURCES[key]}' target='_blank'>source</a>" if len(sources) == 1 else f"<a href='{TODAY_SOURCES[key]}' target='_blank'>{i + 1}</a>" for i, key in enumerate(sources))
        cards.append(f"<div class='panel'><h4>{title}</h4><p style='color:#40566b;font-size:.9rem;line-height:1.55;margin:0 0 .6rem'>{text}</p><p class='small-note bottom' style='margin:0'>Sources: {links}</p></div>")
    card_grid(cards, 3)

    st.markdown("## Three places design adds leverage")
    st.markdown("<p style='color:#65788c;margin-top:-.6rem'>Each one builds on something the network already does. The numbers come from the lab's simulation, where the true lift is known because it was injected.</p>", unsafe_allow_html=True)
    leverage = leverage_cards()
    card_grid([
        f"<div class='panel'><div class='eyebrow' style='margin-bottom:.3rem'>{builds_on}</div><h3 style='margin:.1rem 0 .5rem;font-size:1.2rem'>{title}</h3>"
        f"<p style='color:#40566b;font-size:.9rem;line-height:1.55'>{text}</p><div class='insight bottom' style='font-size:.88rem'>{illustration}</div></div>"
        for title, builds_on, text, illustration, _ in leverage
    ])
    button_row([("Open this design →", f"leverage_{index}", open_in_builder, (card[4],)) for index, card in enumerate(leverage)])
    audience("Measurement built into each rollout gives every campaign a result the retailer can take to the brand's next planning meeting.",
             "Results arrive with the comparison behind them, so lift reads the same way across retailers and campaigns.")

    st.markdown("## From design to a defensible claim")
    st.markdown("<p style='color:#65788c;margin-top:-.6rem'>The design determines the data, the data determines the comparison, and the comparison determines what can be claimed.</p>", unsafe_allow_html=True)
    st.markdown(to_img(measurement_flow(), "Flow from design choices to data, comparison, read-out and claim"), unsafe_allow_html=True)

    st.markdown("## Four design dimensions")
    st.markdown("<p style='color:#65788c;margin-top:-.6rem'>A campaign is one choice from each. They combine freely: 240 combinations, each with its own measurement plan.</p>", unsafe_allow_html=True)
    dimension_cards = []
    for dim, options in DIMENSIONS.items():
        opts = "".join(f"<span class='opt'>{label.split(' /')[0]}</span>" for label in options.values())
        dimension_cards.append(f"<div class='tier-card'><div class='tier-number'>DIMENSION</div><h3>{DIMENSION_TITLES[dim]}</h3><div>{opts}</div><p>{DIMENSION_IMPLICATIONS[dim]}</p></div>")
    card_grid(dimension_cards)
    st.button("Appendix: store map, shopper signals and rollout patterns →", on_click=navigate, args=("Appendix · Store map & signals",))

    st.markdown("## Where the demo goes next")
    steps = [
        ("2 · Design the test", "Start from the brand's goal. See how it will be proven, what it can't support yet, and the result in incremental revenue and iROAS.", "Design the test"),
        ("3 · Answer the brand's questions", "Six questions brands ask about in-store lift, each with an answer, a design and a proof point.", "Answer the brand's questions"),
        ("4 · Scale with the retailer", "How proof grows with a retailer's data, from a regional pilot to a fully instrumented network.", "Scale with the retailer"),
    ]
    card_grid([f"<div class='panel'><h4>{title}</h4><p style='color:#40566b;font-size:.9rem;line-height:1.55;margin:0'>{text}</p></div>" for title, text, _ in steps])
    button_row([("Open", f"walk_{target}", navigate, (target,)) for _, _, target in steps])
    st.markdown("<p class='small-note' style='margin-top:.8rem'>The appendix in the sidebar holds the store map and shopper signals, the design guardrails, and the holdout and waves & dose methods with full statistical output.</p>", unsafe_allow_html=True)
    st.markdown("<div class='insight'><b>Design principle:</b> keep the experiment no more complex than the decision requires. A simple, well-executed holdout supports a strong claim; richer designs unlock richer claims as the partner's data grows.</div>", unsafe_allow_html=True)


def page_field_guide():
    header("APPENDIX · PLAN THE ACTIVATION", "Placements and shopper signals", "What each design choice looks like on the store floor, which first-party transaction data informs it, and what it changes about measurement.")
    audience("Your first-party transaction data picks the placements and targeting, and it's the same data that later proves the result.",
             "Targeting grounded in how shoppers actually buy: co-purchases, hour-of-day shopping missions and brand switching.")

    st.markdown("## Placement: where the screen sits sets the exposure proxy")
    st.markdown(to_img(store_map(), "Store floor plan showing endcap, checkout, shelf-edge and entrance screen placements"), unsafe_allow_html=True)
    card_grid([
        f"<div class='panel'><div style='display:flex;gap:.7rem;align-items:center;margin-bottom:.45rem'><div style='flex:none;width:28px;height:28px;border-radius:50%;background:{PLACEMENT_COLORS[key]};color:white;font-weight:700;display:flex;align-items:center;justify-content:center'>{number}</div>"
        f"<b style='color:#172b42;line-height:1.3'>{title}</b></div><div style='color:#65788c;font-size:.88rem;line-height:1.5'>{note}</div></div>"
        for key, number, title, note in PLACEMENT_NOTES
    ])

    render_targeting(signals_data())

    st.markdown("## Offer mechanic: direct trace or inferred effect")
    html_table(
        [
            {"Offer": "Pure awareness", "Outcome you observe": "Aggregate sales only", "Measurement consequence": "Effect inferred from store-level comparisons"},
            {"Offer": "Digital coupon tie-in", "Outcome you observe": "Redemption events (trackable scans)", "Measurement consequence": "Direct outcome, but redemptions overstate lift. Still needs a control."},
            {"Offer": "Bundle / multi-buy", "Outcome you observe": "Basket units per transaction", "Measurement consequence": "Aggregate or basket-level; no link to a person"},
            {"Offer": "New product trial", "Outcome you observe": "First purchases, with no history", "Measurement consequence": "No pre-period → cross-sectional design instead of DiD"},
        ],
    )

    st.markdown("## Rollout structure: where the comparison comes from")
    st.markdown(to_img(rollout_patterns(), "Three rollout patterns: on/off, staggered and dose variation"), unsafe_allow_html=True)

    st.markdown("<p style='color:#65788c;margin-top:1rem'>Two rollout structures are worked through end to end in the appendix, with full statistical output:</p>", unsafe_allow_html=True)
    html_table(
        [
            {"Appendix method": "Holdout method", "Campaign design": "One fixed treatment; one holdout", "Measurement": "Before/after DiD + Welch interval", "Data requirements": "Store × week product sales", "Question answered": "Did the campaign move sales?"},
            {"Appendix method": "Waves & dose method", "Campaign design": "Staggered launch + variable dose", "Measurement": "Store & week fixed effects; nonlinear dose response", "Data requirements": "Longitudinal store panel + play intensity", "Question answered": "When and at what dose does lift emerge?"},
        ],
    )
    st.button("Next: design the test →", on_click=navigate, args=("Design the test",))


def page_builder():
    header("STEP 2 · DESIGN THE TEST", "Start from the brand's goal, leave with a proof plan", "Pick what the brand wants to achieve. The lab proposes a matched design, shows how it will be proven, and translates the simulated result into incremental revenue and iROAS.")
    audience("Know before launch which claims the network can support, and which data to collect to support more.",
             "Every lift number comes with the comparison behind it, so it holds up in your analytics team's review.")
    st.markdown("### Start from the brand's goal")
    current = Design(**st.session_state.design)
    goal_cards = []
    for goal, proposal, outcome, design in GOALS:
        selected = design == current
        badge = "<span class='badge valid' style='margin-bottom:.4rem;align-self:flex-start'>Selected</span>" if selected else ""
        goal_cards.append(f"<div class='panel{' selected' if selected else ''}'>{badge}<h3 style='margin:.1rem 0 .45rem;font-size:1.15rem'>{goal}</h3>"
                          f"<p style='color:#40566b;font-size:.88rem;line-height:1.5;margin:0 0 .5rem'><b>We'd propose:</b> {proposal}</p>"
                          f"<p class='bottom' style='color:#087e8b;font-size:.86rem;line-height:1.5;margin:0'><b>The brand can say:</b> {outcome}</p></div>")
    for start in (0, 3):
        card_grid(goal_cards[start:start + 3], 3)
        button_row([("Choose this goal", f"goal_{index}", open_in_builder, (GOALS[index][3],)) for index in range(start, start + 3)])
    with st.expander("Advanced: fine-tune the four design dimensions or load a preset"):
        version = st.session_state.design_version
        names = ["Custom", *PRESETS]
        chosen = st.selectbox("Start from a preset", names, index=names.index(st.session_state.preset_name), key=f"preset_{version}")
        if chosen != st.session_state.preset_name:
            if chosen in PRESETS:
                load_design(PRESETS[chosen], chosen)
                st.rerun()
            st.session_state.preset_name = chosen
        cols = st.columns(4)
        for col, (dim, options) in zip(cols, DIMENSIONS.items()):
            with col:
                keys = list(options)
                value = st.selectbox(DIMENSION_TITLES[dim], keys, index=keys.index(st.session_state.design[dim]), format_func=options.get, key=f"{dim}_{version}")
                if value != st.session_state.design[dim]:
                    st.session_state.design[dim] = value
                    st.session_state.preset_name = "Custom"
                    st.session_state.design_version += 1
                    st.rerun()
    design = Design(**st.session_state.design)
    st.markdown(f"<div class='sentence'>{design.describe()}</div>", unsafe_allow_html=True)
    rec, result = run_design(design.placement, design.targeting, design.offer, design.rollout)
    st.markdown("### Measurement recommendation")
    render_recommendation(design, rec)
    st.markdown("### Simulated readout")
    st.markdown(f"<p class='small-note'>{'40 stores × 16 weeks' + (' × 3 dayparts' if design.targeting == 'daypart' else '')}. A market-wide seasonal lift of 6% starts in week 9 in every store; good designs difference it out. Targeted designs reserve half the stores as holdouts; blanket designs run everywhere.</p>", unsafe_allow_html=True)
    render_result(rec, result)


def page_failures():
    header("APPENDIX · DESIGN GUARDRAILS", "Checks to run before launch", "Two patterns seen across retail media that the playbook designs around. Each can produce a confident-looking number, and each is resolved by one design change.")
    audience("Each guardrail is a rollout or offer choice made before launch, so it adds proof without adding cost.",
             "Each check comes down to one question: what is the lift compared against, and could that group see the ad?")
    for index, case in enumerate(WEAK_DESIGNS, 1):
        weak, fix = case["design"], case["fix"]
        weak_rec, weak_result = run_design(weak.placement, weak.targeting, weak.offer, weak.rollout)
        fix_rec, fix_result = run_design(fix.placement, fix.targeting, fix.offer, fix.rollout)
        wp, fp = weak_result["primary"], fix_result["primary"]
        kind = weak_result["fmt"]
        st.markdown(f"## {index:02d} · {case['name']}")
        left, right = st.columns([1.1, 1])
        with left:
            st.markdown(
                f"<div class='verdict invalid'><span class='badge invalid'>Guardrail · {case['guardrail']}</span><h3>{weak_rec.headline}</h3>"
                f"<p><b>{weak.short()}</b></p><p>{case['story']}</p><p><b>How it would be read:</b> {weak_rec.plain}</p></div>"
                f"<div class='verdict {fix_rec.status}'><span class='badge {fix_rec.status}'>The design change · {STATUS_LABEL[fix_rec.status].lower()}</span><h3>{fix_rec.headline}</h3>"
                f"<p><b>{fix.short()}</b></p><p>{case['fix_note']}</p><p><b>How it's read:</b> {fix_rec.plain}</p></div>",
                unsafe_allow_html=True,
            )
        with right:
            fig = go.Figure()
            labels = ["As designed", "With the change"]
            fig.add_trace(go.Bar(x=labels, y=[wp["truth"], fp["truth"]], name="True lift", marker_color="#e07a5f", text=[fmt(wp["truth"], kind), fmt(fp["truth"], kind)], textposition="inside", insidetextanchor="start", textfont=dict(color="white")))
            fig.add_trace(go.Bar(x=labels, y=[wp["estimate"], fp["estimate"]], name="Estimate", marker_color=["#aab7c3", "#087e8b"], text=[fmt(wp["estimate"], kind), fmt(fp["estimate"], kind)], textposition="inside", insidetextanchor="start", textfont=dict(color="white"),
                                 error_y=dict(type="data", symmetric=False, array=[wp["ci_high"] - wp["estimate"], fp["ci_high"] - fp["estimate"]], arrayminus=[wp["estimate"] - wp["ci_low"], fp["estimate"] - fp["ci_low"]], color="#183a50")))
            fig.update_layout(title=f"Recovered: {wp['estimate'] / wp['truth']:.0%} of truth → {fp['estimate'] / fp['truth']:.0%} of truth", yaxis_title=weak_result["unit"], barmode="group")
            st.plotly_chart(chart_style(fig, 360), use_container_width=True)
            b1, b2 = st.columns(2)
            with b1: st.button("Open as designed", key=f"weak_{index}", on_click=open_in_builder, args=(weak,), use_container_width=True)
            with b2: st.button("Open with the change", key=f"fix_{index}", on_click=open_in_builder, args=(fix,), use_container_width=True)
    st.markdown("<div class='insight'><b>The pattern:</b> none of these changes needs a more sophisticated model. Each adds a <i>comparison</i> the original design lacked: stores that haven't launched yet. The proof comes from the design; no amount of analysis can add a comparison that was never built in.</div>", unsafe_allow_html=True)


def objection_cards():
    holdout = run_design(*vars(PRESETS["Regional pilot · endcap coupon holdout"]).values())[1]
    blanket = run_design(*vars(PRESETS["Guardrail · Blanket blast"]).values())[1]["primary"]
    waves = run_design(*vars(Design("entrance", "blanket", "awareness", "staggered")).values())[1]["primary"]
    basket = run_design(*vars(PRESETS["Basket builder · shelf-edge cross-sell bundle"]).values())[1]
    daypart = {row["daypart"]: row for row in run_design(*vars(PRESETS["Daypart activation · checkout dose test"]).values())[1]["daypart"]["breakdown"]}
    hp, redemption = holdout["primary"], holdout["redemption"]
    decomposition = {item["key"]: item for item in basket["decomposition"]}
    gross, shifted = decomposition["sales"]["estimate"], -decomposition["neighbor"]["estimate"]
    others = max(abs(row["estimate"]) for name, row in daypart.items() if name != TARGET_DAYPART)
    return [
        ("How do we know shoppers wouldn't have bought anyway?",
         "We compare against stores that didn't run the campaign in the same weeks, so only the difference counts as lift. Seasonality and promotions hit both groups and cancel out.",
         f"A holdout read lands at <b>{hp['estimate'] / hp['truth']:.0%}</b> of the true lift. A before/after read of a network-wide launch reports <b>{blanket['estimate'] / blanket['truth']:.0%}</b>.",
         PRESETS["Regional pilot · endcap coupon holdout"]),
        ("Redemptions look great. Isn't that our ROI?",
         "Redemptions prove the offer reached shoppers. Some redeemers would have bought anyway, so we pair them with a holdout to show the incremental part.",
         f"Redemption-attributed sales run <b>{redemption['attributed'] / redemption['truth']:.1f}×</b> the incremental lift. Both numbers go in the report, each labeled for what it means.",
         PRESETS["Regional pilot · endcap coupon holdout"]),
        ("Holding back stores costs us revenue.",
         "No store has to miss out. Roll out in waves: the stores still waiting for their launch week are the comparison group for a few weeks, then they go live too.",
         f"A three-wave rollout recovers <b>{waves['estimate'] / waves['truth']:.0%}</b> of the true lift, even with a seasonal upswing at launch.",
         Design("entrance", "blanket", "awareness", "staggered")),
        ("We're a regional grocer. Can we even measure this?",
         "Yes. A 20-store campaign with 20 comparison stores and ordinary store-week sales data gives a readable result. That's where most networks start.",
         f"With 20 + 20 stores the result is <b>{hp['estimate']:.0f} units</b> per store-week, give or take <b>{(hp['ci_high'] - hp['ci_low']) / 2 / hp['estimate']:.0%}</b>.",
         PRESETS["Regional pilot · endcap coupon holdout"]),
        ("Is the ad just moving sales from our other products?",
         "Shelf-edge screens sit next to one product, so we also track the neighboring product and report lift net of any shift between them.",
         f"Gross lift of <b>{gross:.0f} units</b> per store-week, of which <b>{shifted:.0f}</b> came from the neighboring product, reported as net <b>{gross - shifted:.0f} units</b>.",
         PRESETS["Basket builder · shelf-edge cross-sell bundle"]),
        ("Can we see when the ad actually works?",
         "When content is scheduled by time of day, we read the result by time of day too, so the brand sees the lift where it happened.",
         f"Evening lift of <b>{daypart[TARGET_DAYPART]['estimate']:.0f} units</b> per store-week, with every other daypart within <b>±{others:.0f}</b>.",
         PRESETS["Daypart activation · checkout dose test"]),
    ]


def page_objections():
    header("STEP 3 · ANSWER THE BRAND'S QUESTIONS", "Six questions brands ask about in-store lift", "Each answer is written the way it would be said in a meeting, paired with the design to propose and a proof point from the simulation.")
    audience("A ready answer backed by a design turns a measurement objection into a scoping conversation.",
             "Each answer comes with the comparison behind it, so the brand's analysts can check the logic.")
    cards = objection_cards()
    html = [
        f"<div class='panel'><div class='eyebrow' style='margin-bottom:.35rem'>The brand asks</div><h3 style='margin:0 0 .55rem;font-size:1.15rem;line-height:1.35'>“{question}”</h3>"
        f"<p style='color:#40566b;font-size:.9rem;line-height:1.55;margin:0 0 .5rem'>{answer}</p>"
        f"<p class='small-note bottom' style='margin:.2rem 0 .45rem'><b>Propose:</b> {design.short()}</p><div class='insight' style='margin:0;font-size:.87rem'>{proof}</div></div>"
        for question, answer, proof, design in cards
    ]
    for start in (0, 3):
        card_grid(html[start:start + 3], 3)
        button_row([("Open this design", f"objection_{index}", open_in_builder, (cards[index][3],)) for index in range(start, start + 3)])
    st.markdown("<div class='insight'><b>The common thread:</b> every answer is a comparison group: other stores, not-yet-launched stores, the neighboring product, other dayparts. Agreeing on that comparison at proposal stage is what makes the readout land at renewal.</div>", unsafe_allow_html=True)


def ask(question: str):
    st.session_state.pending_question = question


def clear_chat():
    st.session_state.chat = []


def page_assistant():
    header("ASK THE LAB", "Ask anything about the lab", "A Claude-powered assistant that knows every page, brand goal, objection, retailer stage and simulated result in the lab. It answers in business language and points you to where to look.")
    if not api_key_configured():
        st.info("The assistant needs an Anthropic API key. On Render, add ANTHROPIC_API_KEY under the service's Environment settings; locally, put it in a .env file. Then reload the page.", icon="🔑")
        return
    with st.spinner("Preparing the assistant's brief on the lab…"):
        knowledge_base()
    history = st.session_state.setdefault("chat", [])
    if not history:
        st.markdown("<div class='eyebrow' style='margin-top:.4rem'>Try a question</div>", unsafe_allow_html=True)
        for start in (0, 2):
            button_row([(question, f"suggest_{index}", ask, (question,)) for index, question in enumerate(SUGGESTED_QUESTIONS[start:start + 2], start)])
    for message in history:
        if message["role"] in ("user", "assistant"):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    question = st.chat_input("Ask about designs, measurement, brand objections or retailer stages") or st.session_state.pop("pending_question", None)
    if question:
        history.append({"role": "user", "content": question})
        history.append(context_note(st.session_state.get("last_content_page", "Why proof matters"), Design(**st.session_state.design)))
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            reply = st.write_stream(stream_reply(history))
        history.append({"role": "assistant", "content": reply if isinstance(reply, str) else "".join(map(str, reply))})
    if history:
        st.button("Clear conversation", on_click=clear_chat)
    st.markdown("<p class='small-note' style='margin-top:1rem'>Answers come from Claude (claude-opus-5), grounded in this lab's content. All numbers are from the lab's simulation.</p>", unsafe_allow_html=True)


def page_maturity():
    header("STEP 4 · SCALE WITH THE RETAILER", "A measurement roadmap, from pilot to fully instrumented network", "In-store networks deploy modularly, from a single-region pilot to a fully instrumented rollout with aisle-level traffic and basket data. The right proof plan is the most rigorous one the retailer's data can support today.")
    audience("Each campaign's measurement makes the case for the next data investment: timestamps, play logs, aisle traffic.",
             "Expect store-level lift from a new network and shelf-level net lift from a fully instrumented one.")
    stage_cards = []
    for index, tier in enumerate(MATURITY):
        dims_html = ""
        for dim, options in DIMENSIONS.items():
            opts = "".join(f"<span class='opt{'' if key in tier['allowed'][dim] else ' off'}'>{label.split(' /')[0]}</span>" for key, label in options.items())
            dims_html += f"<p style='margin:.6rem 0 .2rem;font-size:.74rem;font-weight:700;color:#40566b;text-transform:uppercase;letter-spacing:.08em'>{DIMENSION_TITLES[dim]}</p><div>{opts}</div>"
        stage_cards.append(
            f"<div class='tier-card'><div class='tier-number'>STAGE {index + 1} · LIKE {tier['comparable'].upper()}</div><h3>{tier['name']}</h3>"
            f"<p>{tier['summary']}</p><div>{dims_html}</div><div class='tier-meta'>Measurement ceiling · {tier['measurement']}</div>"
            f"<p><b>Representative design:</b> {tier['example'].describe()}</p></div>"
        )
    card_grid(stage_cards, extra_class="stages")
    button_row([("Open representative design", f"maturity_{index}", open_in_builder, (tier["example"],)) for index, tier in enumerate(MATURITY)])
    st.markdown("### What each step up unlocks")
    html_table(
        [
            {"Stage": MATURITY[0]["name"], "New data": "Store × week POS, holdout stores", "New designs": "Holdout on/off, staggered waves, coupon tie-ins", "New claims": "Store-level incremental lift"},
            {"Stage": MATURITY[1]["name"], "New data": "Transaction timestamps, aisle traffic, screen play logs", "New designs": "Daypart targeting, conquesting, dose variation, bundles, product launches", "New claims": "When lift happens, how it scales with frequency, category-level halo and share shift"},
            {"Stage": MATURITY[2]["name"], "New data": "Aisle-level traffic counts, basket-level POS", "New designs": "Shelf-edge screens, cross-sell bundles", "New claims": "Lift net of cannibalization and halo, shelf-level frequency optimization"},
        ],
    )
    st.markdown("<div class='insight'><b>The business narrative:</b> start a new partner with a measurement plan they can execute cleanly, such as a store holdout with a DiD readout, and use each campaign to justify the next data investment. Promising shelf-level net lift to a partner without aisle-level data sets up a result nobody can defend.</div>", unsafe_allow_html=True)
    st.markdown("<p class='small-note'>Partner references come from public announcements (<a href='https://www.quad.com/newsroom/quad-expands-in-store-connect-retail-media-network-with-regional-grocer-homeland-stores'>Homeland, 2024</a> · "
                "<a href='https://www.thesavemartcompanies.com/press-2024/blog-post-title-four-da6ab-2jeg2-2ca5r-268z2-zelrk'>Save Mart, 2024</a> · "
                "<a href='https://www.quad.com/newsroom/quad-expands-in-store-connect-retail-media-network-with-chedraui-usa'>Smart & Final, 2026</a>). "
                "Stage data profiles are illustrative, and all data in this lab is simulated.</p>", unsafe_allow_html=True)


def page_easy(data):
    result = data["easy"]
    header("APPENDIX · HOLDOUT METHOD", "Simple treatment / control", "A fixed campaign in a balanced store holdout. A straightforward difference-in-differences estimate removes baseline store gaps and common week-to-week movement.")
    st.markdown("### Campaign design")
    st.markdown("<div class='assumption'><b>Treatment:</b> 20 stores run screens with one fixed ad and frequency. <b>Control:</b> 20 matched-in-spirit stores keep screens off. The campaign starts after week 8; no store changes status mid-test.</div>", unsafe_allow_html=True)
    with st.expander("Data generation assumptions", expanded=False):
        st.write("40 stores × 16 weeks; each store has a persistent sales baseline. Shared weekly seasonality and random store/week noise are added. Treatment stores receive a true +8% sales lift only in the 8 campaign weeks; the seed is fixed so the same data and estimates recur on every visit.")
    means = result["means"]
    fig = px.line(means, x="week", y="sales", color="group", markers=True,
                  color_discrete_map={"Treatment": "#087e8b", "Control": "#aab7c3"},
                  labels={"week": "Week", "sales": "Average weekly product sales", "group": "Group"},
                  title="Average weekly sales · campaign starts at week 9")
    fig.add_vline(x=8.5, line_dash="dash", line_color="#e07a5f", annotation_text="Campaign starts", annotation_position="top left")
    chart_style(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Difference-in-differences · step by step")
    stats = []
    stats.append(stat_html("Pre-period treatment − control", f"{money(result['pre_gap'])} units"))
    stats.append(stat_html("Campaign-period treatment − control", f"{money(result['post_gap'])} units"))
    stats.append(stat_html("DiD = post gap − pre gap", f"{money(result['estimate'])} units"))
    card_grid(stats)
    st.markdown(f"**Uncertainty:** 95% Welch confidence interval **[{money(result['ci_low'])}, {money(result['ci_high'])}]** units per store-week · p = **{p_label(result['p_value'])}**. **Injected effect:** {money(result['true_effect'])} units per store-week (+8% of each treatment store's baseline, averaged across treatment stores).")
    st.progress(float(np.clip(1 - abs(result["estimate"] - result["true_effect"]) / max(abs(result["true_effect"]), 1), 0, 1)), text=f"Recovery accuracy · estimate is {abs(result['estimate'] - result['true_effect']):.1f} units from injected effect")
    st.markdown("<div class='insight'><b>Why this fits:</b> with one fixed treatment and one holdout, the main risk is pre-existing store differences or market-wide movement—not dose or timing complexity. DiD subtracts the baseline gap and shared time change, giving a transparent estimate that is easy to explain and audit.</div>", unsafe_allow_html=True)


def page_medium(data):
    result = data["medium"]
    header("APPENDIX · WAVES & DOSE METHOD", "Staggered rollout + dose variation", "Three rollout waves make timing visible; variable play intensity lets us estimate a diminishing-returns response while controlling for persistent store differences and common weekly shocks.")
    st.markdown("### Campaign design")
    st.markdown("<div class='assumption'><b>60 stores:</b> 30 treatment stores launch in three waves (weeks 5, 7 and 9); 30 never-treated controls. Treatment stores receive low, medium or high frequency. Dose is zero until a store launches, then remains at its assigned intensity.</div>", unsafe_allow_html=True)
    with st.expander("Data generation assumptions", expanded=False):
        st.write("A 16-week balanced store panel has persistent store baselines, shared weekly seasonality and random noise. The true response is 150 × (1 − exp(−1.1 × dose)) units per store-week after launch. This saturating curve and its parameters are retained for recovery checks.")
    sub = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=.08,
                        subplot_titles=["Wave 1 · launch week 5", "Wave 2 · launch week 7", "Wave 3 · launch week 9"])
    colors = {"Treatment": "#087e8b", "Control": "#b6c3cd"}
    for row, wave in enumerate(["Wave 1", "Wave 2", "Wave 3"], 1):
        wave_store = result["data"].loc[result["data"].wave == wave].groupby("week", as_index=False).sales.mean()
        control = result["data"].loc[result["data"].wave == "Control"].groupby("week", as_index=False).sales.mean()
        sub.add_trace(go.Scatter(x=wave_store.week, y=wave_store.sales, mode="lines+markers", name=wave, line=dict(color=colors["Treatment"], width=2.5), showlegend=(row == 1)), row=row, col=1)
        sub.add_trace(go.Scatter(x=control.week, y=control.sales, mode="lines", name="Never-treated control", line=dict(color=colors["Control"], width=2, dash="dot"), showlegend=(row == 1)), row=row, col=1)
        sub.add_vline(x=result["wave_launches"][row] - .5, line_dash="dash", line_color="#e07a5f", row=row, col=1)
    sub.update_xaxes(title_text="Week", row=3, col=1, showgrid=False)
    sub.update_yaxes(title_text="Avg sales", gridcolor="#edf1f4")
    sub.update_layout(height=660, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white", font=dict(family="DM Sans", color="#52677b"), margin=dict(l=15, r=15, t=50, b=15), legend=dict(orientation="h", y=1.04, x=1, xanchor="right"))
    st.plotly_chart(sub, use_container_width=True)
    st.markdown("### Pre-trend check")
    pre_left, pre_right = st.columns([1.35, 1.0])
    with pre_left:
        pre = go.Figure()
        for group, color, dash in [("Treatment", "#087e8b", "solid"), ("Control", "#aab7c3", "dot")]:
            rows = result["pretrend"].loc[result["pretrend"].group == group]
            pre.add_trace(go.Scatter(x=rows.week, y=rows.sales, mode="lines+markers", name=f"{group} stores", line=dict(color=color, width=2.5, dash=dash)))
        pre.update_layout(title="Average sales before any wave launches (weeks 1–4)", xaxis_title="Week", yaxis_title="Avg sales", xaxis=dict(dtick=1))
        chart_style(pre, 330)
        st.plotly_chart(pre, use_container_width=True)
    with pre_right:
        trend = result["pretrend_test"]
        result_card("Pre-launch slope difference · treatment − control", f"{trend['estimate']:+.2f} units / week", f"95% CI [{trend['ci_low']:.2f}, {trend['ci_high']:.2f}]")
        st.markdown(f"**p = {p_label(trend['p_value'])}** (store & week FE, clustered by store). A slope difference indistinguishable from zero supports the parallel-trends assumption the fixed-effects model relies on.")
    st.markdown("### Fixed-effects panel model")
    st.markdown("Each store and each week gets its own intercept. The saturating dose shape is profiled from the panel; uncertainty is clustered by store.")
    response = result["response"]
    left, right = st.columns([1.0, 1.35])
    with left:
        result_card("Estimated response amplitude", f"{money(response['estimate'])} units", f"95% CI [{money(response['ci_low'])}, {money(response['ci_high'])}]")
        result_card("Dose curve curvature · fitted b", f"{result['fitted_b']:.2f}", f"Injected b = {result['true_b']:.2f}")
        st.markdown(f"**Cluster-robust p-value:** {p_label(response['p_value'])}  \n**Model:** sales ~ saturating dose response + store FE + week FE  \n**Injected amplitude:** {money(result['amplitude'])} units")
    with right:
        dose_grid = np.linspace(0, 1.5, 100)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result["binned"].dose_bin, y=result["binned"].twfe_sales, mode="markers", name="Simulated panel · binned", marker=dict(color="#91a4b3", size=10, line=dict(color="white", width=1))))
        fig.add_trace(go.Scatter(x=dose_grid, y=response["estimate"] * (1 - np.exp(-result["fitted_b"] * dose_grid)), mode="lines", name="Fitted response", line=dict(color="#087e8b", width=3)))
        fig.add_trace(go.Scatter(x=dose_grid, y=result["amplitude"] * (1 - np.exp(-result["true_b"] * dose_grid)), mode="lines", name="True lift", line=dict(color="#e07a5f", width=2, dash="dash")))
        fig.update_layout(title="Dose-response · fixed-effects adjusted sales lift", xaxis_title="Assigned dose intensity", yaxis_title="Adjusted lift (units / week)")
        chart_style(fig, 375)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**Reference point:** at dose 0.9, injected lift = **{money(result['true_effect'])}** units; fitted lift = **{money(response['estimate'] * (1 - np.exp(-result['fitted_b'] * 0.9)))}** units. The pre-launch period has no treatment exposure; the wave panels above let you visually inspect whether trajectories move only after rollout.")
    st.markdown("<div class='insight'><b>Why this fits:</b> a single before/after comparison would discard the rollout timing and treat a low-frequency store like a high-frequency store. Store and week fixed effects use the panel structure; the dose-response curve tests whether more plays deliver more lift, with the expected diminishing returns.</div>", unsafe_allow_html=True)


if page == "Why proof matters":
    page_overview()
elif page == "Appendix · Store map & signals":
    page_field_guide()
elif page == "Design the test":
    page_builder()
elif page == "Appendix · Design guardrails":
    page_failures()
elif page == "Answer the brand's questions":
    page_objections()
elif page == "Scale with the retailer":
    page_maturity()
elif page == "Ask the lab":
    page_assistant()
elif page == "Appendix · Holdout method":
    page_easy(demo_data())
elif page == "Appendix · Waves & dose method":
    page_medium(demo_data())

st.markdown("<div class='footer'>Proof at the Shelf · synthetic data only · fixed-seed reproducibility · statistical results are illustrative, not business guidance.</div>", unsafe_allow_html=True)
