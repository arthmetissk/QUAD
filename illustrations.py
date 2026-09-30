"""Inline SVG illustrations for the overview page."""
from __future__ import annotations

import base64

import numpy as np

INK, MUTED, LINE, TEAL, NAVY, CORAL, AMBER, FLOOR = "#172b42", "#65788c", "#d3dde5", "#087e8b", "#183a50", "#e07a5f", "#b7791f", "#f1f6f8"
PLACEMENT_COLORS = {"endcap": TEAL, "checkout": NAVY, "shelf": CORAL, "entrance": AMBER}
FONT = "font-family='Helvetica, Arial, sans-serif'"


def to_img(svg: str, alt: str) -> str:
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"<img src='data:image/svg+xml;base64,{encoded}' alt='{alt}' style='width:100%;height:auto;display:block'/>"


def _marker(x: float, y: float, number: int, color: str) -> str:
    return (f"<circle cx='{x}' cy='{y}' r='13' fill='{color}' stroke='white' stroke-width='2.5'/>"
            f"<text x='{x}' y='{y + 4.5}' text-anchor='middle' font-size='13' font-weight='700' fill='white' {FONT}>{number}</text>")


def store_map() -> str:
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 960 470'>",
             "<defs><marker id='arrow' viewBox='0 0 10 10' refX='8' refY='5' markerWidth='7' markerHeight='7' orient='auto'>"
             f"<path d='M0,0 L10,5 L0,10 z' fill='#9fb2c0'/></marker></defs>",
             f"<rect x='20' y='20' width='920' height='430' rx='18' fill='white' stroke='{LINE}' stroke-width='2'/>",
             f"<rect x='40' y='36' width='880' height='34' rx='6' fill='{FLOOR}'/>",
             f"<text x='480' y='58' text-anchor='middle' font-size='13' fill='{MUTED}' {FONT}>Dairy · Meat · Deli</text>",
             f"<rect x='40' y='84' width='118' height='250' rx='6' fill='{FLOOR}'/>",
             f"<text x='99' y='214' text-anchor='middle' font-size='13' fill='{MUTED}' {FONT}>Produce</text>"]
    categories = ["Bakery", "Cereal", "Snacks", "Beverages", "Pasta", "Frozen", "Household"]
    for index, name in enumerate(categories):
        x = 200 + index * 70
        highlight_endcap = index == 5
        parts.append(f"<rect x='{x}' y='96' width='30' height='12' rx='2' fill='#d3ebe6'/>")
        parts.append(f"<rect x='{x}' y='112' width='30' height='186' rx='3' fill='#e3eaef'/>")
        parts.append(f"<rect x='{x}' y='302' width='30' height='12' rx='2' fill='{TEAL if highlight_endcap else '#d3ebe6'}'/>")
        parts.append(f"<text x='{x + 15}' y='208' text-anchor='middle' font-size='11' fill='{MUTED}' transform='rotate(-90 {x + 15} 205)' {FONT}>{name}</text>")
    snacks_x = 200 + 2 * 70
    for y in (168, 238):
        parts.append(f"<rect x='{snacks_x + 30}' y='{y}' width='7' height='18' rx='1.5' fill='{CORAL}'/>")
    path = "140,440 175,350 175,86 320,86 320,350 600,350 636,386"
    parts.append(f"<polyline points='{path}' fill='none' stroke='#9fb2c0' stroke-width='2.2' stroke-dasharray='7 5' marker-end='url(#arrow)'/>")
    parts.append(f"<text x='330' y='368' font-size='11' fill='#8a9cab' {FONT}>typical shopper path</text>")
    for lane in range(5):
        x = 640 + lane * 58
        parts.append(f"<rect x='{x}' y='388' width='38' height='40' rx='4' fill='#e3eaef'/>")
        parts.append(f"<rect x='{x + 8}' y='380' width='22' height='8' rx='1.5' fill='{NAVY}'/>")
    parts.append(f"<text x='780' y='372' text-anchor='middle' font-size='12' fill='{MUTED}' {FONT}>Checkout lanes</text>")
    parts.append("<rect x='88' y='444' width='104' height='12' fill='white'/>")
    parts.append(f"<text x='140' y='441' text-anchor='middle' font-size='12' fill='{MUTED}' {FONT}>Entrance</text>")
    parts.append(f"<rect x='36' y='376' width='72' height='26' rx='4' fill='{AMBER}'/>")
    parts.append(f"<text x='72' y='393' text-anchor='middle' font-size='10.5' fill='white' {FONT}>lobby screen</text>")
    parts.append(f"<text x='{snacks_x + 15}' y='330' text-anchor='middle' font-size='11' fill='{CORAL}' font-weight='700' {FONT}>shelf-edge screens</text>")
    parts.append(f"<text x='{200 + 5 * 70 + 15}' y='330' text-anchor='middle' font-size='11' fill='{TEAL}' font-weight='700' {FONT}>endcap screen</text>")
    parts += [_marker(600, 308, 1, TEAL),
              _marker(612, 408, 2, NAVY), _marker(snacks_x + 58, 150, 3, CORAL), _marker(124, 389, 4, AMBER)]
    parts.append("</svg>")
    return "".join(parts)


def measurement_flow() -> str:
    steps = [
        ("Design choices", "placement · targeting", "offer · rollout"),
        ("Data it generates", "POS, timestamps,", "traffic, play logs"),
        ("Identification", "holdout stores, waves", "or frequency levels"),
        ("Read-out", "the comparison math", "matched to the design"),
        ("Defensible claim", "lift, response curve,", "net incrementality"),
    ]
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 960 140'>",
             "<defs><marker id='flow' viewBox='0 0 10 10' refX='8' refY='5' markerWidth='8' markerHeight='8' orient='auto'>"
             f"<path d='M0,0 L10,5 L0,10 z' fill='{TEAL}'/></marker></defs>"]
    for index, (title, line1, line2) in enumerate(steps):
        x = 8 + index * 191
        last = index == len(steps) - 1
        parts.append(f"<rect x='{x}' y='20' width='164' height='100' rx='14' fill='{TEAL if last else 'white'}' stroke='{TEAL if last else LINE}' stroke-width='1.6'/>")
        parts.append(f"<text x='{x + 16}' y='44' font-size='10.5' font-weight='700' letter-spacing='1.5' fill='{'#bfeee5' if last else TEAL}' {FONT}>STEP {index + 1}</text>")
        parts.append(f"<text x='{x + 16}' y='68' font-size='15' font-weight='700' fill='{'white' if last else INK}' {FONT}>{title}</text>")
        for offset, line in enumerate((line1, line2)):
            parts.append(f"<text x='{x + 16}' y='{90 + offset * 17}' font-size='12' fill='{'#dff5f1' if last else MUTED}' {FONT}>{line}</text>")
        if not last:
            parts.append(f"<line x1='{x + 168}' y1='70' x2='{x + 188}' y2='70' stroke='{TEAL}' stroke-width='2.2' marker-end='url(#flow)'/>")
    parts.append("</svg>")
    return "".join(parts)


def rollout_patterns() -> str:
    off, launch_line = "#e3eaef", CORAL
    shades = {"low": "#9fd8cf", "medium": "#3fa99e", "high": TEAL}
    panels = [
        ("Simple on/off", ["Holdout vs. campaign stores,", "before vs. after → DiD"]),
        ("Staggered rollout", ["Not-yet-launched stores are", "the controls → staggered DiD"]),
        ("Dose variation", ["Low / medium / high frequency", "traces a response curve"]),
    ]
    rows, cols, cell, gap = 6, 12, 14, 3
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 960 250'>"]
    for panel, (title, caption) in enumerate(panels):
        x0 = 20 + panel * 320
        parts.append(f"<text x='{x0}' y='22' font-size='14' font-weight='700' fill='{INK}' {FONT}>{title}</text>")
        parts.append(f"<text x='{x0}' y='40' font-size='10.5' fill='{MUTED}' {FONT}>rows = stores · columns = weeks</text>")
        for r in range(rows):
            for c in range(cols):
                x, y = x0 + c * (cell + gap), 52 + r * (cell + gap)
                if panel == 0:
                    color = TEAL if r < 3 and c >= 6 else off
                elif panel == 1:
                    start = {0: 3, 1: 3, 2: 6, 3: 6, 4: 9}.get(r, 99)
                    color = TEAL if c >= start else off
                else:
                    level = {0: "low", 1: "low", 2: "medium", 3: "medium", 4: "high"}.get(r)
                    color = shades[level] if level and c >= 6 else off
                parts.append(f"<rect x='{x}' y='{y}' width='{cell}' height='{cell}' rx='2.5' fill='{color}'/>")
        grid_bottom = 52 + rows * (cell + gap)
        if panel in (0, 2):
            lx = x0 + 6 * (cell + gap) - gap / 2
            parts.append(f"<line x1='{lx}' y1='48' x2='{lx}' y2='{grid_bottom}' stroke='{launch_line}' stroke-width='1.6' stroke-dasharray='4 3'/>")
        if panel == 1:
            for r0, c0 in ((0, 3), (2, 6), (4, 9)):
                lx = x0 + c0 * (cell + gap) - gap / 2
                parts.append(f"<line x1='{lx}' y1='{52 + r0 * (cell + gap) - 2}' x2='{lx}' y2='{52 + (r0 + (1 if r0 == 4 else 2)) * (cell + gap)}' stroke='{launch_line}' stroke-width='1.6' stroke-dasharray='4 3'/>")
        for offset, line in enumerate(caption):
            parts.append(f"<text x='{x0}' y='{grid_bottom + 24 + offset * 16}' font-size='12' fill='{INK}' {FONT}>{line}</text>")
    legend_y = 238
    items = [(TEAL, "campaign on"), (shades["low"], "lower dose"), (off, "no campaign")]
    x = 12
    for color, label in items:
        parts.append(f"<rect x='{x}' y='{legend_y - 10}' width='12' height='12' rx='2' fill='{color}'/>")
        parts.append(f"<text x='{x + 18}' y='{legend_y}' font-size='11.5' fill='{MUTED}' {FONT}>{label}</text>")
        x += 30 + len(label) * 6.5
    parts.append(f"<line x1='{x}' y1='{legend_y - 10}' x2='{x}' y2='{legend_y + 2}' stroke='{launch_line}' stroke-width='1.6' stroke-dasharray='4 3'/>")
    parts.append(f"<text x='{x + 8}' y='{legend_y}' font-size='11.5' fill='{MUTED}' {FONT}>launch</text>")
    parts.append("</svg>")
    return "".join(parts)
