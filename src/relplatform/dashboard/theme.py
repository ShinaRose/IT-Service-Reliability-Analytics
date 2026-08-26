"""Shared visual system for every Streamlit page (Home + pages/). Extracted from the
original single-page app.py so new pages don't duplicate ~150 lines of CSS. This is
the one place the dark/teal design system lives.

Import order matters for callers: this module imports only `streamlit`, `altair`,
`pandas`, and `networkx` (altair is already a transitive streamlit dependency and
networkx is already a project dependency used throughout relplatform.generator/
structural -- no new package either way), nothing from `relplatform.config` or anything
that transitively reads env vars, so it's always safe to import before a page has
pushed st.secrets into os.environ.
"""
from __future__ import annotations

import altair as alt
import networkx as nx
import pandas as pd
import streamlit as st

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
  --rp-bg: #0A0E16;
  --rp-surface: #121926;
  --rp-surface-2: #171F31;
  --rp-border: #232C40;
  --rp-accent: #3FD9C7;
  --rp-accent-dim: #1F5F58;
  --rp-ok: #4ADE94;    --rp-ok-soft: #10261B;
  --rp-warn: #F3B94D;  --rp-warn-soft: #2B2210;
  --rp-bad: #F1706B;   --rp-bad-soft: #2C1414;
  --rp-text: #EDF1F5;
  --rp-text-dim: #9AA7B8;
  --rp-text-faint: #5C6880;

  /* Per-panel signature hues: each major section gets its own color so the page reads
     as distinct zones at a glance, not one accent repeated seven times. Not arbitrary:
     blue for the clustering/data-processing panel, amber for priority (reusing the same
     amber as the "medium" DORA band, since it already means "pay attention" on this page),
     violet for recovery, coral for a forward-looking forecast, rose for the risk/failure
     domain (reusing the "bad" semantic color, since that IS this panel's subject). DORA
     and the AI exec summary bookend the page in the primary teal on purpose. */
  --rp-blue: #5EC8F2;   --rp-blue-soft: #0E2530;
  --rp-violet: #B18CF5; --rp-violet-soft: #211A33;
  --rp-coral: #FF9166;  --rp-coral-soft: #2C1B10;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'IBM Plex Sans', sans-serif; letter-spacing: -0.01em; }

/* Page background: a faint accent glow in the corner, not a loud gradient hero */
[data-testid="stAppViewContainer"], .stApp {
  background:
    radial-gradient(1100px 560px at 12% -8%, rgba(63,217,199,0.055), transparent 60%),
    var(--rp-bg) !important;
}
[data-testid="stHeader"] { background: transparent !important; }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--rp-bg); }
::-webkit-scrollbar-thumb { background: var(--rp-border); border-radius: 6px; }
::-webkit-scrollbar-thumb:hover { background: var(--rp-accent-dim); }

/* ---------------- Eyebrow label (reused everywhere: hero, panels, sidebar) ---------------- */
.eyebrow {
  font-family: 'IBM Plex Mono', monospace; font-size: 11.5px; letter-spacing: 0.09em;
  text-transform: uppercase; color: var(--rp-accent); display: flex; align-items: center; gap: 7px;
  margin-bottom: 8px;
}
.eyebrow .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--rp-accent); flex-shrink: 0; }

/* ---------------- Hero ---------------- */
.hero { padding: 4px 0 30px; }
.hero-title {
  font-family: 'IBM Plex Sans', sans-serif; font-weight: 700; font-size: clamp(26px, 3.2vw, 38px);
  letter-spacing: -0.02em; line-height: 1.18; color: var(--rp-text); margin: 0 0 12px; max-width: 24ch;
}
.hero-meta { font-family: 'IBM Plex Mono', monospace; font-size: 12.5px; color: var(--rp-text-faint); }

/* ---------------- Headline stat grid ---------------- */
.stat-grid {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px;
  background: var(--rp-border); border: 1px solid var(--rp-border); border-radius: 14px;
  overflow: hidden; margin-bottom: 30px;
}
.stat-card { background: var(--rp-surface); padding: 20px 22px; transition: background 0.15s ease; }
.stat-card:hover { background: var(--rp-surface-2); }
.stat-value {
  font-family: 'IBM Plex Mono', monospace; font-size: 25px; font-weight: 600;
  color: var(--rp-accent); letter-spacing: -0.01em; line-height: 1.2;
}
.stat-label { font-size: 12.5px; color: var(--rp-text-faint); margin-top: 6px; }
.stat-sub { font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: var(--rp-text-dim); margin-top: 4px; }
@media (max-width: 900px) {
  .stat-grid { grid-template-columns: repeat(2, 1fr); }
}

/* ---------------- Panels (st.container(border=True)) ----------------
   Streamlit's own bordered-container testid isn't stable across versions (this one
   doesn't emit stVerticalBlockBorderWrapper at all). Anchored instead to our own
   .panel-head marker, which panel_header() renders as the first element inside every
   real panel: `stLayoutWrapper > stVerticalBlock:has(.panel-head)` matches exactly the
   panels that call panel_header() and nothing else (column cells and the page's
   outermost block share the same testid pair but never contain that marker). */
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(.panel-head) {
  background: var(--rp-surface) !important;
  border: 1px solid var(--rp-border) !important;
  border-radius: 14px !important;
  box-shadow: 0 1px 2px rgba(0,0,0,0.25), 0 16px 32px -22px rgba(0,0,0,0.6);
  padding: 20px 22px !important;
  position: relative;
}
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(.panel-head)::before {
  content: ""; position: absolute; top: -1px; left: 18px; right: 18px; height: 2px;
  background: linear-gradient(90deg, var(--rp-accent), transparent 85%);
  border-radius: 2px; opacity: 0.65; pointer-events: none;
}
div[data-testid="stLayoutWrapper"]:has(.panel-head) { margin-bottom: 20px; }

/* Per-panel accent overrides: same :has() anchoring, scoped by an accent-* class on
   the panel's own .panel-head so each panel's top bar + eyebrow pick up its signature
   hue instead of the default teal. */
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(.panel-head.accent-blue)::before   { background: linear-gradient(90deg, var(--rp-blue), transparent 85%); }
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(.panel-head.accent-amber)::before  { background: linear-gradient(90deg, var(--rp-warn), transparent 85%); }
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(.panel-head.accent-violet)::before { background: linear-gradient(90deg, var(--rp-violet), transparent 85%); }
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(.panel-head.accent-coral)::before  { background: linear-gradient(90deg, var(--rp-coral), transparent 85%); }
div[data-testid="stLayoutWrapper"] > div[data-testid="stVerticalBlock"]:has(.panel-head.accent-rose)::before   { background: linear-gradient(90deg, var(--rp-bad), transparent 85%); }

.panel-head.accent-blue .eyebrow   { color: var(--rp-blue); }   .panel-head.accent-blue .eyebrow .dot   { background: var(--rp-blue); }
.panel-head.accent-amber .eyebrow  { color: var(--rp-warn); }   .panel-head.accent-amber .eyebrow .dot  { background: var(--rp-warn); }
.panel-head.accent-violet .eyebrow { color: var(--rp-violet); } .panel-head.accent-violet .eyebrow .dot { background: var(--rp-violet); }
.panel-head.accent-coral .eyebrow  { color: var(--rp-coral); }  .panel-head.accent-coral .eyebrow .dot  { background: var(--rp-coral); }
.panel-head.accent-rose .eyebrow   { color: var(--rp-bad); }    .panel-head.accent-rose .eyebrow .dot   { background: var(--rp-bad); }

.panel-head { display: flex; align-items: flex-end; justify-content: space-between; gap: 14px; margin-bottom: 14px; flex-wrap: wrap; }
.panel-title { font-family: 'IBM Plex Sans', sans-serif; font-weight: 600; font-size: 17px; letter-spacing: -0.01em; color: var(--rp-text); margin: 0; }
.panel-note { font-size: 12.5px; color: var(--rp-text-faint); text-align: right; max-width: 42ch; }

/* ---------------- Metrics ---------------- */
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 1.5rem; color: var(--rp-text); }
[data-testid="stMetricLabel"] { color: var(--rp-text-faint) !important; font-size: 12.5px; }
[data-testid="stMetricDelta"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 12.5px; }

/* ---------------- Band pills ---------------- */
.band-pill {
  display: inline-flex; align-items: center; gap: 5px;
  font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 600;
  letter-spacing: 0.03em; text-transform: uppercase;
  padding: 3px 10px; border-radius: 999px; margin-top: 4px;
}
.band-elite  { background: var(--rp-ok-soft);   color: var(--rp-ok); }
.band-high   { background: rgba(63,217,199,0.12); color: var(--rp-accent); }
.band-medium { background: var(--rp-warn-soft); color: var(--rp-warn); }
.band-low    { background: var(--rp-bad-soft);  color: var(--rp-bad); }

/* Traffic-light pills (SLO ship/freeze): reuse the same visual language as band-pill */
.light-pill {
  display: inline-flex; align-items: center; gap: 6px;
  font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 600;
  letter-spacing: 0.03em; text-transform: uppercase;
  padding: 4px 12px; border-radius: 999px;
}
.light-green  { background: var(--rp-ok-soft);   color: var(--rp-ok); }
.light-yellow { background: var(--rp-warn-soft); color: var(--rp-warn); }
.light-red    { background: var(--rp-bad-soft);  color: var(--rp-bad); }

/* ---------------- Sidebar ---------------- */
section[data-testid="stSidebar"] { background: var(--rp-surface); border-right: 1px solid var(--rp-border); }
section[data-testid="stSidebar"] h1 { font-size: 19px !important; margin-top: 0; }
section[data-testid="stSidebar"] hr { border-color: var(--rp-border); margin: 16px 0; }
.control-label {
  font-family: 'IBM Plex Mono', monospace; font-size: 10.5px; letter-spacing: 0.07em;
  text-transform: uppercase; color: var(--rp-text-dim); margin: 2px 0 0;
}

/* ---------------- Buttons ---------------- */
div[data-testid="stButton"] > button {
  border: 1px solid var(--rp-accent); color: var(--rp-accent); background: transparent;
  border-radius: 8px; font-weight: 500; transition: background-color 0.15s ease, box-shadow 0.15s ease, color 0.15s ease;
}
div[data-testid="stButton"] > button:hover {
  border-color: var(--rp-accent); color: #05110F; background-color: var(--rp-accent);
  box-shadow: 0 6px 20px -6px rgba(63,217,199,0.45);
}
div[data-testid="stButton"] > button:focus:not(:active) { border-color: var(--rp-accent); color: var(--rp-accent); }

/* ---------------- Multiselect chips ---------------- */
[data-testid="stMultiSelectTagsContainer"] span[data-tag] {
  background-color: rgba(63,217,199,0.14) !important; border: 1px solid rgba(63,217,199,0.4) !important;
  border-radius: 6px !important;
}

/* ---------------- Dataframe / expander shells ---------------- */
[data-testid="stDataFrame"], [data-testid="stExpander"] {
  border: 1px solid var(--rp-border) !important; border-radius: 10px !important; overflow: hidden;
}
[data-testid="stExpander"] summary {
  font-family: 'IBM Plex Sans', sans-serif; background: var(--rp-surface) !important; color: var(--rp-text) !important;
}

/* ---------------- Selects / sliders ----------------
   config.toml sets base="dark" as the default, and the elements above are already
   force-styled regardless of Streamlit's active theme -- but a viewer who manually
   flips Streamlit's own light/dark toggle (hamburger menu > Settings) would otherwise
   still see these specific native widgets flip to light-mode colors while everything
   else on the page stays forced dark. Pinned the same way as the rest of this file. */
[data-testid="stSelectbox"] > div > div, [data-testid="stMultiSelect"] > div > div {
  background-color: var(--rp-surface) !important; border-color: var(--rp-border) !important; color: var(--rp-text) !important;
}
[data-baseweb="select"] * { color: var(--rp-text) !important; }
[data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stSlider"] [data-testid="stTickBarMax"] {
  color: var(--rp-text-faint) !important;
}
div[data-testid="stSlider"] > div > div > div > div { background-color: var(--rp-accent) !important; }

@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; }
}
</style>
"""


def inject() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def band_pill(band: str) -> str:
    return f'<span class="band-pill band-{band}">{band}</span>'


def light_pill(light: str) -> str:
    return f'<span class="light-pill light-{light}">{light}</span>'


def eyebrow_html(text: str) -> str:
    return f'<div class="eyebrow"><span class="dot"></span>{text}</div>'


def panel_header(eyebrow_text: str, title: str, note: str = "", accent: str = "") -> None:
    """accent: '' (teal, the default/brand color) or one of blue/amber/violet/coral/rose.
    See the accent-* CSS rules for what each panel's signature hue is used for."""
    note_html = f'<div class="panel-note">{note}</div>' if note else ""
    accent_cls = f" accent-{accent}" if accent else ""
    st.markdown(
        f'<div class="panel-head{accent_cls}"><div>{eyebrow_html(eyebrow_text)}'
        f'<h3 class="panel-title">{title}</h3></div>{note_html}</div>',
        unsafe_allow_html=True,
    )


def stat_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="stat-sub">{sub}</div>' if sub else ""
    return f'<div class="stat-card"><div class="stat-value">{value}</div><div class="stat-label">{label}</div>{sub_html}</div>'


SOURCE_LABELS = {"synthetic": "SYNTHETIC DATA", "github": "GITHUB-DERIVED", "uploaded": "USER-UPLOADED"}
SOURCE_COLORS = {"synthetic": "var(--rp-accent)", "github": "var(--rp-blue)", "uploaded": "var(--rp-violet)"}


def source_badge(source: str, detail: str = "") -> None:
    """Persistent, unmissable label for which of the three data sources (synthetic /
    github / uploaded) the numbers on screen right now came from. Every metric on a
    page using this must be truthfully attributable to whichever source is shown."""
    label = SOURCE_LABELS.get(source, source.upper())
    color = SOURCE_COLORS.get(source, "var(--rp-accent)")
    detail_html = f'<span style="color: var(--rp-text-faint); font-weight: 400;">&middot; {detail}</span>' if detail else ""
    st.markdown(
        f'<div style="display:inline-flex; align-items:center; gap:8px; font-family:\'IBM Plex Mono\',monospace; '
        f'font-size:12px; font-weight:600; letter-spacing:0.05em; padding:6px 14px; border-radius:999px; '
        f'border:1px solid {color}; color:{color}; margin-bottom:16px;">'
        f'<span style="width:7px;height:7px;border-radius:50%;background:{color};"></span>{label} {detail_html}</div>',
        unsafe_allow_html=True,
    )


def assumption_note(text: str) -> None:
    """A visible callout for a number that rests on a stated assumption, e.g. 'downtime
    = full incident duration', 'euro/minute cost is a config estimate, not measured'.
    Every new phase's UI is expected to use this next to any such number rather than
    letting the assumption live only in a docstring."""
    st.markdown(
        f'<div style="border-left: 3px solid var(--rp-warn); background: var(--rp-warn-soft); '
        f'border-radius: 0 8px 8px 0; padding: 10px 14px; margin: 8px 0;">'
        f'<div style="font-family: \'IBM Plex Mono\', monospace; font-size: 10.5px; letter-spacing: 0.07em; '
        f'text-transform: uppercase; color: var(--rp-warn); margin-bottom: 3px;">Assumption</div>'
        f'<div style="font-size: 12.5px; color: var(--rp-text-dim);">{text}</div></div>',
        unsafe_allow_html=True,
    )


# ---------------- Charts ----------------
# Hand-built with Altair instead of st.bar_chart/st.line_chart: Streamlit's native
# chart shorthand binds mouse-wheel zoom/pan for exploring the x-axis, and a two-finger
# trackpad scroll gesture is exactly the kind of input that triggers it -- the chart
# zooms instead of the page scrolling underneath it, with no parameter on st.bar_chart/
# st.line_chart to turn that off. These helpers render the same visual result with no
# .interactive() binding at all, so scrolling over a chart always just scrolls the page.
_CHART_SERIES_COLORS = ["#3FD9C7", "#5EC8F2", "#B18CF5", "#FF9166", "#4ADE94", "#F3B94D", "#F1706B", "#EDF1F5"]


def _themed(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#9AA7B8", titleColor="#9AA7B8", gridColor="#232C40",
            domainColor="#232C40", tickColor="#232C40",
            labelFont="IBM Plex Mono", titleFont="IBM Plex Mono", labelFontSize=11,
        )
        .configure_legend(labelColor="#9AA7B8", titleColor="#9AA7B8", labelFont="IBM Plex Mono", labelFontSize=11)
    )


def bar_chart(data, color="#3FD9C7", height: int = 260, empty_message: str = "No data to chart.") -> None:
    """Static bar chart, no scroll/zoom/pan capture. `data`: a Series (index=category)
    for one series, or a DataFrame (index=category, columns=series) for grouped bars.
    Same input shape st.bar_chart accepts. `empty_message`: shown instead of a chart
    when `data` is empty -- callers with a specific reason ("didn't clear the
    threshold" vs. generic "no data") should pass one rather than leaving the default."""
    if not len(data):
        st.caption(empty_message)
        return

    if isinstance(data, pd.Series):
        cat_col = data.index.name or "category"
        df = data.rename("value").reset_index()
        chart = (
            alt.Chart(df).mark_bar(color=color)
            .encode(x=alt.X(f"{cat_col}:N", sort=None, title=None), y=alt.Y("value:Q", title=None),
                    tooltip=[cat_col, "value"])
        )
    else:
        cat_col = data.index.name or "category"
        df = data.reset_index().melt(id_vars=cat_col, var_name="series", value_name="value")
        colors = color if isinstance(color, list) else _CHART_SERIES_COLORS
        chart = (
            alt.Chart(df).mark_bar()
            .encode(
                x=alt.X(f"{cat_col}:N", sort=None, title=None),
                xOffset="series:N",
                y=alt.Y("value:Q", title=None),
                color=alt.Color("series:N", scale=alt.Scale(range=colors), legend=alt.Legend(title=None)),
                tooltip=[cat_col, "series", "value"],
            )
        )
    st.altair_chart(_themed(chart).properties(height=height), width="stretch")


def render_dependency_graph_svg(g: nx.DiGraph, report: list[dict], width: int = 720, height: int = 420) -> str:
    """Node-link diagram of the service dependency graph. Node radius = blast radius
    (services affected if this one fails), node color = criticality (PageRank) --
    both real numbers from relplatform.structural.graph.structural_report, not
    decorative sizing. Positions come from nx.spring_layout with a fixed seed, so the
    layout is stable across reruns instead of jittering on every page load.

    Hand-built inline SVG, not st.graphviz_chart: that needs a system-level Graphviz
    binary installed, a real deployment risk on Streamlit Community Cloud's free tier
    (no apt-get access without a packages.txt this project doesn't otherwise need).
    Colors are computed as literal hex/rgb, the same pattern the Evaluation page's
    confusion-matrix heatmap already uses, not CSS var() references -- keeps this
    function usable standalone without depending on theme.inject() having run first.
    """
    by_service = {r["service"]: r for r in report}
    pos = nx.spring_layout(g, seed=42, k=1.4, iterations=200)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    margin = 60

    def scale(xy):
        x, y = xy
        sx = margin + (x - x_lo) / max(1e-6, x_hi - x_lo) * (width - 2 * margin)
        sy = margin + (y - y_lo) / max(1e-6, y_hi - y_lo) * (height - 2 * margin)
        return sx, sy

    max_blast = max((r["blast_radius_count"] for r in report), default=0) or 1
    max_crit = max((r["criticality_pagerank"] for r in report), default=0) or 1e-6

    def node_radius(svc: str) -> float:
        return 14 + (by_service.get(svc, {}).get("blast_radius_count", 0) / max_blast) * 18

    def node_color(svc: str) -> str:
        t = min(1.0, by_service.get(svc, {}).get("criticality_pagerank", 0) / max_crit)
        r = round(23 + t * (94 - 23))
        gr = round(31 + t * (200 - 31))
        b = round(49 + t * (242 - 49))
        return f"rgb({r},{gr},{b})"

    edges_svg = []
    for u, v in g.edges():
        x1, y1 = scale(pos[u])
        x2, y2 = scale(pos[v])
        dx, dy = x2 - x1, y2 - y1
        dist = max(1e-6, (dx**2 + dy**2) ** 0.5)
        r_target = node_radius(v) + 5  # stop the line short so the arrowhead clears the target circle
        x2s, y2s = x2 - dx / dist * r_target, y2 - dy / dist * r_target
        edges_svg.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2s:.1f}" y2="{y2s:.1f}" '
            f'stroke="#3A4560" stroke-width="1.4" marker-end="url(#rp-graph-arrow)" opacity="0.8"/>'
        )

    nodes_svg = []
    for svc in g.nodes():
        x, y = scale(pos[svc])
        radius = node_radius(svc)
        label = svc.removesuffix("-service")
        nodes_svg.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{node_color(svc)}" stroke="#0A0E16" stroke-width="2"/>'
            f'<text x="{x:.1f}" y="{y + radius + 16:.1f}" text-anchor="middle" '
            f'font-family="IBM Plex Mono, monospace" font-size="11" fill="#9AA7B8">{label}</text>'
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        f'<defs><marker id="rp-graph-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#3A4560"/></marker></defs>'
        f'{"".join(edges_svg)}{"".join(nodes_svg)}</svg>'
    )


def line_chart(data: pd.DataFrame, height: int = 280, empty_message: str = "No data to chart.") -> None:
    """Static multi-series line chart, no scroll/zoom/pan capture. `data`: a DataFrame
    with the x-axis as its index and one column per series. Same input shape
    st.line_chart accepts. `empty_message`: see bar_chart's docstring."""
    if not len(data):
        st.caption(empty_message)
        return

    idx_name = data.index.name or "x"
    df = data.reset_index().melt(id_vars=idx_name, var_name="series", value_name="value")
    chart = (
        alt.Chart(df).mark_line()
        .encode(
            x=alt.X(f"{idx_name}:Q", title=idx_name),
            y=alt.Y("value:Q", title=None),
            color=alt.Color("series:N", scale=alt.Scale(range=_CHART_SERIES_COLORS), legend=alt.Legend(title=None)),
            tooltip=[idx_name, "series", "value"],
        )
    )
    st.altair_chart(_themed(chart).properties(height=height), width="stretch")
