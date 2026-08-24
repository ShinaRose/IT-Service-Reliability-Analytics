"""Shared visual system for every Streamlit page (Home + pages/). Extracted from the
original single-page app.py so new pages don't duplicate ~150 lines of CSS. This is
the one place the dark/teal design system lives.

Import order matters for callers: this module imports only `streamlit`, nothing from
`relplatform.config` or anything that transitively reads env vars, so it's always safe
to import before a page has pushed st.secrets into os.environ.
"""
from __future__ import annotations

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
[data-testid="stExpander"] summary { font-family: 'IBM Plex Sans', sans-serif; }

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
