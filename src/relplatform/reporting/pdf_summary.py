"""One-page PDF executive summary, laid out from the same report dict every dashboard
page already reads (relplatform.pipeline.compute_full_report) -- no new computation
here, only formatting. Uses fpdf2 specifically because it's pure Python with no
system-level dependencies (no Pango/Cairo the way HTML-to-PDF renderers need); a
library that needs native libraries to install would risk breaking the one-command
deploy this whole platform is built around, and would add real memory footprint on
Streamlit Community Cloud's capped free tier.
"""
from __future__ import annotations

from datetime import datetime, timezone

from fpdf import FPDF

PAGE_MARGIN_MM = 15
MAX_NARRATIVE_CHARS = 2200  # keeps the page to one, even with a narrative attached


def _band_label(band: str) -> str:
    return str(band).upper()


def build_exec_summary_pdf(report: dict, seed: int, period_label: str = "current period", ai_narrative: str | None = None) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margin(PAGE_MARGIN_MM)
    pdf.set_auto_page_break(auto=True, margin=PAGE_MARGIN_MM)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Reliability Analytics -- Executive Summary", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 9)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pdf.set_text_color(110, 110, 110)
    pdf.cell(0, 6, f"Generated {generated}  |  Period: {period_label}  |  Synthetic data, seed={seed}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    # ---- DORA metrics ----
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "DORA Metrics", new_x="LMARGIN", new_y="NEXT")
    dora = report["dora_metrics"]
    dora_rows = [
        ("Deployment Frequency", f"{dora['deployment_frequency']['value_per_day']}/day", dora["deployment_frequency"]["band"]),
        ("Lead Time for Changes", f"{dora['lead_time_for_changes']['median_hours']}h median", dora["lead_time_for_changes"]["band"]),
        ("Change Failure Rate", f"{dora['change_failure_rate']['value_pct']}%", dora["change_failure_rate"]["band"]),
        ("Time to Restore", f"{dora['time_to_restore']['median_hours']}h median", dora["time_to_restore"]["band"]),
    ]
    pdf.set_font("Helvetica", "", 10)
    with pdf.table(col_widths=(70, 55, 55), text_align=("LEFT", "LEFT", "LEFT")) as table:
        header = table.row()
        for h in ("Metric", "Value", "Band"):
            header.cell(h)
        for metric, value, band in dora_rows:
            row = table.row()
            row.cell(metric)
            row.cell(value)
            row.cell(_band_label(band))
    pdf.ln(2)

    # ---- Top risk services ----
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Top Risk Services", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    risk_rows = sorted(report.get("risk_scores", []), key=lambda r: r["risk_score"], reverse=True)[:5]
    if risk_rows:
        with pdf.table(col_widths=(60, 40, 40, 40), text_align=("LEFT", "RIGHT", "RIGHT", "RIGHT")) as table:
            header = table.row()
            for h in ("Service", "Risk Score", "Incidents/mo", "MTTR p90 (min)"):
                header.cell(h)
            for r in risk_rows:
                row = table.row()
                row.cell(r["service"])
                row.cell(f"{r['risk_score']:.1f}")
                row.cell(f"{r['incidents_per_month']:.1f}")
                row.cell(f"{r['mttr_p90_minutes']:.0f}")
    else:
        pdf.cell(0, 6, "No risk scores available.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # ---- Operational health ----
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, "Operational Health", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    nr = report.get("noise_reduction", {})
    if nr:
        pdf.cell(
            0, 6,
            f"Alert noise reduction: {nr['noise_reduction_rate']:.1%}  "
            f"({nr['n_alerts']:,} raw alerts -> {nr['n_distinct_clusters']:,} distinct clusters)",
            new_x="LMARGIN", new_y="NEXT",
        )

    breaches = [c for c in report.get("capacity_forecasts", []) if c.get("status") == "breach_projected"]
    if breaches:
        names = ", ".join(c["service"] for c in breaches)
        pdf.cell(0, 6, f"Capacity breach projected: {names}", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.cell(0, 6, "No statistically significant capacity breach projected for any service.", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # ---- AI narrative (optional) ----
    if ai_narrative:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, "Narrative Summary (AI-generated)", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        text = ai_narrative if len(ai_narrative) <= MAX_NARRATIVE_CHARS else ai_narrative[:MAX_NARRATIVE_CHARS] + "... [truncated]"
        pdf.multi_cell(0, 5, text)

    return bytes(pdf.output())
