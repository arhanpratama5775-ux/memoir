"""HTML exporter for the memoir project.

Generates a beautiful, standalone HTML file from a ``Memoir`` instance.
The output is a self-contained page with embedded CSS, responsive layout,
dark-mode support, collapsible sections, sidebar navigation, and
print-friendly styles — resembling a professional documentation site.
"""

from __future__ import annotations

import html as html_lib
import logging
import os
from datetime import datetime
from typing import Dict, List

from memoir.models.chapter import Chapter, ChapterType, Memoir
from memoir.models.commit_data import CommitData, GitStats
from memoir.models.forecast import Forecast, RiskLevel
from memoir.models.pattern import Pattern, PatternType, Severity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour mappings
# ---------------------------------------------------------------------------

_SEVERITY_COLORS: Dict[Severity, str] = {
    Severity.LOW: "#22c55e",
    Severity.MEDIUM: "#eab308",
    Severity.HIGH: "#f97316",
    Severity.CRITICAL: "#ef4444",
}

_RISK_COLORS: Dict[RiskLevel, str] = {
    RiskLevel.LOW: "#22c55e",
    RiskLevel.MODERATE: "#eab308",
    RiskLevel.HIGH: "#f97316",
    RiskLevel.CRITICAL: "#ef4444",
}

_SEVERITY_EMOJI: Dict[Severity, str] = {
    Severity.LOW: "🟢",
    Severity.MEDIUM: "🟡",
    Severity.HIGH: "🟠",
    Severity.CRITICAL: "🔴",
}

_RISK_EMOJI: Dict[RiskLevel, str] = {
    RiskLevel.LOW: "🟢",
    RiskLevel.MODERATE: "🟡",
    RiskLevel.HIGH: "🟠",
    RiskLevel.CRITICAL: "🔴",
}

_PATTERN_TYPE_LABEL: Dict[PatternType, str] = {
    PatternType.RECURRING_FIX: "Recurring Fix",
    PatternType.TECHNICAL_DEBT: "Technical Debt",
    PatternType.LEARNING_CURVE: "Learning Curve",
    PatternType.BURNOUT_INDICATOR: "Burnout Indicator",
    PatternType.CODE_OWNERSHIP: "Code Ownership",
    PatternType.ANTI_PATTERN: "Anti-Pattern",
    PatternType.IRREGULAR_HOURS: "Irregular Hours",
    PatternType.HIGH_CHURN: "High Churn",
}

_CHAPTER_TYPE_ICON: Dict[ChapterType, str] = {
    ChapterType.PROLOGUE: "📖",
    ChapterType.PATTERN: "🔍",
    ChapterType.MILESTONE: "🏆",
    ChapterType.CRISIS: "🔥",
    ChapterType.CURRENT_STATE: "📍",
    ChapterType.FORECAST: "🔮",
}

# ---------------------------------------------------------------------------
# Embedded stylesheet
# ---------------------------------------------------------------------------

_CSS = r"""
/* ===== Reset & Base ===== */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #ffffff;
  --bg-secondary: #f8fafc;
  --bg-card: #ffffff;
  --text: #1e293b;
  --text-secondary: #64748b;
  --text-muted: #94a3b8;
  --border: #e2e8f0;
  --border-light: #f1f5f9;
  --accent: #0ea5e9;
  --accent-hover: #0284c7;
  --sidebar-bg: #f8fafc;
  --sidebar-width: 280px;
  --code-bg: #f1f5f9;
  --shadow-sm: 0 1px 2px rgba(0,0,0,.05);
  --shadow-md: 0 4px 6px -1px rgba(0,0,0,.1), 0 2px 4px -2px rgba(0,0,0,.1);
  --radius: 8px;
  --radius-lg: 12px;
  --transition: 0.2s ease;
  --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f172a;
    --bg-secondary: #1e293b;
    --bg-card: #1e293b;
    --text: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --border: #334155;
    --border-light: #1e293b;
    --accent: #38bdf8;
    --accent-hover: #7dd3fc;
    --sidebar-bg: #1e293b;
    --code-bg: #334155;
    --shadow-sm: 0 1px 2px rgba(0,0,0,.2);
    --shadow-md: 0 4px 6px -1px rgba(0,0,0,.3), 0 2px 4px -2px rgba(0,0,0,.2);
  }
}

html { scroll-behavior: smooth; font-size: 16px; }

body {
  font-family: var(--font-sans);
  color: var(--text);
  background: var(--bg);
  line-height: 1.7;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* ===== Layout ===== */
.layout {
  display: flex;
  min-height: 100vh;
}

/* ===== Sidebar ===== */
.sidebar {
  position: fixed;
  top: 0; left: 0; bottom: 0;
  width: var(--sidebar-width);
  background: var(--sidebar-bg);
  border-right: 1px solid var(--border);
  padding: 2rem 1.25rem;
  overflow-y: auto;
  z-index: 100;
  transition: transform var(--transition);
}

.sidebar-title {
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 0.25rem;
}

.sidebar-subtitle {
  font-size: 0.8rem;
  color: var(--text-muted);
  margin-bottom: 1.5rem;
}

.sidebar nav a {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.45rem 0.75rem;
  margin-bottom: 2px;
  font-size: 0.85rem;
  color: var(--text-secondary);
  text-decoration: none;
  border-radius: var(--radius);
  transition: background var(--transition), color var(--transition);
}

.sidebar nav a:hover,
.sidebar nav a.active {
  background: var(--border-light);
  color: var(--text);
}

.sidebar .sidebar-section-label {
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
  margin-top: 1.25rem;
  margin-bottom: 0.5rem;
  padding-left: 0.75rem;
}

/* ===== Main content ===== */
.main {
  flex: 1;
  margin-left: var(--sidebar-width);
  padding: 3rem 3.5rem 6rem;
  max-width: 920px;
}

/* ===== Cover / Hero ===== */
.hero {
  text-align: center;
  padding: 3rem 0 2.5rem;
  margin-bottom: 2rem;
  border-bottom: 1px solid var(--border);
}

.hero-icon { font-size: 3.5rem; margin-bottom: 1rem; }

.hero h1 {
  font-size: 2.5rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  margin-bottom: 0.35rem;
}

.hero h2 {
  font-size: 1.15rem;
  font-weight: 400;
  color: var(--text-secondary);
  margin-bottom: 1rem;
}

.hero .meta {
  font-size: 0.85rem;
  color: var(--text-muted);
}

/* Health gauge */
.health-gauge {
  display: inline-flex;
  flex-direction: column;
  align-items: center;
  margin-top: 1.5rem;
  padding: 1.25rem 2rem;
  background: var(--bg-secondary);
  border-radius: var(--radius-lg);
  border: 1px solid var(--border);
}

.health-gauge .label {
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
  margin-bottom: 0.5rem;
}

.health-gauge svg { margin-bottom: 0.35rem; }

.health-gauge .score-text {
  font-size: 1.5rem;
  font-weight: 800;
}

.health-gauge .score-label {
  font-size: 0.8rem;
  color: var(--text-secondary);
}

/* ===== Chapter ===== */
.chapter {
  margin-bottom: 3rem;
  scroll-margin-top: 2rem;
}

.chapter-header {
  margin-bottom: 1.25rem;
}

.chapter-header h2 {
  font-size: 1.65rem;
  font-weight: 700;
  letter-spacing: -0.01em;
  margin-bottom: 0.15rem;
}

.chapter-header .subtitle {
  font-size: 0.95rem;
  color: var(--text-secondary);
  font-style: italic;
}

.chapter-header .period {
  font-size: 0.8rem;
  color: var(--text-muted);
  margin-top: 0.35rem;
}

.chapter-badge {
  display: inline-block;
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  padding: 0.2rem 0.6rem;
  border-radius: 9999px;
  background: var(--accent);
  color: #fff;
  margin-bottom: 0.75rem;
}

.narrative {
  font-size: 0.95rem;
  line-height: 1.8;
  color: var(--text);
  margin-bottom: 1.5rem;
}

/* ===== Pattern callout ===== */
.pattern-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-left: 4px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.25rem;
  margin-bottom: 1rem;
  box-shadow: var(--shadow-sm);
}

.pattern-card.severity-low    { border-left-color: #22c55e; }
.pattern-card.severity-medium { border-left-color: #eab308; }
.pattern-card.severity-high   { border-left-color: #f97316; }
.pattern-card.severity-critical { border-left-color: #ef4444; }

.pattern-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  cursor: pointer;
  user-select: none;
}

.pattern-title {
  font-size: 0.9rem;
  font-weight: 600;
}

.pattern-severity-badge {
  font-size: 0.65rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  padding: 0.15rem 0.5rem;
  border-radius: 9999px;
  color: #fff;
}

.pattern-body {
  margin-top: 0.75rem;
  font-size: 0.85rem;
  color: var(--text-secondary);
  line-height: 1.65;
  overflow: hidden;
  transition: max-height 0.3s ease, opacity 0.2s ease;
}

.pattern-body.collapsed { max-height: 0; opacity: 0; margin-top: 0; }
.pattern-body.expanded  { max-height: 2000px; opacity: 1; }

.pattern-meta {
  margin-top: 0.65rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  font-size: 0.78rem;
}

.pattern-meta .tag {
  background: var(--code-bg);
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  font-family: var(--font-mono);
  font-size: 0.72rem;
}

.pattern-recommendation {
  margin-top: 0.65rem;
  padding: 0.6rem 0.85rem;
  background: var(--bg-secondary);
  border-radius: var(--radius);
  font-size: 0.82rem;
  font-style: italic;
  color: var(--text-secondary);
}

/* ===== Forecast card ===== */
.forecast-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.25rem 1.5rem;
  margin-bottom: 1rem;
  box-shadow: var(--shadow-sm);
}

.forecast-card.risk-low      { border-left: 4px solid #22c55e; }
.forecast-card.risk-moderate { border-left: 4px solid #eab308; }
.forecast-card.risk-high     { border-left: 4px solid #f97316; }
.forecast-card.risk-critical { border-left: 4px solid #ef4444; }

.forecast-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  cursor: pointer;
  user-select: none;
}

.forecast-title {
  font-size: 0.95rem;
  font-weight: 600;
}

.risk-badge {
  font-size: 0.65rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  padding: 0.15rem 0.5rem;
  border-radius: 9999px;
  color: #fff;
}

.forecast-body {
  margin-top: 0.75rem;
  font-size: 0.85rem;
  color: var(--text-secondary);
  line-height: 1.65;
  overflow: hidden;
  transition: max-height 0.3s ease, opacity 0.2s ease;
}

.forecast-body.collapsed { max-height: 0; opacity: 0; margin-top: 0; }
.forecast-body.expanded  { max-height: 2000px; opacity: 1; }

.forecast-indicators {
  margin-top: 0.75rem;
}

.forecast-indicator {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0.35rem 0;
  border-bottom: 1px solid var(--border-light);
  font-size: 0.82rem;
}

.forecast-indicator:last-child { border-bottom: none; }

.forecast-indicator .name { color: var(--text-secondary); }
.forecast-indicator .value { font-family: var(--font-mono); font-size: 0.78rem; }
.forecast-indicator .trend-up { color: #ef4444; }
.forecast-indicator .trend-stable { color: #eab308; }
.forecast-indicator .trend-down { color: #22c55e; }

.forecast-recommendation {
  margin-top: 0.75rem;
  padding: 0.6rem 0.85rem;
  background: var(--bg-secondary);
  border-radius: var(--radius);
  font-size: 0.82rem;
  font-style: italic;
  color: var(--text-secondary);
}

/* ===== Commit timeline ===== */
.commit-timeline {
  position: relative;
  padding-left: 1.5rem;
  margin-bottom: 1.5rem;
}

.commit-timeline::before {
  content: '';
  position: absolute;
  left: 0.4rem;
  top: 0.25rem;
  bottom: 0.25rem;
  width: 2px;
  background: var(--border);
}

.commit-item {
  position: relative;
  padding: 0.35rem 0 0.35rem 0.5rem;
  font-size: 0.82rem;
}

.commit-item::before {
  content: '';
  position: absolute;
  left: -1.25rem;
  top: 0.65rem;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--accent);
  border: 2px solid var(--bg);
}

.commit-hash {
  font-family: var(--font-mono);
  font-size: 0.75rem;
  background: var(--code-bg);
  padding: 0.1rem 0.35rem;
  border-radius: 3px;
  color: var(--accent);
}

.commit-date { color: var(--text-muted); font-size: 0.78rem; }
.commit-msg { color: var(--text-secondary); }
.commit-net { font-family: var(--font-mono); font-size: 0.75rem; }
.commit-net.positive { color: #22c55e; }
.commit-net.negative { color: #ef4444; }
.commit-net.neutral  { color: var(--text-muted); }

/* ===== Stats table ===== */
.stats-section { margin-top: 3rem; }

.stats-section h2 {
  font-size: 1.4rem;
  font-weight: 700;
  margin-bottom: 1rem;
}

.stats-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.85rem;
}

.stats-table th,
.stats-table td {
  text-align: left;
  padding: 0.55rem 0.85rem;
  border-bottom: 1px solid var(--border-light);
}

.stats-table th {
  font-weight: 600;
  color: var(--text-secondary);
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.stats-table td { color: var(--text); }

.stats-table tr:hover td { background: var(--bg-secondary); }

/* ===== Chapter stats (inline table) ===== */
.chapter-stats {
  margin-top: 1.25rem;
  padding: 1rem;
  background: var(--bg-secondary);
  border-radius: var(--radius);
  border: 1px solid var(--border-light);
}

.chapter-stats h4 {
  font-size: 0.8rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--text-muted);
  margin-bottom: 0.65rem;
}

.chapter-stats table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.82rem;
}

.chapter-stats td {
  padding: 0.3rem 0.5rem;
  border-bottom: 1px solid var(--border-light);
}

.chapter-stats td:first-child { color: var(--text-secondary); }
.chapter-stats td:last-child  { font-family: var(--font-mono); font-size: 0.8rem; }

/* ===== Collapsible toggle ===== */
.toggle-icon {
  display: inline-block;
  transition: transform 0.2s ease;
  font-size: 0.7rem;
  color: var(--text-muted);
}

.toggle-icon.open { transform: rotate(90deg); }

/* ===== Separator ===== */
.separator {
  border: none;
  border-top: 1px solid var(--border);
  margin: 2.5rem 0;
}

/* ===== Mobile hamburger ===== */
.menu-toggle {
  display: none;
  position: fixed;
  top: 1rem; left: 1rem;
  z-index: 200;
  width: 40px; height: 40px;
  border-radius: var(--radius);
  border: 1px solid var(--border);
  background: var(--bg-card);
  box-shadow: var(--shadow-md);
  cursor: pointer;
  font-size: 1.2rem;
  line-height: 1;
  color: var(--text);
  align-items: center;
  justify-content: center;
}

/* ===== Responsive ===== */
@media (max-width: 1024px) {
  .sidebar { transform: translateX(-100%); }
  .sidebar.open { transform: translateX(0); box-shadow: var(--shadow-md); }
  .main { margin-left: 0; padding: 2rem 1.5rem 4rem; }
  .menu-toggle { display: flex; }
}

@media (max-width: 640px) {
  .hero h1 { font-size: 1.75rem; }
  .main { padding: 1.25rem 1rem 3rem; }
  .health-gauge { padding: 1rem 1.25rem; }
  .chapter-header h2 { font-size: 1.35rem; }
}

/* ===== Print ===== */
@media print {
  .sidebar, .menu-toggle { display: none !important; }
  .main { margin-left: 0; padding: 1rem; }
  .pattern-body.collapsed,
  .forecast-body.collapsed { max-height: none; opacity: 1; margin-top: 0.75rem; }
  body { font-size: 11pt; }
  .hero { page-break-after: always; }
  .chapter { page-break-inside: avoid; }
}
"""


class HtmlExporter:
    """Export a ``Memoir`` to a standalone HTML file with embedded CSS.

    The output is a fully self-contained page that looks like a
    professional documentation site, complete with:

    - Sidebar chapter navigation
    - Responsive layout (mobile hamburger menu)
    - Dark-mode support (via ``prefers-color-scheme``)
    - Colour-coded severity and risk indicators
    - Collapsible pattern and forecast sections
    - A coloured SVG health-score gauge
    - Print-friendly styles

    Example::

        exporter = HtmlExporter()
        path = exporter.export(memoir, "output/memoir.html")
    """

    def export(self, memoir: Memoir, output_path: str) -> str:
        """Export memoir as a standalone HTML file.

        Args:
            memoir: The memoir document to export.
            output_path: Destination file path.

        Returns:
            The absolute path of the written file.

        Raises:
            OSError: If the file cannot be written.
        """
        logger.info("Exporting memoir to HTML: %s", output_path)

        page = self._build_page(memoir)

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(page)

        abs_path = os.path.abspath(output_path)
        logger.info("HTML export complete: %s", abs_path)
        return abs_path

    # ==================================================================
    # Page composition
    # ==================================================================

    def _build_page(self, memoir: Memoir) -> str:
        """Assemble the full HTML page."""
        title = html_lib.escape(f"Developer Memoir — {memoir.repo_name}")
        sidebar = self._build_sidebar(memoir)
        main = self._build_main(memoir)
        script = self._build_script()

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title}</title>
  <style>{_CSS}</style>
</head>
<body>
<button class="menu-toggle" id="menuToggle" aria-label="Toggle navigation">&#9776;</button>
<div class="layout">
  {sidebar}
  {main}
</div>
{script}
</body>
</html>"""

    # ==================================================================
    # Sidebar
    # ==================================================================

    def _build_sidebar(self, memoir: Memoir) -> str:
        """Build the sidebar navigation HTML."""
        repo = html_lib.escape(memoir.repo_name)
        date = memoir.generated_at.strftime("%d %b %Y")

        nav_items: list[str] = []

        # Chapter links
        nav_items.append('<div class="sidebar-section-label">Chapters</div>')
        for chapter in memoir.chapters:
            icon = _CHAPTER_TYPE_ICON.get(chapter.chapter_type, "📄")
            ch_title = html_lib.escape(chapter.title)
            anchor = _slugify(chapter.title)
            nav_items.append(
                f'<a href="#{anchor}">{icon} {ch_title}</a>'
            )

        # Forecast link
        if memoir.forecasts:
            nav_items.append('<div class="sidebar-section-label">Analysis</div>')
            nav_items.append(
                f'<a href="#forecasts">🔮 Risk Forecasts</a>'
            )

        # Stats link
        nav_items.append(
            f'<a href="#statistics">📊 Statistics</a>'
        )

        nav_html = "\n".join(nav_items)

        return f"""<aside class="sidebar" id="sidebar">
  <div class="sidebar-title">📖 Developer Memoir</div>
  <div class="sidebar-subtitle">{repo} &middot; {date}</div>
  <nav>{nav_html}</nav>
</aside>"""

    # ==================================================================
    # Main content area
    # ==================================================================

    def _build_main(self, memoir: Memoir) -> str:
        """Build the main content column."""
        parts: list[str] = []

        # Hero / cover
        parts.append(self._build_hero(memoir))

        # Chapters
        for chapter in memoir.chapters:
            parts.append(self._build_chapter(chapter))

        # Forecasts
        if memoir.forecasts:
            parts.append(self._build_forecasts_section(memoir.forecasts))

        # Stats appendix
        parts.append(self._build_stats_section(memoir.git_stats))

        content = "\n".join(parts)
        return f'<main class="main" id="main">{content}</main>'

    # ==================================================================
    # Hero / cover
    # ==================================================================

    def _build_hero(self, memoir: Memoir) -> str:
        """Build the hero / cover section."""
        repo = html_lib.escape(memoir.repo_name)
        date = memoir.generated_at.strftime("%d %B %Y")
        score = memoir.overall_health_score
        gauge = self._build_health_gauge(score)

        return f"""<section class="hero" id="cover">
  <div class="hero-icon">📖</div>
  <h1>Developer Memoir</h1>
  <h2>{repo}</h2>
  <p class="meta">Generated on {date}</p>
  {gauge}
</section>"""

    @staticmethod
    def _build_health_gauge(score: float) -> str:
        """Build an SVG semi-circular gauge for the health score."""
        # Determine colour
        if score >= 80:
            color = "#22c55e"
            label = "Excellent"
        elif score >= 60:
            color = "#84cc16"
            label = "Good"
        elif score >= 40:
            color = "#eab308"
            label = "Fair"
        elif score >= 20:
            color = "#f97316"
            label = "Poor"
        else:
            color = "#ef4444"
            label = "Critical"

        # SVG arc: semi-circle from left (180°) to right (0°)
        # radius 60, center (70, 70)
        # arc length for percentage
        r = 60
        cx, cy = 70, 70
        # Angle for the filled portion (0 = right, pi = left)
        # We go from 180° (left) clockwise
        angle = 180 * (score / 100)
        angle_rad = 180 - angle  # convert to math angle
        end_x = cx + r * _cos_deg(angle_rad)
        end_y = cy - r * _sin_deg(angle_rad)

        large_arc = 1 if score > 50 else 0

        filled_path = (
            f"M {cx - r},{cy} "
            f"A {r},{r} 0 {large_arc} 1 {end_x:.2f},{end_y:.2f}"
        )

        # Background arc (full semi-circle)
        bg_path = (
            f"M {cx - r},{cy} "
            f"A {r},{r} 0 1 1 {cx + r},{cy}"
        )

        return f"""<div class="health-gauge">
  <div class="label">Repository Health</div>
  <svg width="140" height="80" viewBox="0 0 140 80">
    <path d="{bg_path}" fill="none" stroke="var(--border)" stroke-width="10" stroke-linecap="round"/>
    <path d="{filled_path}" fill="none" stroke="{color}" stroke-width="10" stroke-linecap="round"/>
    <text x="{cx}" y="{cy - 5}" text-anchor="middle" fill="var(--text)"
          font-size="22" font-weight="800" font-family="var(--font-sans)">
      {score:.0f}
    </text>
    <text x="{cx}" y="{cy + 14}" text-anchor="middle" fill="var(--text-muted)"
          font-size="10" font-family="var(--font-sans)">
      /100
    </text>
  </svg>
  <div class="score-label">{label}</div>
</div>"""

    # ==================================================================
    # Chapter
    # ==================================================================

    def _build_chapter(self, chapter: Chapter) -> str:
        """Build a single chapter section."""
        anchor = _slugify(chapter.title)
        title = html_lib.escape(chapter.title)
        subtitle = html_lib.escape(chapter.subtitle)
        icon = _CHAPTER_TYPE_ICON.get(chapter.chapter_type, "📄")
        type_label = chapter.chapter_type.value.replace("_", " ").title()

        # Period
        period_html = ""
        if chapter.period_start and chapter.period_end:
            start = chapter.period_start.strftime("%d %b %Y")
            end = chapter.period_end.strftime("%d %b %Y")
            period_html = f'<p class="period">{start} — {end}</p>'

        # Narrative
        narrative_html = ""
        if chapter.narrative:
            # Convert newlines to paragraphs
            paragraphs = chapter.narrative.strip().split("\n\n")
            paras = []
            for p in paragraphs:
                p_escaped = html_lib.escape(p.strip())
                paras.append(f"<p>{p_escaped}</p>")
            narrative_html = f'<div class="narrative">{"".join(paras)}</div>'

        # Patterns
        patterns_html = ""
        for pattern in chapter.patterns:
            patterns_html += self._build_pattern(pattern)

        # Key commits
        commits_html = ""
        if chapter.key_commits:
            commits_html = self._build_commit_timeline(chapter.key_commits)

        # Chapter stats
        stats_html = ""
        if chapter.stats:
            stats_html = self._build_chapter_stats(chapter.stats)

        return f"""<section class="chapter" id="{anchor}">
  <hr class="separator"/>
  <div class="chapter-header">
    <span class="chapter-badge">{icon} {html_lib.escape(type_label)}</span>
    <h2>{title}</h2>
    <p class="subtitle">{subtitle}</p>
    {period_html}
  </div>
  {narrative_html}
  {patterns_html}
  {commits_html}
  {stats_html}
</section>"""

    # ==================================================================
    # Pattern card
    # ==================================================================

    @staticmethod
    def _build_pattern(pattern: Pattern) -> str:
        """Build a pattern callout card."""
        severity_class = f"severity-{pattern.severity.value}"
        color = _SEVERITY_COLORS.get(pattern.severity, "#94a3b8")
        emoji = _SEVERITY_EMOJI.get(pattern.severity, "⚠️")
        type_label = _PATTERN_TYPE_LABEL.get(
            pattern.pattern_type, pattern.pattern_type.value
        )
        title = html_lib.escape(pattern.title)
        description = html_lib.escape(pattern.description)

        # Unique ID for toggle
        uid = f"pat-{pattern.id}"

        # Affected files
        files_html = ""
        if pattern.affected_files:
            tags = []
            for f in pattern.affected_files[:6]:
                tags.append(f'<span class="tag">{html_lib.escape(f)}</span>')
            extra = (
                f'<span class="tag">+{len(pattern.affected_files) - 6} more</span>'
                if len(pattern.affected_files) > 6
                else ""
            )
            files_html = f'<div class="pattern-meta">{"".join(tags)}{extra}</div>'

        # Occurrence count + dates
        occ_html = ""
        if pattern.occurrence_count > 1:
            first = pattern.first_seen.strftime("%d %b %Y")
            last = pattern.last_seen.strftime("%d %b %Y")
            conf = pattern.confidence * 100
            occ_html = (
                f'<div class="pattern-meta">'
                f'<span class="tag">{pattern.occurrence_count} occurrences</span>'
                f'<span class="tag">{first} — {last}</span>'
                f'<span class="tag">Confidence: {conf:.0f}%</span>'
                f'</div>'
            )

        # Recommendation
        rec_html = ""
        if pattern.recommendation:
            rec = html_lib.escape(pattern.recommendation)
            rec_html = f'<div class="pattern-recommendation">💡 {rec}</div>'

        return f"""<div class="pattern-card {severity_class}">
  <div class="pattern-header" onclick="toggleSection('{uid}', this)">
    <span class="pattern-title">{emoji} {title}</span>
    <span>
      <span class="pattern-severity-badge" style="background:{color}">{type_label}</span>
      <span class="toggle-icon" id="icon-{uid}">▶</span>
    </span>
  </div>
  <div class="pattern-body collapsed" id="{uid}">
    <p>{description}</p>
    {files_html}
    {occ_html}
    {rec_html}
  </div>
</div>"""

    # ==================================================================
    # Forecast card
    # ==================================================================

    @staticmethod
    def _build_forecast(forecast: Forecast) -> str:
        """Build a forecast warning card."""
        risk_class = f"risk-{forecast.risk_level.value}"
        color = _RISK_COLORS.get(forecast.risk_level, "#94a3b8")
        emoji = _RISK_EMOJI.get(forecast.risk_level, "⚠️")
        title = html_lib.escape(forecast.title)
        description = html_lib.escape(forecast.description)
        probability_pct = forecast.probability * 100

        uid = f"fc-{forecast.id}"

        # Indicators
        indicators_html = ""
        if forecast.indicators:
            rows = []
            for ind in forecast.indicators:
                trend_class = {
                    "rising": "trend-up",
                    "stable": "trend-stable",
                    "declining": "trend-down",
                }.get(ind.trend, "")
                trend_arrow = {"rising": "↑", "stable": "→", "declining": "↓"}.get(
                    ind.trend, "?"
                )
                rows.append(
                    f'<div class="forecast-indicator">'
                    f'<span class="name">{html_lib.escape(ind.name)}</span>'
                    f'<span class="value">{ind.current_value:.1f} / {ind.threshold_value:.1f} '
                    f'<span class="{trend_class}">{trend_arrow} {ind.trend}</span></span>'
                    f'</div>'
                )
            indicators_html = (
                f'<div class="forecast-indicators">{"".join(rows)}</div>'
            )

        # Historical precedent
        precedent_html = ""
        if forecast.historical_precedent:
            prec = html_lib.escape(forecast.historical_precedent)
            precedent_html = f'<p>📜 <em>Historical precedent: {prec}</em></p>'

        # Recommendation
        rec_html = ""
        if forecast.recommendation:
            rec = html_lib.escape(forecast.recommendation)
            rec_html = f'<div class="forecast-recommendation">💡 {rec}</div>'

        return f"""<div class="forecast-card {risk_class}">
  <div class="forecast-header" onclick="toggleSection('{uid}', this)">
    <span class="forecast-title">{emoji} {title}</span>
    <span>
      <span class="risk-badge" style="background:{color}">{forecast.risk_level.value.upper()}</span>
      <span class="toggle-icon" id="icon-{uid}">▶</span>
    </span>
  </div>
  <div class="forecast-body collapsed" id="{uid}">
    <p>{description}</p>
    <p><strong>Probability:</strong> {probability_pct:.0f}% &nbsp;|&nbsp; <strong>Timeline:</strong> {html_lib.escape(forecast.estimated_timeline)}</p>
    {indicators_html}
    {precedent_html}
    {rec_html}
  </div>
</div>"""

    # ==================================================================
    # Forecasts section
    # ==================================================================

    def _build_forecasts_section(self, forecasts: List[Forecast]) -> str:
        """Build the forecasts section."""
        cards = "\n".join(self._build_forecast(f) for f in forecasts)
        return f"""<section class="chapter" id="forecasts">
  <hr class="separator"/>
  <div class="chapter-header">
    <span class="chapter-badge">🔮 Forecasts</span>
    <h2>Risk Forecasts</h2>
  </div>
  {cards}
</section>"""

    # ==================================================================
    # Commit timeline
    # ==================================================================

    @staticmethod
    def _build_commit_timeline(commits: List[CommitData]) -> str:
        """Build a chronological commit timeline."""
        sorted_commits = sorted(commits, key=lambda c: c.date)
        items: list[str] = []

        for commit in sorted_commits:
            date_str = commit.date.strftime("%Y-%m-%d")
            subject = commit.message.split("\n", 1)[0].strip()
            if len(subject) > 80:
                subject = subject[:77] + "..."
            subject = html_lib.escape(subject)

            net = commit.insertions - commit.deletions
            if net > 0:
                net_class = "positive"
                net_str = f"+{net}"
            elif net < 0:
                net_class = "negative"
                net_str = str(net)
            else:
                net_class = "neutral"
                net_str = "0"

            items.append(
                f'<div class="commit-item">'
                f'<span class="commit-hash">{html_lib.escape(commit.short_hash)}</span> '
                f'<span class="commit-date">{date_str}</span> '
                f'<span class="commit-msg">— {subject}</span> '
                f'<span class="commit-net {net_class}">{net_str}</span>'
                f'</div>'
            )

        items_html = "\n".join(items)
        return f"""<div class="commit-timeline">
  <h4 style="font-size:0.85rem; font-weight:600; margin-bottom:0.65rem; color:var(--text-secondary);">📝 Key Commits</h4>
  {items_html}
</div>"""

    # ==================================================================
    # Chapter stats
    # ==================================================================

    @staticmethod
    def _build_chapter_stats(stats: dict) -> str:
        """Build an inline chapter statistics table."""
        rows: list[str] = []
        for key, value in stats.items():
            label = key.replace("_", " ").replace("-", " ").title()
            display = f"{value:,}" if isinstance(value, int) else str(value)
            rows.append(
                f"<tr><td>{html_lib.escape(label)}</td>"
                f"<td>{html_lib.escape(display)}</td></tr>"
            )

        rows_html = "\n".join(rows)
        return f"""<div class="chapter-stats">
  <h4>📊 Chapter Statistics</h4>
  <table>{rows_html}</table>
</div>"""

    # ==================================================================
    # Stats appendix
    # ==================================================================

    @staticmethod
    def _build_stats_section(stats: GitStats) -> str:
        """Build the statistics appendix section."""
        # Overview table
        overview_rows = [
            ("Total Commits", f"{stats.total_commits:,}"),
            ("Unique Authors", f"{stats.unique_authors:,}"),
            ("Files Changed", f"{stats.total_files_changed:,}"),
            ("Lines Added", f"{stats.total_insertions:,}"),
            ("Lines Removed", f"{stats.total_deletions:,}"),
            ("Avg. Message Length", f"{stats.avg_message_length:.0f} chars"),
            ("First Commit", stats.first_commit_date.strftime("%d %b %Y")),
            ("Latest Commit", stats.last_commit_date.strftime("%d %b %Y")),
        ]
        overview_html = "\n".join(
            f"<tr><td>{label}</td><td>{value}</td></tr>"
            for label, value in overview_rows
        )

        # Authors
        authors_html = ""
        if stats.author_names:
            names = ", ".join(html_lib.escape(n) for n in stats.author_names[:20])
            extra = (
                f" (+{len(stats.author_names) - 20} more)"
                if len(stats.author_names) > 20
                else ""
            )
            authors_html = f'<p style="margin-top:1rem;font-size:0.85rem;color:var(--text-secondary);"><strong>Authors:</strong> {names}{extra}</p>'

        # Day-of-week table
        day_html = ""
        if stats.commit_frequency_by_day:
            day_order = [
                "Monday", "Tuesday", "Wednesday",
                "Thursday", "Friday", "Saturday", "Sunday",
            ]
            rows = []
            for day in day_order:
                count = stats.commit_frequency_by_day.get(day, 0)
                if count > 0:
                    rows.append(f"<tr><td>{day}</td><td>{count:,}</td></tr>")
            if rows:
                day_html = f"""<h3 style="font-size:1.1rem;font-weight:600;margin-top:2rem;margin-bottom:0.75rem;">Commits by Day of Week</h3>
<table class="stats-table"><tr><th>Day</th><th>Commits</th></tr>{"".join(rows)}</table>"""

        # Peak hours
        hours_html = ""
        if stats.hourly_distribution:
            sorted_hours = sorted(
                stats.hourly_distribution.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:8]
            rows = [
                f"<tr><td>{h:02d}:00</td><td>{c:,}</td></tr>"
                for h, c in sorted_hours
            ]
            hours_html = f"""<h3 style="font-size:1.1rem;font-weight:600;margin-top:2rem;margin-bottom:0.75rem;">Peak Hours</h3>
<table class="stats-table"><tr><th>Hour</th><th>Commits</th></tr>{"".join(rows)}</table>"""

        # Most changed files
        files_html = ""
        if stats.most_changed_files:
            rows = []
            for fp, count in stats.most_changed_files[:15]:
                display = (
                    html_lib.escape(fp) if len(fp) <= 50
                    else html_lib.escape("…" + fp[-49:])
                )
                rows.append(
                    f"<tr><td><code>{display}</code></td><td>{count:,}</td></tr>"
                )
            files_html = f"""<h3 style="font-size:1.1rem;font-weight:600;margin-top:2rem;margin-bottom:0.75rem;">Most Changed Files</h3>
<table class="stats-table"><tr><th>File</th><th>Changes</th></tr>{"".join(rows)}</table>"""

        return f"""<section class="stats-section" id="statistics">
  <hr class="separator"/>
  <h2>📊 Statistics Summary</h2>
  <table class="stats-table"><tr><th>Metric</th><th>Value</th></tr>{overview_html}</table>
  {authors_html}
  {day_html}
  {hours_html}
  {files_html}
</section>"""

    # ==================================================================
    # JavaScript
    # ==================================================================

    @staticmethod
    def _build_script() -> str:
        """Build the inline JavaScript for interactivity."""
        return """<script>
// Toggle collapsible sections
function toggleSection(id, header) {
  var body = document.getElementById(id);
  var icon = document.getElementById('icon-' + id);
  if (!body) return;
  if (body.classList.contains('collapsed')) {
    body.classList.remove('collapsed');
    body.classList.add('expanded');
    if (icon) icon.classList.add('open');
  } else {
    body.classList.remove('expanded');
    body.classList.add('collapsed');
    if (icon) icon.classList.remove('open');
  }
}

// Mobile sidebar toggle
(function() {
  var btn = document.getElementById('menuToggle');
  var sidebar = document.getElementById('sidebar');
  if (!btn || !sidebar) return;
  btn.addEventListener('click', function() {
    sidebar.classList.toggle('open');
  });
  // Close sidebar on link click (mobile)
  sidebar.querySelectorAll('nav a').forEach(function(a) {
    a.addEventListener('click', function() {
      if (window.innerWidth <= 1024) {
        sidebar.classList.remove('open');
      }
    });
  });
})();
</script>"""


# ---------------------------------------------------------------------------
# Math helpers (no import of math to keep it lightweight)
# ---------------------------------------------------------------------------

import math as _math


def _cos_deg(angle: float) -> float:
    """Cosine of an angle specified in degrees."""
    return _math.cos(_math.radians(angle))


def _sin_deg(angle: float) -> float:
    """Sine of an angle specified in degrees."""
    return _math.sin(_math.radians(angle))


def _slugify(text: str) -> str:
    """Convert text to an HTML-safe anchor slug."""
    slug = text.lower().strip()
    result: list[str] = []
    for ch in slug:
        if ch.isalnum():
            result.append(ch)
        elif ch in (" ", "-", "_"):
            result.append("-")
    return "".join(result).strip("-")
