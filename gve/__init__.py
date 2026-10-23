"""Grounded Variance Explainer.

Core engine, kept separate from any UI so it can sit behind Streamlit today and
an API (e.g. FastAPI) later.

Pipeline:
    load_gl -> run_preflight -> compute_variances -> draft_commentary -> validate
"""

from .loader import load_gl, load_coa, REQUIRED_COLUMNS
from .preflight import run_preflight, Issue
from .variance import compute_variances, prepare_for_analysis, fmt_money, fmt_pct
from .commentary import draft_commentary, build_facts
from .validator import validate

__all__ = [
    "load_gl", "load_coa", "REQUIRED_COLUMNS",
    "run_preflight", "Issue",
    "prepare_for_analysis", "compute_variances", "fmt_money", "fmt_pct",
    "build_facts", "draft_commentary",
    "validate",
]
