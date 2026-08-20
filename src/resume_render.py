"""Render a resume variant to PDF.

Key design choice: the LLM only ever edits *content* (bullets, skill emphasis).
This module owns *formatting*. So no matter what the model does, the layout,
fonts, and structure of your resume can never drift. That's what keeps a tailored
resume from turning into a brand-new document.

Uses WeasyPrint (HTML/CSS -> PDF). On some systems WeasyPrint needs system libs
(pango/cairo). If it won't install, see the README for the Playwright fallback.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

from jinja2 import Template

from . import config


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_context(resume_content: dict) -> dict:
    """Combine shared identity/education with a resume variant's content."""
    profile = _load(config.DATA_DIR / "candidate_profile.json")
    ident = profile["identity"]
    return {
        "name": ident["full_name"],
        "location": ident["location"],
        "phone": ident["phone"],
        "email": ident["email"],
        "linkedin": ident["linkedin"],
        "github": ident["github"],
        "education": profile["education"],
        "experience": resume_content["experience"],
        "projects": resume_content["projects"],
        "skills": resume_content["skills"],
    }


def _safe_company(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def render_pdf(resume_content: dict, company: str) -> str:
    """Render to output/resumes/Aryan_Patial_{company}.pdf and return the path."""
    template = Template((config.TEMPLATES_DIR / "resume.html").read_text(encoding="utf-8"))
    html_str = template.render(**build_context(resume_content))

    filename = f"Aryan_Patial_{_safe_company(company)}.pdf"
    out_path = config.OUTPUT_RESUMES_DIR / filename

    from weasyprint import HTML  # imported lazily so the rest of the app runs without it
    HTML(string=html_str).write_pdf(str(out_path))
    return str(out_path)
