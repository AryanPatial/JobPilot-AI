"""The graph nodes. Each is a plain function: read state, do one job, return a
partial update. This is the whole workflow, one function per box in the diagram.
"""
from __future__ import annotations
import json
from pathlib import Path

from . import config
from .schemas import ResumeMatch, TailoringPlan, DraftedAnswer
from .sources.greenhouse import get_source
from . import resume_render
from . import tracker


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_resume_variants() -> list[dict]:
    return [_load_json(p) for p in sorted(config.RESUMES_DIR.glob("*.json"))]


# --------------------------------------------------------------------------- #
# 1. SEARCH  — pull jobs, pick the first match to process (V1 = one job)
# --------------------------------------------------------------------------- #
def search_jobs(state):
    criteria = state["search_criteria"]
    source = get_source(criteria.get("source", "greenhouse"))
    jobs = source.search(criteria)
    if not jobs:
        return {"jobs": [], "status": "no jobs matched criteria", "error": "no_jobs"}

    chosen = jobs[0]  # V1: process the first match. V2: rank all, pick best.
    return {
        "jobs": jobs,
        "current_job": chosen,
        "job_description": chosen["description"],
        "status": f"selected job: {chosen['title']} @ {chosen['company']}",
    }


# --------------------------------------------------------------------------- #
# 2. MATCH  — score every resume variant against the JD, pick the best
# --------------------------------------------------------------------------- #
def match_resumes(state):
    jd = state["job_description"]
    variants = _load_resume_variants()
    llm = config.get_llm(temperature=0.0, structured_schema=ResumeMatch)

    scores = []
    for v in variants:
        summary = _resume_summary(v)
        prompt = (
            "Score how well this resume fits the job description.\n\n"
            f"JOB DESCRIPTION:\n{jd}\n\n"
            f"RESUME ({v['label']}):\n{summary}\n\n"
            "Return a fit score 0-100, brief reasoning, matched and missing keywords."
        )
        result: ResumeMatch = llm.invoke(prompt)
        scores.append({
            "resume_id": v["resume_id"],
            "label": v["label"],
            "score": result.score,
            "reasoning": result.reasoning,
            "matched": result.matched_keywords,
            "missing": result.missing_keywords,
        })

    best = max(scores, key=lambda s: s["score"])
    return {
        "resume_scores": scores,
        "selected_resume_id": best["resume_id"],
        "status": f"best resume: {best['resume_id']} ({best['score']}/100)",
    }


def _resume_summary(v: dict) -> str:
    lines = []
    for j in v["experience"]:
        lines.append(f"{j['title']} @ {j['company']}: " + "; ".join(j["bullets"]))
    for p in v["projects"]:
        lines.append(f"Project {p['name']}: " + "; ".join(p["bullets"]))
    lines.append("Skills: " + " | ".join(f"{k}: {val}" for k, val in v["skills"].items()))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 3. TAILOR  — light, honest edits to the winning resume's CONTENT only
# --------------------------------------------------------------------------- #
def tailor_resume(state):
    jd = state["job_description"]
    rid = state["selected_resume_id"]
    variant = next(v for v in _load_resume_variants() if v["resume_id"] == rid)

    llm = config.get_llm(temperature=0.3, structured_schema=TailoringPlan)
    prompt = (
        "You are tailoring an existing resume to a job description. Rules:\n"
        "- NEVER invent experience, skills, employers, or metrics.\n"
        "- Only reword a FEW existing bullets to surface relevant work.\n"
        "- Keep each bullet roughly the same length.\n\n"
        f"JOB DESCRIPTION:\n{jd}\n\n"
        f"CURRENT RESUME CONTENT:\n{_resume_summary(variant)}\n\n"
        "Return an emphasis list, a few revised bullets, and a summary of changes."
    )
    plan: TailoringPlan = llm.invoke(prompt)

    # Apply the revised bullets back onto a copy of the variant (match by original text).
    tailored = json.loads(json.dumps(variant))  # deep copy
    rewrites = {b.original.strip(): b.revised.strip() for b in plan.revised_bullets}
    for job in tailored["experience"]:
        job["bullets"] = [rewrites.get(b.strip(), b) for b in job["bullets"]]
    for proj in tailored["projects"]:
        proj["bullets"] = [rewrites.get(b.strip(), b) for b in proj["bullets"]]

    return {
        "tailored_resume": tailored,
        "status": "resume tailored: " + plan.summary_of_changes[:120],
    }


# --------------------------------------------------------------------------- #
# 4. RENDER  — content + template -> PDF (formatting is code-controlled)
# --------------------------------------------------------------------------- #
def render_resume(state):
    company = state["current_job"]["company"]
    path = resume_render.render_pdf(state["tailored_resume"], company)
    return {"resume_pdf_path": path, "status": f"resume PDF ready: {path}"}


# --------------------------------------------------------------------------- #
# 5. PREPARE  — map standard fields, draft answers to open-ended questions
# --------------------------------------------------------------------------- #
def prepare_application(state):
    profile = _load_json(config.DATA_DIR / "candidate_profile.json")
    ident = profile["identity"]
    ans = profile["application_answers"]

    fields = {
        "first_name": ident["first_name"],
        "last_name": ident["last_name"],
        "email": ident["email"],
        "phone": ident["phone"],
        "location": ident["location"],
        "linkedin": ident["linkedin"],
        "github": ident["github"],
        "resume_file": state.get("resume_pdf_path"),
        "work_authorization": ans["work_authorization_us"],
        "requires_sponsorship": ans["requires_sponsorship_now_or_future"],
    }

    # Draft answers to any free-text questions found in the JD-adjacent flow.
    # (In V1 we don't scrape the live form; we pre-draft the common one.)
    open_qs = []
    common = ["Why are you interested in this role?"]
    llm = config.get_llm(temperature=0.4, structured_schema=DraftedAnswer)
    role = state["current_job"]["title"]
    company = state["current_job"]["company"]
    for q in common:
        prompt = (
            f"Candidate: {ident['full_name']}, MS Data Science at UNT.\n"
            f"Role: {role} at {company}.\n"
            f"Question: {q}\n"
            "Write a concise 2-3 sentence answer grounded only in a data/ML background. "
            "Do not invent specifics about the company."
        )
        drafted: DraftedAnswer = llm.invoke(prompt)
        open_qs.append({"question": q, "drafted_answer": drafted.answer, "confidence": drafted.confidence})

    return {
        "application_fields": fields,
        "open_questions": open_qs,
        "status": "application package prepared — ready for human review",
    }


# --------------------------------------------------------------------------- #
# 6. HUMAN REVIEW  — the graph pauses here (see graph.py interrupt_before)
# --------------------------------------------------------------------------- #
def human_review(state):
    # After you resume the graph, we treat that as approval and record it.
    return {"status": "approved by human"}


# --------------------------------------------------------------------------- #
# 7. TRACK  — append the row to the Excel tracker
# --------------------------------------------------------------------------- #
def track_application(state):
    job = state["current_job"]
    scores = {s["resume_id"]: s["score"] for s in state["resume_scores"]}
    path = tracker.log_application(
        company=job["company"],
        job_title=job["title"],
        resume_used=Path(state["resume_pdf_path"]).name,
        match_score=scores.get(state["selected_resume_id"]),
        job_url=job["url"],
        status="Prepared",
    )
    return {"status": f"logged to tracker: {path}"}
