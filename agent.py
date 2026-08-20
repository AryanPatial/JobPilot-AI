"""
agent.py  —  The brain: the state, the pipeline steps (nodes), and the graph.

The state is the shared clipboard that flows through every step. Each step reads
what it needs and writes back a small update. The graph just says what order the
steps run in. Read build_graph() at the bottom to see the whole flow at a glance.
"""
from __future__ import annotations
import json
import sqlite3
from typing import TypedDict, Optional, List

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

import helpers as H


# ------------------------------------------------------------------ STATE --- #
class State(TypedDict, total=False):
    board_tokens: List[str]
    titles: List[str]

    jobs: list
    current_job: Optional[dict]
    job_description: Optional[str]

    resume_scores: list
    selected_resume_id: Optional[str]

    tailored_resume: Optional[dict]
    resume_pdf_path: Optional[str]

    application_fields: Optional[dict]
    open_questions: Optional[list]

    status: str
    error: Optional[str]


# ------------------------------------------------ structured LLM outputs --- #
class ResumeMatch(BaseModel):
    score: int = Field(ge=0, le=100, description="Fit score 0-100 vs the JD")
    reasoning: str = Field(description="One or two sentences on the score")
    matched_keywords: List[str] = Field(default_factory=list)
    missing_keywords: List[str] = Field(default_factory=list)


class Bullet(BaseModel):
    original: str
    revised: str = Field(description="Reworded bullet. Same facts, nothing invented.")


class TailoringPlan(BaseModel):
    revised_bullets: List[Bullet] = Field(default_factory=list)
    summary_of_changes: str


class WhyAnswer(BaseModel):
    answer: str = Field(description="2-3 sentence answer grounded only in the candidate's real background")


def _resume_summary(v: dict) -> str:
    lines = [f"{j['title']} @ {j['company']}: " + "; ".join(j["bullets"]) for j in v["experience"]]
    lines += [f"Project {p['name']}: " + "; ".join(p["bullets"]) for p in v["projects"]]
    lines.append("Skills: " + " | ".join(f"{k}: {val}" for k, val in v["skills"].items()))
    return "\n".join(lines)


# ------------------------------------------------------------------ NODES --- #
def search_jobs(state: State):
    jobs = H.search_jobs_greenhouse(state["board_tokens"], state["titles"])
    if not jobs:
        return {"jobs": [], "error": "no_jobs", "status": "no US jobs matched"}
    chosen = jobs[0]  # V1: first match. (V2: rank all, pick best.)
    return {
        "jobs": jobs,
        "current_job": chosen,
        "job_description": chosen["description"],
        "status": f"selected: {chosen['title']} @ {chosen['company']} ({chosen['location']})",
    }


def match_resumes(state: State):
    jd = state["job_description"]
    llm = H.get_llm(temperature=0.0, structured_schema=ResumeMatch)
    scores = []
    for v in H.load_resume_variants():
        prompt = (
            "Score how well this resume fits the job description (0-100).\n\n"
            f"JOB DESCRIPTION:\n{jd}\n\nRESUME ({v['label']}):\n{_resume_summary(v)}"
        )
        r: ResumeMatch = H.invoke_llm(llm, prompt)
        scores.append({"resume_id": v["resume_id"], "label": v["label"], "score": r.score,
                       "reasoning": r.reasoning, "matched": r.matched_keywords, "missing": r.missing_keywords})
    best = max(scores, key=lambda s: s["score"])
    return {"resume_scores": scores, "selected_resume_id": best["resume_id"],
            "status": f"best resume: {best['resume_id']} ({best['score']}/100)"}


def tailor_resume(state: State):
    jd = state["job_description"]
    rid = state["selected_resume_id"]
    variant = next(v for v in H.load_resume_variants() if v["resume_id"] == rid)

    llm = H.get_llm(temperature=0.3, structured_schema=TailoringPlan)
    prompt = (
        "Tailor this resume to the JD. Rules: NEVER invent experience/skills/metrics. "
        "Only reword a FEW existing bullets to surface relevant work. Keep lengths similar.\n\n"
        f"JOB DESCRIPTION:\n{jd}\n\nRESUME CONTENT:\n{_resume_summary(variant)}"
    )
    plan: TailoringPlan = H.invoke_llm(llm, prompt)

    tailored = json.loads(json.dumps(variant))  # deep copy
    rewrites = {b.original.strip(): b.revised.strip() for b in plan.revised_bullets}
    for job in tailored["experience"]:
        job["bullets"] = [rewrites.get(b.strip(), b) for b in job["bullets"]]
    for proj in tailored["projects"]:
        proj["bullets"] = [rewrites.get(b.strip(), b) for b in proj["bullets"]]
    return {"tailored_resume": tailored, "status": "resume tailored: " + plan.summary_of_changes[:100]}


def render_resume(state: State):
    path = H.render_resume_pdf(state["tailored_resume"], state["current_job"]["company"])
    return {"resume_pdf_path": path, "status": f"PDF ready: {path}"}


def prepare_application(state: State):
    profile = H.load_profile()
    ident = profile["identity"]
    ans = dict(profile["application_answers"])   # copy

    # Fill the AUTO fields per job
    if ans.get("salary_expectation") == "AUTO":
        ans["salary_expectation"] = H.salary_expectation_from_jd(state["job_description"])
    if ans.get("earliest_start_date") == "AUTO":
        ans["earliest_start_date"] = H.earliest_start_date()

    variant = next(v for v in H.load_resume_variants() if v["resume_id"] == state["selected_resume_id"])
    yrs = H.years_of_experience(variant)

    fields = {
        "first_name": ident["first_name"], "last_name": ident["last_name"],
        "email": ident["email"], "phone": ident["phone"], "location": ident["location"],
        "country": ident.get("country", "United States"),
        "linkedin": ident["linkedin"], "github": ident["github"],
        "resume_file": state.get("resume_pdf_path"),
        "years_experience": yrs,
        "work_authorization": ans["work_authorization_us"],
        "requires_sponsorship": ans["requires_sponsorship_now_or_future"],
        "visa_status": ans["visa_status"],
        "willing_to_relocate": ans["willing_to_relocate"],
        "salary_expectation": ans["salary_expectation"],
        "earliest_start_date": ans["earliest_start_date"],
        "gender": ans["gender"], "race_ethnicity": ans["race_ethnicity"],
        "veteran_status": ans["veteran_status"], "disability_status": ans["disability_status"],
        "how_did_you_hear": ans["how_did_you_hear"],
    }

    # Screening questions (non-legal). Facts still come from the file; the only
    # computed one is the office location, which follows the job we picked.
    screening = {k: v for k, v in profile.get("screening_answers", {}).items()
                 if not k.startswith("_")}
    if screening.get("preferred_office_location") == "AUTO":
        screening["preferred_office_location"] = state["current_job"]["location"]
    fields.update(screening)
    fields["job_location"] = state["current_job"]["location"]

    # LLM writes ONLY the prose answer; every fact above came from the file.
    job = state["current_job"]
    llm = H.get_llm(temperature=0.4, structured_schema=WhyAnswer)
    edu = profile["education"][0]
    why: WhyAnswer = H.invoke_llm(
        llm,
        f"Candidate: {ident['full_name']}, {edu['degree']} at {edu['school']}, "
        f"~{yrs} yrs experience.\n"
        f"Role: {job['title']} at {job['company']}.\n"
        "Write a concise 2-3 sentence answer to 'Why are you interested in this role?' "
        "grounded only in a data/ML background. Do not invent company specifics."
    )
    open_qs = [{"question": "Why are you interested in this role?", "drafted_answer": why.answer}]

    return {"application_fields": fields, "open_questions": open_qs,
            "status": "application package prepared"}


def human_review(state: State):
    # Reaching here means you resumed the graph = you approved after review.
    return {"status": "approved by human"}


def track_application(state: State):
    job = state["current_job"]
    scores = {s["resume_id"]: s["score"] for s in state["resume_scores"]}
    from pathlib import Path
    path = H.log_application(
        company=job["company"], job_title=job["title"],
        resume_used=Path(state["resume_pdf_path"]).name,
        match_score=scores.get(state["selected_resume_id"]),
        job_url=job["url"], status="Prepared",
    )
    return {"status": f"logged to tracker: {path}"}


# ------------------------------------------------------------------ GRAPH --- #
def _after_search(state: State) -> str:
    """If the search found nothing, stop cleanly instead of crashing downstream."""
    return "end" if state.get("error") == "no_jobs" else "match_resumes"


def _make_checkpointer():
    """SQLite checkpointing so a crash (or the human pause) resumes mid-run.

    NOTE: SqliteSaver.from_conn_string() is a *context manager*, so it can't be
    handed straight to compile(). We own the connection instead and keep it open
    for the life of the process. check_same_thread=False because LangGraph may
    touch it from a worker thread.
    """
    conn = sqlite3.connect(str(H.CHECKPOINT_DB), check_same_thread=False)
    return SqliteSaver(conn)


def build_graph():
    g = StateGraph(State)
    g.add_node("search_jobs", search_jobs)
    g.add_node("match_resumes", match_resumes)
    g.add_node("tailor_resume", tailor_resume)
    g.add_node("render_resume", render_resume)
    g.add_node("prepare_application", prepare_application)
    g.add_node("human_review", human_review)
    g.add_node("track_application", track_application)

    g.add_edge(START, "search_jobs")
    g.add_conditional_edges("search_jobs", _after_search,
                            {"match_resumes": "match_resumes", "end": END})
    g.add_edge("match_resumes", "tailor_resume")
    g.add_edge("tailor_resume", "render_resume")
    g.add_edge("render_resume", "prepare_application")
    g.add_edge("prepare_application", "human_review")
    g.add_edge("human_review", "track_application")
    g.add_edge("track_application", END)

    # Pause before human_review so you can auto-fill the form + review before submit.
    return g.compile(checkpointer=_make_checkpointer(), interrupt_before=["human_review"])
