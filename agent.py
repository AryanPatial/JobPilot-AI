"""LangGraph state machine: state, nodes, routing, graph."""
from __future__ import annotations
import json
import sqlite3
from typing import TypedDict, Optional, List

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

import helpers as H
import llm_tasks as T
import scoring as S


# ------------------------------------------------------------------ STATE --- #
class State(TypedDict, total=False):
    board_tokens: List[str]
    intent: str                      # plain-English, e.g. "gen AI roles"
    titles: List[str]                # LLM-expanded from intent

    jobs: list
    current_job: Optional[dict]
    job_description: Optional[str]

    resume_scores: list
    selected_resume_id: Optional[str]

    job_index: int                   # which of `jobs` we're on
    skipped_jobs: list               # jobs abandoned for poor JD coverage
    selected_projects: list          # chosen from the experience pool
    tailored_resume: Optional[dict]
    coverage: Optional[dict]         # deterministic score + named gaps
    attempts: int                    # project combinations tried (hard cap)
    resume_pdf_path: Optional[str]

    # The exploration loop keeps the best attempt, not the last one.
    tried_project_sets: list         # sorted id-tuples already scored, as lists
    best_resume: Optional[dict]
    best_coverage: Optional[dict]
    best_projects: list
    invented_terms: list             # anti-invention guard, not part of the score
    jd_requirements: list            # fixed per job, so combinations are comparable

    application_fields: Optional[dict]
    open_questions: Optional[list]

    status: str
    submitted_status: str            # "Applied" once you confirm you submitted
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
    """Intent -> LLM title expansion -> keyword+US prefilter -> ONE batched judge."""
    intent = state.get("intent") or "data and AI roles"
    titles = T.expand_query(intent)
    print(f"  [search] '{intent}' -> {len(titles)} title variants")
    print(f"           {', '.join(titles[:8])}"
          + (f" … +{len(titles) - 8} more" if len(titles) > 8 else ""))

    jobs = H.search_jobs_greenhouse(state["board_tokens"], titles)
    print(f"  [search] {len(jobs)} US candidates after keyword prefilter")

    # Don't offer a job that's already in the tracker as Applied
    done = H.already_applied()
    if done:
        before = len(jobs)
        jobs = [j for j in jobs if (j.get("url") or "").split("?")[0] not in done]
        if before != len(jobs):
            print(f"  [search] {before - len(jobs)} already applied to - skipped")
    if not jobs:
        return {"jobs": [], "titles": titles, "error": "no_jobs",
                "status": "no US jobs matched"}

    yrs = H.years_of_experience(H.load_resume_variants()[0])
    kept = T.judge_jobs(intent, jobs, yrs)
    print(f"  [search] {len(kept)} survived intent + seniority judging")
    if not kept:
        return {"jobs": jobs, "titles": titles, "error": "no_jobs",
                "status": "nothing passed the fit judge"}

    chosen = kept[0]
    return {
        "jobs": kept, "titles": titles, "current_job": chosen,
        "job_description": chosen["description"], "attempts": 0,
        "job_index": 0, "skipped_jobs": [],
        "status": f"selected: {chosen['title']} @ {chosen['company']} "
                  f"({chosen['location']}) - {chosen.get('verdict', '')}",
    }


def match_resumes(state: State):
    jd = state["job_description"]
    scores = []
    for v in H.load_resume_variants():
        prompt = (
            "Score how well this resume fits the job description (0-100).\n\n"
            f"JOB DESCRIPTION:\n{jd}\n\nRESUME ({v['label']}):\n{_resume_summary(v)}"
        )
        r: ResumeMatch = H.call_llm(prompt, structured_schema=ResumeMatch, temperature=0.0)
        scores.append({"resume_id": v["resume_id"], "label": v["label"], "score": r.score,
                       "reasoning": r.reasoning, "matched": r.matched_keywords, "missing": r.missing_keywords})
    best = max(scores, key=lambda s: s["score"])
    return {"resume_scores": scores, "selected_resume_id": best["resume_id"],
            "status": f"best resume: {best['resume_id']} ({best['score']}/100)"}


def select_projects(state: State):
    """Pick which of the candidate's REAL projects belong on this resume."""
    pool = H.load_experience_pool()
    attempts = state.get("attempts", 0)
    missing = (state.get("coverage") or {}).get("missing") if attempts else None
    tried = state.get("tried_project_sets", [])
    picks = T.select_content(state["job_description"], pool,
                             missing_skills=missing, avoid=tried)
    print(f"  [select] combination {attempts + 1}/{S.MAX_ATTEMPTS}: "
          f"{', '.join(p['name'][:30] for p in picks)}")
    return {"selected_projects": picks,
            "status": f"selected {len(picks)} projects from a pool of {len(pool)}"}


def tailor_resume(state: State):
    """Assemble base variant + selected projects, then reword bullets to the JD."""
    rid = state["selected_resume_id"]
    variant = next(v for v in H.load_resume_variants() if v["resume_id"] == rid)

    assembled = json.loads(json.dumps(variant))          # deep copy
    assembled["projects"] = [{"name": p["name"], "bullets": list(p["bullets"])}
                             for p in state["selected_projects"]]

    plan: TailoringPlan = T.H.call_llm(
        "Tailor this resume to the JD. Rules: NEVER invent experience, skills, or "
        "metrics. Only reword a FEW existing bullets to surface relevant work. "
        "Keep lengths similar.\n\n"
        f"JOB DESCRIPTION:\n{state['job_description']}\n\n"
        f"RESUME CONTENT:\n{_resume_summary(assembled)}",
        structured_schema=TailoringPlan, temperature=0.3)

    rewrites = {b.original.strip(): b.revised.strip() for b in plan.revised_bullets}
    for job in assembled["experience"]:
        job["bullets"] = [rewrites.get(b.strip(), b) for b in job["bullets"]]
    for proj in assembled["projects"]:
        proj["bullets"] = [rewrites.get(b.strip(), b) for b in proj["bullets"]]

    return {"tailored_resume": assembled,
            "attempts": state.get("attempts", 0) + 1,
            "status": "tailored: " + plan.summary_of_changes[:90]}


def grade_resume(state: State):
    """Score the tailored résumé - evidence from the model, verdict from code."""
    resume = state["tailored_resume"]
    vocab = H.claimable_vocabulary()

    # One text, shown to the model and used to verify its citations
    edu = "; ".join(f"{e['degree']}, {e['school']} ({e.get('dates', '')})"
                    for e in H.load_profile().get("education", []))
    graded_text = S.resume_text(resume, lower=False) + " Education: " + edu

    # Extracted once per job and reused for every combination, so the three attempts sit the s
    requirements = state.get("jd_requirements")
    if not requirements:
        requirements = T.extract_requirements(state["job_description"])
        print(f"  [score] {len(requirements)} requirements extracted from the JD "
              f"({sum(r['must_have'] for r in requirements)} required)")

    cited = T.cite_evidence(requirements, graded_text)
    cov = S.verify_and_score(graded_text, cited)

    met = len(cov["covered"])
    print(f"  [score] {cov['score']}/100 (combination {state['attempts']}/"
          f"{S.MAX_ATTEMPTS}) - {met}/{len(requirements)} requirements evidenced")
    for name in cov["missing"][:4]:
        print(f"            unmet: {name[:64]}")
    if cov["fabricated"]:
        print(f"  [score] REJECTED {len(cov['fabricated'])} citation(s) that are not "
              f"in the résumé: {', '.join(cov['fabricated'][:3])}")

    # invention guard, not part of the score
    invented = S.introduced_terms(resume, H.EXPERIENCE_POOL_PATH,
                                  *H.resume_variant_paths())
    if invented:
        print(f"  [guard] terms on the résumé with no basis in your files: "
              f"{', '.join(invented[:6])}")

    # Remember this combination so select_projects cannot serve it up again.
    tried = list(state.get("tried_project_sets", []))
    tried.append(sorted(p["id"] for p in state["selected_projects"] if "id" in p))

    # Keep the best attempt, not the most recent one
    out = {"coverage": cov, "tried_project_sets": tried,
           "invented_terms": invented, "jd_requirements": requirements,
           "status": f"coverage {cov['score']}/100"}
    best = state.get("best_coverage")
    if best is None or cov["score"] > best["score"]:
        if best is not None:
            print(f"  [score] new best - {best['score']} -> {cov['score']}")
        out.update({"best_coverage": cov,
                    "best_resume": state["tailored_resume"],
                    "best_projects": state["selected_projects"]})
    else:
        print(f"  [score] keeping the earlier best of {best['score']}/100")
    return out


def render_resume(state: State):
    """Render the BEST combination found, and promote it into the state."""
    job = state["current_job"]
    best_cov = state.get("best_coverage") or state.get("coverage")
    resume = state.get("best_resume") or state["tailored_resume"]
    projects = state.get("best_projects") or state.get("selected_projects", [])

    path = H.render_resume_pdf(resume, job["company"], job["title"])
    return {"resume_pdf_path": path, "tailored_resume": resume,
            "coverage": best_cov, "selected_projects": projects,
            "status": f"PDF ready ({best_cov['score']}/100): {path}"}


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

    # Screening questions (non-legal)
    screening = {k: v for k, v in profile.get("screening_answers", {}).items()
                 if not k.startswith("_")}
    if screening.get("preferred_office_location") == "AUTO":
        screening["preferred_office_location"] = state["current_job"]["location"]
    fields.update(screening)
    fields["job_location"] = state["current_job"]["location"]

    # LLM writes ONLY the prose answer; every fact above came from the file.
    job = state["current_job"]
    edu = profile["education"][0]
    why: WhyAnswer = H.call_llm(
        f"Candidate: {ident['full_name']}, {edu['degree']} at {edu['school']}, "
        f"~{yrs} yrs experience.\n"
        f"Role: {job['title']} at {job['company']}.\n"
        "Write a concise 2-3 sentence answer to 'Why are you interested in this role?' "
        "grounded only in a data/ML background. Do not invent company specifics.",
        structured_schema=WhyAnswer, temperature=0.4,
    )
    open_qs = [{"question": "Why are you interested in this role?", "drafted_answer": why.answer}]

    return {"application_fields": fields, "open_questions": open_qs,
            "status": "application package prepared"}


def next_job(state: State):
    """This job scored too low to be worth applying to - move to the next one."""
    jobs = state["jobs"]
    idx = state.get("job_index", 0) + 1
    cov = state.get("best_coverage") or state.get("coverage") or {}
    skipped = list(state.get("skipped_jobs", []))
    skipped.append({"title": state["current_job"]["title"],
                    "company": state["current_job"]["company"],
                    "score": cov.get("score")})
    print(f"  [skip] {state['current_job']['title'][:46]} scored "
          f"{cov.get('score')}/100 (< {S.MIN_SCORE_TO_APPLY}) - trying the next job")

    if idx >= len(jobs):
        return {"skipped_jobs": skipped, "error": "no_suitable_jobs",
                "status": f"all {len(jobs)} jobs scored below "
                          f"{S.MIN_SCORE_TO_APPLY}/100"}

    nxt = jobs[idx]
    print(f"  [next] {nxt['title'][:50]} @ {nxt['company']}")
    return {"job_index": idx, "current_job": nxt,
            "job_description": nxt["description"],
            "attempts": 0, "coverage": None, "skipped_jobs": skipped,
            "tried_project_sets": [], "best_coverage": None,
            "best_resume": None, "best_projects": [], "jd_requirements": [],
            "status": f"moved to {nxt['title']} @ {nxt['company']}"}


def human_review(state: State):
    # Reaching here means you resumed the graph = you approved after review.
    return {"status": "approved by human"}


def track_application(state: State):
    job = state["current_job"]
    scores = {s["resume_id"]: s["score"] for s in state["resume_scores"]}
    from pathlib import Path
    cov = state.get("coverage") or {}
    path = H.log_application(
        company=job["company"], job_title=job["title"],
        resume_used=Path(state["resume_pdf_path"]).name,
        match_score=scores.get(state["selected_resume_id"]),
        coverage=cov.get("score"), attempts=state.get("attempts"),
        job_url=job["url"], status=state.get("submitted_status", "Prepared"),
    )
    return {"status": f"logged to tracker: {path}"}


# ------------------------------------------------------------------ GRAPH --- #
def _after_search(state: State) -> str:
    """Nothing found -> stop cleanly instead of crashing three nodes later."""
    return "end" if state.get("error") == "no_jobs" else "match_resumes"


def _after_grade(state: State) -> str:
    """Explore, then commit to the best combination found."""
    attempts = state.get("attempts", 0)
    best = (state.get("best_coverage") or {}).get("score", 0)

    # A perfect score is the only early exit. Not "good enough" - unbeatable.
    if best >= S.PERFECT_SCORE:
        print(f"  [score] {best}/100 - no combination can beat it, stopping here "
              f"after {attempts} of {S.MAX_ATTEMPTS}")
        return "good_enough"

    if attempts < S.MAX_ATTEMPTS:
        missing = (state.get("coverage") or {}).get("missing", [])
        used = {p["id"] for p in state.get("selected_projects", []) if "id" in p}
        reachable = S.retry_could_help(missing, H.load_experience_pool(), used)
        if reachable:
            print(f"  [score] trying another combination - unused projects "
                  f"cover: {', '.join(reachable[:4])}")
        else:
            print(f"  [score] trying another combination - best so far "
                  f"{best}/100")
        return "retry"

    label = ("at target" if best >= S.TARGET_SCORE
             else f"below the {S.TARGET_SCORE} target")
    if best < S.MIN_SCORE_TO_APPLY:
        return "next_job"

    print(f"  [score] best of {S.MAX_ATTEMPTS} combinations: {best}/100 "
          f"({label}) - applying with that one")
    return "good_enough"


def _after_next_job(state: State) -> str:
    """Ran out of jobs to try, or got a fresh one to score."""
    return "end" if state.get("error") == "no_suitable_jobs" else "select_projects"


def _make_checkpointer():
    """SQLite checkpointing so a crash resumes mid-run."""
    conn = sqlite3.connect(str(H.CHECKPOINT_DB), check_same_thread=False)
    return SqliteSaver(conn)


def build_graph():
    g = StateGraph(State)
    for name, fn in [
        ("search_jobs", search_jobs), ("match_resumes", match_resumes),
        ("select_projects", select_projects), ("tailor_resume", tailor_resume),
        ("grade_resume", grade_resume), ("next_job", next_job),
        ("render_resume", render_resume),
        ("prepare_application", prepare_application),
        ("human_review", human_review), ("track_application", track_application),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "search_jobs")
    g.add_conditional_edges("search_jobs", _after_search,
                            {"match_resumes": "match_resumes", "end": END})
    g.add_edge("match_resumes", "select_projects")
    g.add_edge("select_projects", "tailor_resume")
    g.add_edge("tailor_resume", "grade_resume")

    # CYCLE 1 (exploration): grade -> select_projects -> tailor -> grade
    g.add_conditional_edges("grade_resume", _after_grade,
                            {"retry": "select_projects",
                             "next_job": "next_job",
                             "good_enough": "render_resume"})
    g.add_conditional_edges("next_job", _after_next_job,
                            {"select_projects": "select_projects", "end": END})

    g.add_edge("render_resume", "prepare_application")
    g.add_edge("prepare_application", "human_review")
    g.add_edge("human_review", "track_application")
    g.add_edge("track_application", END)

    # Pause before human_review so you can review the filled form before submit.
    return g.compile(checkpointer=_make_checkpointer(),
                     interrupt_before=["human_review"])
