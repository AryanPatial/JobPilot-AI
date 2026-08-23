"""
llm_tasks.py  —  Every decision the LLM is allowed to make, in one place.

Rules that apply to everything in this file:

  * Batch, never loop. One call judging 40 jobs beats 40 calls judging one.
  * Structured output only. Every function returns a validated Pydantic object.
  * The LLM decides *relevance and wording*. It never decides a fact about the
    candidate - those come from candidate_profile.json via helpers.
  * Legal / EEO / attested answers never reach this file at all.

helpers.call_llm() is the single provider swap point; nothing here knows or
cares whether it's talking to GPT or Gemini.
"""
from __future__ import annotations

import json
from typing import List

from pydantic import BaseModel, Field

import helpers as H

# How many pre-filtered jobs we send to the judge in one call.
JUDGE_BATCH_LIMIT = 40

# Every LLM call made this run, so the terminal can show how much of the form
# was the model and how much was deterministic rules.
CALLS = 0


def _count():
    global CALLS
    CALLS += 1


# ============================================================== 1. QUERY ==== #
class ExpandedQuery(BaseModel):
    """Title variants for a plain-English intent like 'gen AI roles'."""
    titles: List[str] = Field(
        description="Job title keywords/variants, deliberately over-inclusive")
    reasoning: str = Field(description="One sentence on the interpretation")


def expand_query(intent: str, *, limit: int = 28) -> list[str]:
    """'gen AI roles' -> [AI Engineer, LLM Engineer, Applied Scientist,
    Member of Technical Staff, Forward Deployed Engineer, ...]

    Deliberately broad: this is a keyword pre-filter, and a false positive here
    is cheap (the judge in step 2 removes it) while a false negative is not -
    a title we never fetch can never be recovered.
    """
    _count()
    result: ExpandedQuery = H.call_llm(
        "You expand a job-search intent into the title keywords that real postings "
        "actually use. You know the industry's non-obvious names for roles.\n\n"
        f"INTENT: {intent}\n\n"
        f"Return up to {limit} title keywords or short phrases.\n\n"
        "RULES\n"
        "- Individual-contributor and hands-on roles only. Exclude Manager, Director, "
        "Ethics, Policy, Program/Project Manager, QA, Sales, Marketing, Recruiter.\n"
        "- Include the industry's INSIDER names, not just the obvious ones. For AI/ML "
        "that means things like: Member of Technical Staff, Forward Deployed Engineer, "
        "Applied Scientist, Research Engineer, Agentic Engineer, Solutions Engineer, "
        "MTS. Generic lists that miss these are failures.\n"
        "- Include adjacent seniorities (Associate, New Grad, I, II) but not Staff/"
        "Principal/Lead.\n"
        "- Prefer short fragments that survive substring matching: 'Machine Learning' "
        "beats 'Machine Learning Engineer II'.\n"
        "- Be over-inclusive on genuinely adjacent engineering roles. A false positive "
        "is cheap - a later step filters it. A title never fetched is lost forever.\n"
        "- No duplicates, no companies, no locations.",
        structured_schema=ExpandedQuery, temperature=0.4)

    seen, out = set(), []
    for t in result.titles:
        t = t.strip()
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out[:limit]


# ============================================================== 2. JUDGE ==== #
class JobVerdict(BaseModel):
    """One job's verdict. index refers to the numbered list in the prompt."""
    index: int = Field(description="The job's number from the list")
    matches_intent: bool = Field(description="Does the role genuinely fit the intent?")
    seniority_ok: bool = Field(
        description="Is it plausible for the candidate's years of experience? "
                    "False for Staff/Principal/Director or '5+ years' roles.")
    reason: str = Field(description="Under 15 words")


class JobVerdicts(BaseModel):
    verdicts: List[JobVerdict]


def judge_jobs(intent: str, jobs: list[dict], years_experience: float) -> list[dict]:
    """ONE call that judges every candidate job for both intent fit and
    seniority fit (spec items 1 and 3, batched together).

    Keyword matching cannot see that a "Staff ML Engineer, 8+ years" posting is
    a bad use of an application when you have ~2 years. This can.

    Returns the input jobs, each with `verdict` attached, keeping only those the
    model accepted on both axes.
    """
    if not jobs:
        return []
    batch = jobs[:JUDGE_BATCH_LIMIT]

    listing = "\n".join(
        f"{i}. {j['title']}  [{j.get('location', '')}]  — {_one_line(j)}"
        for i, j in enumerate(batch))

    _count()
    result: JobVerdicts = H.call_llm(
        "You screen job postings for a candidate.\n\n"
        f"CANDIDATE INTENT: {intent}\n"
        f"CANDIDATE EXPERIENCE: about {years_experience} years (recent graduate level)\n\n"
        f"JOBS:\n{listing}\n\n"
        "For EVERY job above return a verdict with its index.\n"
        "  matches_intent — does the actual role fit the intent? Reject postings that "
        "merely mention the keywords (e.g. a Sales role at an AI company).\n"
        "  seniority_ok  — plausible for this experience level? Reject Staff, Principal, "
        "Lead, Director, Manager, and anything demanding 5+ years. Accept new-grad, "
        "junior, associate, mid-level, and unlabelled roles.",
        structured_schema=JobVerdicts, temperature=0.0)

    kept = []
    for v in result.verdicts:
        if not (0 <= v.index < len(batch)):
            continue
        if v.matches_intent and v.seniority_ok:
            job = dict(batch[v.index])
            job["verdict"] = v.reason
            kept.append(job)
    return kept


def _one_line(job: dict, chars: int = 220) -> str:
    """A short slice of the JD so the judge sees more than a title."""
    return " ".join((job.get("description") or "").split())[:chars]


# ============================================================= 3. SELECT ==== #
class ProjectPick(BaseModel):
    project_id: str = Field(description="id from the pool, exactly as given")
    why: str = Field(description="Under 12 words: why this fits the JD")


class ContentSelection(BaseModel):
    picks: List[ProjectPick] = Field(description="Best projects, strongest first")


def select_content(jd: str, pool: list[dict], *, n: int = 3,
                   missing_skills: list[str] | None = None) -> list[dict]:
    """Choose which of the candidate's REAL projects belong on this résumé.

    This is selection, not generation - the crucial distinction. The retry loop
    calls this again with `missing_skills` from the deterministic scorer, so the
    way to raise the score is to swap in a different true project, never to
    invent a skill. That keeps the "never invent" rule structurally enforced
    rather than merely requested in a prompt.
    """
    catalogue = "\n".join(
        f"- {p['id']}  |  {p['name']}  |  tags: {', '.join(p.get('tags', []))}\n"
        f"    {' '.join(p['bullets'])[:260]}"
        for p in pool)

    nudge = ""
    if missing_skills:
        nudge = ("\nThe previous selection did not evidence these JD requirements: "
                 f"{', '.join(missing_skills[:10])}. Prefer projects that genuinely "
                 "demonstrate them. If no project does, pick on overall relevance - "
                 "do NOT stretch a project to claim something it does not show.\n")

    _count()
    result: ContentSelection = H.call_llm(
        "Pick which of this candidate's real projects to put on a résumé for one job.\n\n"
        f"JOB DESCRIPTION:\n{jd[:3500]}\n\n"
        f"PROJECT POOL:\n{catalogue}\n{nudge}\n"
        f"Choose exactly {n}, strongest first. Use project_id values verbatim. "
        "Favour direct relevance to the JD's actual work over impressiveness, and "
        "avoid two near-duplicate projects when one would do.",
        structured_schema=ContentSelection, temperature=0.2)

    by_id = {p["id"]: p for p in pool}
    chosen: list[dict] = []
    for pick in result.picks:
        proj = by_id.get(pick.project_id)
        if proj and proj not in chosen:
            chosen.append({**proj, "why": pick.why})
    # Never return fewer than asked: top up in pool order if the model dropped one.
    for p in pool:
        if len(chosen) >= n:
            break
        if p not in chosen and p["id"] not in {c["id"] for c in chosen}:
            chosen.append({**p, "why": "filler - model returned too few picks"})
    return chosen[:n]


_CHOICE_CACHE_PATH = H.OUTPUT_DIR / "choices.json"
_choice_cache: dict | None = None


def _cache() -> dict:
    global _choice_cache
    if _choice_cache is None:
        try:
            _choice_cache = json.loads(_CHOICE_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            _choice_cache = {}
    return _choice_cache


def profile_summary(resume: dict | None = None) -> str:
    """Everything the model needs to answer a form question truthfully.

    Two halves, and the split matters:

      FACTS      - identity, work authorisation, EEO answers. Straight from
                   candidate_profile.json, never inferred.
      EVIDENCE   - education, employment history, projects and skills, from the
                   resume variants and the experience pool.

    The evidence half is what lets "Do you have expertise coding in Python?" be
    answered from the actual skills list rather than guessed, and stops the
    model claiming a technology that appears nowhere in the candidate's work.
    """
    prof = H.load_profile()
    ident, ans = prof["identity"], prof["application_answers"]

    def one(v):
        return v[0] if isinstance(v, list) else v

    lines = [
        "== FACTS (from the candidate's profile - authoritative) ==",
        f"Name: {ident['full_name']}. Location: {ident['location']}.",
        f"Email: {ident['email']}. Phone: {ident['phone']}.",
        f"LinkedIn: {ident['linkedin']}  GitHub: {ident['github']}",
        f"Work authorized in the US: {ans['work_authorization_us']}. "
        f"Requires visa sponsorship: {ans['requires_sponsorship_now_or_future']}. "
        f"Visa status: {ans['visa_status']}.",
        f"Willing to relocate: {ans['willing_to_relocate']}. "
        f"Gender: {one(ans['gender'])}. Race/ethnicity: {one(ans['race_ethnicity'])}. "
        f"Not a veteran. No disability.",
    ]

    for e in prof.get("education", []):
        lines.append(f"Education: {e['degree']}, {e['school']} ({e['dates']})"
                     + (f", {e['detail']}" if e.get("detail") else ""))

    variants = H.load_resume_variants()
    if resume is None and variants:
        resume = variants[0]

    lines.append("")
    lines.append("== EVIDENCE (the candidate's real work - answer FROM this) ==")

    if resume:
        lines.append(f"Years of professional experience: {H.years_of_experience(resume)}")
        for job in resume.get("experience", []):
            lines.append(f"- {job['title']}, {job['company']} ({job['dates']})")
            for b in job.get("bullets", []):
                lines.append(f"    * {b}")

    skills: dict[str, str] = {}
    for v in variants:
        sk = v.get("skills", {})
        if isinstance(sk, dict):
            skills.update(sk)
    if skills:
        lines.append("Skills:")
        for cat, val in skills.items():
            lines.append(f"    {cat}: {val}")

    pool = H.load_experience_pool()
    if pool:
        lines.append("Projects:")
        for proj in pool:
            lines.append(f"    - {proj['name']}: {' '.join(proj['bullets'])[:200]}")

    lines.append("")
    lines.append("Answer only from the above. If something is not evidenced here, "
                 "say so or decline - never invent a technology, employer, metric "
                 "or credential.")
    return "\n".join(lines)


# =========================================================== 6. WHOLE FORM == #
class FormAnswer(BaseModel):
    """One answer, tied back to the question's number in the prompt."""
    q: int = Field(description="The question number from the list")
    option: int = Field(default=-1, description="For a dropdown: 0-based index of "
                                                "the chosen option. -1 if none fit.")
    text: str = Field(default="", description="For a text box: what to write. "
                                              "Empty if it cannot be answered.")
    why: str = Field(default="", description="Under 10 words")


class FormAnswers(BaseModel):
    answers: List[FormAnswer]


def answer_form(questions: list[dict], profile: str, job_context: str) -> dict[int, dict]:
    """Answer an ENTIRE application form in one call.

    `questions` is what Playwright actually found on the page:
        [{"idx": 9, "label": "...", "kind": "dropdown", "options": [...]}, ...]

    One call rather than one per box, for three reasons: the model sees the
    whole form at once (so it notices the same question asked twice in
    different words), it costs a fraction as much, and the caller can then
    fill the page in a single top-to-bottom pass.

    Returns {idx: {"option": int|None, "text": str|None, "why": str}}.
    Anything legally binding is filtered out by the caller before it gets here.
    """
    if not questions:
        return {}

    lines = []
    for n, q in enumerate(questions):
        if q["kind"] == "dropdown":
            opts = "  ".join(f"[{i}] {o}" for i, o in enumerate(q["options"]))
            lines.append(f"{n}. ({q['kind']}) {q['label']}\n     OPTIONS: {opts}")
        else:
            lines.append(f"{n}. ({q['kind']}) {q['label']}")

    _count()
    result: FormAnswers = H.call_llm(
        "Fill in a job application for this candidate.\n\n"
        f"CANDIDATE RECORD (authoritative - never contradict it):\n{profile}\n\n"
        f"ROLE: {job_context[:600]}\n\n"
        f"QUESTIONS:\n" + "\n".join(lines) + "\n\n"
        "Answer EVERY question, returning its number in `q`.\n"
        "  dropdown -> set `option` to the index of the best option. Use -1 only "
        "if no option is truthful for this candidate.\n"
        "  text     -> set `text` to what should be written, first person, "
        "specific, under 90 words. Optional or open-ended boxes ('Additional "
        "Information', 'Personal Preferences', 'anything else we should know') "
        "still get a real answer drawn from the projects and skills - they are "
        "a chance to say something useful, not a box to skip. Leave `text` empty "
        "ONLY if answering needs a fact the record does not contain, such as a "
        "past salary figure or a reference's contact details.\n\n"
        "Rules:\n"
        "- Facts come from the record. Never invent an employer, technology, "
        "metric, credential, or personal circumstance.\n"
        "- Skills questions are answerable from the skills and projects listed - "
        "use them, and name the concrete work.\n"
        "- If the same thing is asked twice in different words, answer both "
        "consistently.\n"
        "- Prefer answering over declining, but never guess a fact you were not "
        "given.",
        structured_schema=FormAnswers, temperature=0.2)

    out: dict[int, dict] = {}
    for a in result.answers:
        if not (0 <= a.q < len(questions)):
            continue
        q = questions[a.q]
        if q["kind"] == "dropdown":
            if 0 <= a.option < len(q["options"]):
                out[q["idx"]] = {"option": a.option, "text": None, "why": a.why}
        elif (a.text or "").strip():
            out[q["idx"]] = {"option": None, "text": a.text.strip(), "why": a.why}
    return out
