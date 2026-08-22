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


# ============================================================= 4. CHOOSE ==== #
class OptionChoice(BaseModel):
    index: int = Field(description="0-based index of the chosen option, or -1 if none fit")
    why: str = Field(description="Under 10 words")


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


def llm_choose(label: str, options: list[str], profile_summary: str,
               intended: str | None = None) -> int | None:
    """Ask the model which option to select.

    `intended` is the value from candidate_profile.json for this field, when we
    have one. Passing it in is what keeps this safe: the model is matching a
    known fact to the form's wording, not deciding the fact. Given
    "I am not a protected veteran" it picks "I have never served in the
    military"; it is never left to guess whether the candidate is a veteran.

    Returns an INDEX, so there is no fuzzy string matching on the way back and
    only a real option can ever be selected. -1 / None means leave it blank.
    """
    if not options:
        return None
    key = json.dumps([label.strip().lower(), options, intended], sort_keys=True)
    cache = _cache()
    if key in cache:
        return cache[key]

    listing = "\n".join(f"{i}. {o}" for i, o in enumerate(options))
    known = (f"\nTHE CANDIDATE'S ANSWER TO THIS, FROM THEIR PROFILE: {intended!r}\n"
             "Select the option that expresses exactly this. Do not substitute a "
             "different meaning, and never pick an option that contradicts it.\n"
             if intended else "")
    try:
        _count()
        result: OptionChoice = H.call_llm(
            "Choose the dropdown option that best answers a job-application question "
            "for this candidate.\n\n"
            f"CANDIDATE:\n{profile_summary}\n\n"
            f"QUESTION: {label}\n{known}\nOPTIONS:\n{listing}\n\n"
            "Reply with the index number only. If no option is truthful for this "
            "candidate, return -1 — a blank field is always better than a wrong "
            "answer on a job application.",
            structured_schema=OptionChoice, temperature=0.0)
    except Exception as exc:
        print(f"  [chooser] LLM unavailable, leaving blank: {str(exc)[:70]}")
        return None

    idx = result.index if 0 <= result.index < len(options) else None
    cache[key] = idx
    try:
        _CHOICE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CHOICE_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass
    if idx is not None:
        print(f"  [chooser] {label[:44]!r} -> {options[idx]!r} ({result.why})")
    return idx


def profile_summary() -> str:
    """A short, factual description of the candidate for the dropdown chooser.
    Facts only, straight from the profile - no inference, no prose."""
    prof = H.load_profile()
    ident, ans = prof["identity"], prof["application_answers"]
    edu = prof["education"][0]
    def one(v):
        return v[0] if isinstance(v, list) else v
    return (
        f"{ident['full_name']}, based in {ident['location']}. "
        f"{edu['degree']}, {edu['school']}. "
        f"Work authorized in the US: {ans['work_authorization_us']}. "
        f"Requires visa sponsorship: {ans['requires_sponsorship_now_or_future']}. "
        f"Visa status: {ans['visa_status']}. "
        f"Willing to relocate: {ans['willing_to_relocate']}. "
        f"Gender: {one(ans['gender'])}. Race/ethnicity: {one(ans['race_ethnicity'])}. "
        f"Not a veteran. No disability."
    )


# ============================================================= 5. ANSWER ==== #
class WrittenAnswer(BaseModel):
    answer: str = Field(description="The answer text. Empty string if it cannot "
                                    "be answered truthfully from the profile.")
    confident: bool = Field(description="False if this needs the human")


def llm_answer(label: str, profile_summary: str, job_context: str = "",
               *, max_words: int = 90) -> str | None:
    """Write a free-text application answer on the candidate's behalf.

    Used for open questions the rules have no value for - "why this role",
    "describe a project", "what interests you about us". Never used for legal
    attestations: apply.NEVER_ANSWER filters those out before we get here, and
    apply.LLM_FORBIDDEN keeps the attested dropdowns rules-only.

    Grounded strictly in the profile summary and the JD. Returns None rather
    than guessing when the question needs a fact we don't hold.
    """
    key = json.dumps(["ANSWER", label.strip().lower(), job_context[:60]], sort_keys=True)
    cache = _cache()
    if key in cache:
        return cache[key]

    try:
        _count()
        result: WrittenAnswer = H.call_llm(
            "Write a job-application answer for this candidate. Truthful, specific, "
            "first person, no hype, no invented facts.\n\n"
            f"CANDIDATE:\n{profile_summary}\n\n"
            f"ROLE CONTEXT: {job_context[:600]}\n\n"
            f"QUESTION: {label}\n\n"
            f"Keep it under {max_words} words. Ground every claim in the candidate "
            "details above - never invent an employer, a metric, a technology, or a "
            "personal circumstance. If the question asks for something not present "
            "in the candidate details (a salary figure, a date, a reference, a "
            "legal declaration), set confident=false and return an empty answer.",
            structured_schema=WrittenAnswer, temperature=0.3)
    except Exception as exc:
        print(f"  [answer] LLM unavailable: {str(exc)[:70]}")
        return None

    text = (result.answer or "").strip()
    out = text if (result.confident and text) else None
    cache[key] = out
    try:
        _CHOICE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CHOICE_CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass
    if out:
        print(f"  [answer] {label[:44]!r} -> {out[:60]}...")
    return out
