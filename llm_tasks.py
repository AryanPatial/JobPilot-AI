"""All LLM prompts. Every function returns a validated Pydantic object."""
from __future__ import annotations

import json
import re
from typing import List

from pydantic import BaseModel, Field

import helpers as H

# How many jobs go into ONE judging call
JUDGE_CHUNK = 40

# Stop once we have this many good jobs - no point paying to judge the rest.
ENOUGH_MATCHES = 12

# Every LLM call made this run, so the terminal can show how much of the form was the mode
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
    """'gen AI roles' -> [AI Engineer, LLM Engineer, Applied Scientist,"""
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
    """Judge EVERY candidate for intent fit and seniority fit (spec items 1"""
    kept: list[dict] = []
    if not jobs:
        return kept

    for start in range(0, len(jobs), JUDGE_CHUNK):
        chunk = jobs[start:start + JUDGE_CHUNK]
        listing = "\n".join(
            f"{i}. {j['title']}  [{j.get('location', '')}]  - {_one_line(j)}"
            for i, j in enumerate(chunk))

        _count()
        result: JobVerdicts = H.call_llm(
            "You screen job postings for a candidate.\n\n"
            f"CANDIDATE INTENT: {intent}\n"
            f"CANDIDATE EXPERIENCE: about {years_experience} years "
            "(recent graduate level)\n\n"
            f"JOBS:\n{listing}\n\n"
            "For EVERY job above return a verdict with its index.\n"
            "  matches_intent - does the actual role fit the intent? Reject "
            "postings that merely mention the keywords (a Sales role at an AI "
            "company), and reject non-engineering roles.\n"
            "  seniority_ok  - plausible for this experience level? Reject "
            "Staff, Senior Staff, Principal, Lead, Manager, Head, Director, "
            "Architect, and anything demanding 5+ years. Accept new-grad, "
            "junior, associate, mid-level, 'Senior' only when the posting asks "
            "for 3 years or fewer, and unlabelled roles.",
            structured_schema=JobVerdicts, temperature=0.0)

        for v in result.verdicts:
            if not (0 <= v.index < len(chunk)):
                continue
            if v.matches_intent and v.seniority_ok:
                job = dict(chunk[v.index])
                job["verdict"] = v.reason
                kept.append(job)

        done = min(start + JUDGE_CHUNK, len(jobs))
        print(f"  [judge] {done}/{len(jobs)} screened, {len(kept)} kept")
        if len(kept) >= ENOUGH_MATCHES:
            print(f"  [judge] {ENOUGH_MATCHES} good matches found - stopping early")
            break

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


def _force_new_combination(chosen: list[dict], pool: list[dict],
                           avoid: set[tuple], missing: list[str] | None,
                           n: int) -> list[dict]:
    """Swap projects until the set is one we have not scored before."""
    import scoring as S

    picked_ids = {c["id"] for c in chosen}
    spare = sorted((p for p in pool if p["id"] not in picked_ids),
                   key=lambda p: -S.gap_fit(p, missing or []))

    # Replace the weakest pick first (the model ordered them strongest-first).
    for slot in range(len(chosen) - 1, -1, -1):
        for cand in spare:
            trial = list(chosen)
            trial[slot] = {**cand, "why": "swapped in to try a different angle"}
            if tuple(sorted(c["id"] for c in trial)) not in avoid:
                return trial
    return chosen          # pool exhausted - every combination already tried


def select_content(jd: str, pool: list[dict], *, n: int = 3,
                   missing_skills: list[str] | None = None,
                   avoid: list | None = None) -> list[dict]:
    """Choose which of the candidate's REAL projects belong on this résumé."""
    # Send every bullet in full
    catalogue = "\n".join(
        f"- {p['id']}  |  {p['name']}  |  tags: {', '.join(p.get('tags', []))}\n"
        f"    {' '.join(p['bullets'])}"
        for p in pool)

    avoid_sets = {tuple(sorted(a)) for a in (avoid or [])}

    nudge = ""
    if missing_skills:
        nudge = ("\nThe previous selection did not evidence these JD requirements: "
                 f"{', '.join(missing_skills[:10])}. Prefer projects that genuinely "
                 "demonstrate them. If no project does, pick on overall relevance - "
                 "do NOT stretch a project to claim something it does not show.\n")
    if avoid_sets:
        already = "; ".join(" + ".join(a) for a in sorted(avoid_sets))
        nudge += ("\nThese combinations have already been tried and scored, so "
                  f"return a DIFFERENT set: {already}\n")

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
    chosen = chosen[:n]

    # The model was told what has been tried; it does not always listen.
    if tuple(sorted(c["id"] for c in chosen)) in avoid_sets:
        chosen = _force_new_combination(chosen, pool, avoid_sets, missing_skills, n)
    return chosen


# ====================================================== 4. GRADING ========= #
class Requirement(BaseModel):
    """One thing the JD asks for, and the resume line that proves it."""
    requirement: str = Field(description="What the job asks for, in the JD's own words, "
                                         "under 12 words")
    must_have: bool = Field(description="True if the JD states it as required, "
                                        "False if preferred/nice-to-have")
    evidence: str = Field(default="", description="ONE bullet from the resume, copied "
                                                  "EXACTLY character for character, that "
                                                  "already demonstrates this. Empty string "
                                                  "if no bullet demonstrates it.")


class Requirements(BaseModel):
    requirements: List[Requirement]


def extract_requirements(jd: str) -> list[dict]:
    """What this job requires - derived from the JD ALONE, once per job."""
    _count()
    result: Requirements = H.call_llm(
        "List what this job requires, for scoring résumés against it.\n\n"
        "List 8 to 15 requirements THAT A RESUME CAN DEMONSTRATE: technologies, "
        "technical skills, kinds of work done, scale, domain, years of "
        "experience, education. Use the JD's own language. Mark each as required "
        "or preferred.\n"
        "  * EXCLUDE anything no résumé could ever evidence - willingness to "
        "travel, communication style, cooperation mindset, comfort with "
        "ambiguity, attending conferences, enthusiasm. Listing those makes every "
        "candidate look equally unqualified and tells the reader nothing.\n"
        "  * Split broad responsibilities into the technical capability "
        "underneath: 'advise customers on LLM architecture' -> 'designs LLM "
        "application architecture'.\n"
        "  * Leave every `evidence` field empty. You are not looking at a résumé.\n\n"
        f"JOB DESCRIPTION:\n{jd[:6000]}",
        structured_schema=Requirements, temperature=0.0)
    return [{"requirement": r.requirement, "must_have": r.must_have} for r in result.requirements]


def cite_evidence(requirements: list[dict], resume_text: str) -> list[dict]:
    """For each fixed requirement, quote the résumé line that proves it."""
    listing = "\n".join(f"{i}. [{'required' if r['must_have'] else 'preferred'}] "
                        f"{r['requirement']}" for i, r in enumerate(requirements))
    _count()
    result: Requirements = H.call_llm(
        "For each requirement below, find a résumé line that ALREADY "
        "demonstrates it.\n\n"
        "Return one entry per requirement, in the same order, copying the "
        "requirement text back verbatim. Rules for `evidence`:\n"
        "  * Copy the résumé line EXACTLY, character for character. Do not "
        "paraphrase, shorten, merge or clean it up.\n"
        "  * A line counts only if it genuinely shows the requirement. Related "
        "subject matter is not evidence.\n"
        "  * If nothing in the résumé demonstrates it, leave evidence EMPTY. An "
        "empty evidence field is the correct answer for some requirements - do "
        "not stretch to fill it.\n\n"
        f"REQUIREMENTS:\n{listing}\n\n"
        f"RESUME:\n{resume_text[:6000]}",
        structured_schema=Requirements, temperature=0.0)

    # Trust the fixed list for wording and weight; take only evidence from the model, matched
    cited = {i: r.evidence for i, r in enumerate(result.requirements)}
    by_text = {_norm_req(r.requirement): r.evidence for r in result.requirements}
    out = []
    for i, req in enumerate(requirements):
        ev = by_text.get(_norm_req(req["requirement"]), cited.get(i, ""))
        out.append({**req, "evidence": ev or ""})
    return out


def _norm_req(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


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
    """Everything the model needs to answer a form question truthfully."""
    prof = H.load_profile()
    ident, ans = prof["identity"], prof["application_answers"]

    def one(v):
        return v[0] if isinstance(v, list) else v

    from datetime import date
    today = date.today()

    # The model has no clock, and it is bad at date arithmetic even when given one: told "toda
    def _finished(dates: str) -> bool | None:
        end = (dates or "").split("\u2013")[-1].split("-")[-1].strip()
        if not end or "present" in end.lower():
            return False
        parsed = H._parse_month_year(end)
        if not parsed:
            return None
        year, month = parsed
        return (year, month) <= (today.year, today.month)

    edu_lines = []
    for e in prof.get("education", []):
        done = _finished(e.get("dates", ""))
        state = ("COMPLETED" if done else
                 "IN PROGRESS" if done is False else "dates unclear")
        edu_lines.append(f"    {e['degree']}, {e['school']} "
                         f"({e.get('dates', '')}) - {state}")
    studying = any("IN PROGRESS" in l for l in edu_lines)

    lines = [
        f"== TODAY IS {today:%B %d, %Y} ==",
        "Use this for anything about timing - availability, start dates, how "
        "much experience has accrued.",
        "Education status (already worked out for you - do not recompute):",
        *edu_lines,
        f"    => Currently a student: {'Yes' if studying else 'No'}. "
        f"All degrees finished: {'No' if studying else 'Yes'}.",
        "",
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
        # Omitted once, and the model answered "How did you hear about this job?" with "I found th
        f"How they heard about the job: {one(ans['how_did_you_hear'])}. "
        f"Use this verbatim - do not substitute a job board or referral.",
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

    circ = {k: v for k, v in prof.get("personal_circumstances", {}).items()
            if not k.startswith("_")}
    if circ:
        lines.append("")
        lines.append("== CIRCUMSTANCES (things no resume can show - use these "
                     "verbatim, never guess) ==")
        for k, v in circ.items():
            lines.append(f"    {k.replace('_', ' ')}: {v}")

    lines.append("")
    lines.append("Answer only from the above. If something is not evidenced here, "
                 "say so or decline - never invent a technology, employer, metric "
                 "or credential.")
    return "\n".join(l for l in lines if l is not None)


# ================================================= 6. CIRCUMSTANCE RULES == #
# Questions with a definite stored answer are resolved HERE, from the profile, and never r
CIRCUMSTANCE_RULES: list[tuple[str, str]] = [
    (r"interview.*(this company|with us|here|at \w+)\b.*(before|previously|past|prior)",
     "interviewed_at_this_company_recently"),
    (r"(ever|previously|before).*interview", "interviewed_at_this_company_recently"),
    (r"interview.*(last|past|previous)\s+\d+\s*(month|week)", "interviewed_anywhere_last_3_months"),
    (r"interview.*(recently|elsewhere|another|other)", "interviewed_anywhere_last_3_months"),
    (r"\binterview", "interviewed_anywhere_last_3_months"),      # catch-all
    (r"(previously|ever|before).*appl(y|ied)", "previously_applied_to_this_company"),
    (r"appl(y|ied).*(before|previously|in the past)", "previously_applied_to_this_company"),
    # \b stops this matching inside 'preferred'
    (r"\breferred\b|\bemployee referral\b", "referred_by_an_employee"),
    (r"currently employed|are you employed", "currently_employed"),
    (r"notice period", "notice_period"),
    (r"other offers|competing offers|offers pending", "other_offers_pending"),
    (r"criminal|convicted|felony", "criminal_record"),
    (r"willing to travel|able to travel|can you travel", "can_travel"),
    (r"deadline|timeline consideration", "deadlines_or_timeline_constraints"),
]


def _circumstance_answer(label: str) -> str | None:
    """The stored answer for this question, or None if we have no rule."""
    circ = H.load_profile().get("personal_circumstances", {})
    low = label.lower()
    for pattern, key in CIRCUMSTANCE_RULES:
        if key in circ and re.search(pattern, low):
            return circ[key]
    return None


def _pick_option(options: list[str], answer: str) -> int | None:
    """Match a stored answer onto this form's options, without the model.
    Yes/No is the common case; otherwise fall back to a containment match."""
    want = answer.strip().lower()
    lowered = [o.strip().lower() for o in options]
    for i, o in enumerate(lowered):                      # exact
        if o == want:
            return i
    head = want.split(",")[0].split()[0] if want else ""
    if head in ("yes", "no"):                            # leading token only
        for i, o in enumerate(lowered):
            if o.split()[:1] == [head]:
                return i
    for i, o in enumerate(lowered):                      # containment
        if want in o or o in want:
            return i
    return None


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
    """Answer an ENTIRE application form in one call."""
    if not questions:
        return {}

    # Resolve anything with a stored answer FIRST, deterministically, and drop it from the bat
    out: dict[int, dict] = {}
    remaining = []
    for q in questions:
        stored = _circumstance_answer(q["label"])
        if stored is None:
            remaining.append(q)
            continue
        if q["kind"] == "dropdown":
            idx = _pick_option(q["options"], stored)
            if idx is None:
                remaining.append(q)          # our answer doesn't fit the options
                continue
            out[q["idx"]] = {"option": idx, "text": None, "why": "from profile"}
            print(f"  [rule]  {q['label'][:46]:<48} -> {q['options'][idx][:34]}")
        else:
            out[q["idx"]] = {"option": None, "text": stored, "why": "from profile"}
            print(f"  [rule]  {q['label'][:46]:<48} -> {stored[:34]}")

    questions = remaining
    if not questions:
        return out

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
        "- The CIRCUMSTANCES block answers questions about interviews, "
        "referrals, prior applications, notice period, deadlines and offers. "
        "If a line there answers the question, use EXACTLY that answer. Do not "
        "invert it, soften it, or reason around it: 'referred by an employee: "
        "No' means the answer is No.\n"
        "- Facts come from the record. Never invent an employer, technology, "
        "metric, credential, or personal circumstance.\n"
        "- Skills questions are answerable from the skills and projects listed - "
        "use them, and name the concrete work.\n"
        "- If the same thing is asked twice in different words, answer both "
        "consistently.\n"
        "- Prefer answering over declining, but never guess a fact you were not "
        "given.",
        structured_schema=FormAnswers, temperature=0.2)

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
