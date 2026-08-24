"""Resume/JD scoring and verification. No LLM calls in this file."""
from __future__ import annotations

import json
import re
from pathlib import Path

# Calibrated against 6 real postings x 2 résumé variants (August 2026)
TARGET_SCORE = 60          # what a good tailoring should reach; reported, not a stop
MIN_SCORE_TO_APPLY = 40    # the BEST attempt must clear this or the job is skipped
MAX_ATTEMPTS = 3           # how many DIFFERENT project combinations to try per job
PERFECT_SCORE = 100        # nothing above it exists, so exploring further is pure waste

# The loop does not exit at TARGET_SCORE


# --------------------------------------------------------------------------- #
# Vocabulary: every skill/tool the candidate can legitimately claim.
# --------------------------------------------------------------------------- #
_SPLIT = re.compile(r"[,/|;()]| and ")

# A term earns a place in the vocabulary if it looks like a technology rather than English
_TECHY = re.compile(r"[a-z][A-Z]|[0-9]|-|\.|/")

_STOP = {
    "and", "the", "with", "for", "from", "into", "over", "across", "using", "via",
    "a", "an", "of", "to", "on", "in", "by", "as", "at", "that", "this", "it",
    "be", "or", "may", "will", "can", "are", "is", "was", "were", "has", "have",
    "job", "jobs", "role", "roles", "team", "teams", "work", "working", "time",
    "apply", "annual", "compensation", "benefits", "health", "salary", "equal",
    "requirements", "responsibilities", "qualifications", "experience", "years",
    "new", "real", "full", "end", "each", "all", "per", "its", "their", "our",
    "you", "your", "we", "us", "who", "what", "how", "why", "when", "where",
    "data", "analysis", "service", "services", "system", "systems", "project",
    "projects", "product", "products", "company", "business", "support", "help",
    "built", "build", "designed", "shipped", "engineered", "developed", "added",
    "including", "such", "other", "more", "most", "also", "well", "strong",
}


def _skill_terms(text: str) -> set[str]:
    """Split a curated skills line - 'Python, SQL, CI/CD (GitHub Actions)' -
    into individual claimable terms."""
    out = set()
    for chunk in _SPLIT.split(text):
        chunk = chunk.strip().strip(".").lower()
        if 2 <= len(chunk) <= 34 and chunk not in _STOP and not chunk.isdigit():
            out.add(chunk)
    return out


def _technical_terms(text: str) -> set[str]:
    """Mine prose (bullets, project names) for technology names only, so
    ordinary English never reaches the vocabulary."""
    out = set()
    for tok in re.findall(r"\b[A-Za-z][\w.+#/-]{1,26}\b", text):
        clean = tok.strip(".-/").lower()
        if len(clean) < 2 or clean in _STOP or clean.isdigit():
            continue
        if _TECHY.search(tok) or tok.isupper():      # PySpark, SHA-256, RAG, SQL
            out.add(clean)
    return out


def build_vocabulary(*json_paths: Path) -> set[str]:
    """Every term the candidate may truthfully claim, taken from their own files."""
    vocab: set[str] = set()
    for path in json_paths:
        if not Path(path).exists():
            continue
        blob = json.loads(Path(path).read_text(encoding="utf-8"))

        for resume in ([blob] if "skills" in blob else []):
            skills = resume.get("skills", {})
            for line in (skills.values() if isinstance(skills, dict) else skills):
                vocab |= _skill_terms(str(line))

        def walk(node):
            if isinstance(node, dict):
                for k, v in node.items():
                    if k in ("bullets", "name", "title", "tags"):
                        vocab.update(_technical_terms(" ".join(v) if isinstance(v, list)
                                                      else str(v)))
                    else:
                        walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)
        walk(blob)

    return vocab


def _content_words(phrase: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9+#.-]{3,}", phrase.lower())
            if w not in _STOP}


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def jd_requirements(jd: str, vocab: set[str], *, top: int = 25) -> list[tuple[str, int]]:
    """Terms the JD asks for that the candidate could plausibly have, weighted
    by how often the JD repeats them. Sorted most-important first."""
    low = " " + re.sub(r"\s+", " ", jd.lower()) + " "
    hits: list[tuple[str, int]] = []
    for term in vocab:
        n = low.count(f" {term} ") + low.count(f" {term},") + low.count(f" {term}.")
        if n:
            # Multi-word phrases are stronger signal than single tokens.
            weight = n * (2 if " " in term else 1)
            hits.append((term, weight))
    hits.sort(key=lambda t: (-t[1], t[0]))
    return hits[:top]


def resume_text(resume: dict, *, lower: bool = True) -> str:
    """Flatten a résumé dict to searchable text."""
    parts: list[str] = []
    for job in resume.get("experience", []):
        parts += [job.get("title", ""), job.get("company", "")] + job.get("bullets", [])
    for proj in resume.get("projects", []):
        parts += [proj.get("name", "")] + proj.get("bullets", [])
    skills = resume.get("skills", {})
    parts += list(skills.values()) if isinstance(skills, dict) else list(skills)
    text = " ".join(parts)
    return text.lower() if lower else text


def score_resume(resume: dict, jd: str, vocab: set[str]) -> dict:
    """Weighted keyword coverage, 0-100, plus the named gaps."""
    reqs = jd_requirements(jd, vocab)
    if not reqs:
        return {"score": 100, "covered": [], "missing": [], "requirements": []}

    text = " " + resume_text(resume) + " "
    covered, missing, got, total = [], [], 0, 0
    for term, weight in reqs:
        total += weight
        if term in text:
            got += weight
            covered.append(term)
        else:
            missing.append(term)

    return {
        "score": round(100 * got / total) if total else 100,
        "covered": covered,
        "missing": missing,
        "requirements": [t for t, _ in reqs],
    }


def gap_fit(project: dict, missing: list[str]) -> int:
    """How many of the JD's missing terms this project genuinely evidences."""
    if not missing:
        return 0
    blob = (project["name"] + " " + " ".join(project["bullets"])
            + " " + " ".join(project.get("tags", []))).lower()
    exact = sum(1 for term in missing if term in blob)
    # Requirements arrive as phrases now; fall back to content-word overlap so ranking still w
    return exact or gap_fit_phrase(project, missing)


# --------------------------------------------------------------------------- #
# Verified scoring: the model cites evidence, THIS code decides the number.
# --------------------------------------------------------------------------- #
MIN_EVIDENCE_CHARS = 25    # below this a "quote" is too short to prove anything


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def verify_and_score(graded_text: str, requirements: list[dict]) -> dict:
    """Check every citation against the real resume, then compute the score."""
    if not requirements:
        return {"score": 100, "covered": [], "missing": [], "fabricated": [],
                "requirements": [], "detail": []}

    haystack = _norm(graded_text)
    covered, missing, fabricated, detail = [], [], [], []
    got = total = 0

    for r in requirements:
        weight = 2 if r.get("must_have") else 1
        total += weight
        ev = _norm(r.get("evidence", ""))
        name = r["requirement"]

        if not ev:
            verdict = "unmet"
        elif len(ev) < MIN_EVIDENCE_CHARS:
            verdict = "unmet"          # too short to prove anything
        elif ev in haystack:
            verdict = "met"
        else:
            verdict = "fabricated"

        if verdict == "met":
            got += weight
            covered.append(name)
        else:
            missing.append(name)
            if verdict == "fabricated":
                fabricated.append(name)
        detail.append({"requirement": name, "must_have": r.get("must_have"),
                       "verdict": verdict})

    return {"score": round(100 * got / total) if total else 100,
            "covered": covered, "missing": missing, "fabricated": fabricated,
            "requirements": [r["requirement"] for r in requirements],
            "detail": detail}


def introduced_terms(resume: dict, *source_paths: Path) -> list[str]:
    """Words on the finished résumé that appear in NONE of the candidate's files."""
    source = set()
    for path in source_paths:
        if Path(path).exists():
            source |= _content_words(Path(path).read_text(encoding="utf-8"))
    return sorted(_content_words(resume_text(resume, lower=False)) - source)


def unclaimable_terms(resume: dict, vocab: set[str]) -> list[str]:
    """Technology names on the finished resume that appear nowhere in the"""
    # needs original case
    companies = {c.lower() for j in resume.get("experience", [])
                 for c in re.split(r"[^A-Za-z0-9]+", j.get("company", "")) if c}
    return sorted(_technical_terms(resume_text(resume, lower=False))
                  - vocab - companies)


def gap_fit_phrase(project: dict, missing: list[str]) -> int:
    """How well an unused project speaks to the missing REQUIREMENTS."""
    if not missing:
        return 0
    blob = _content_words(project["name"] + " " + " ".join(project["bullets"])
                          + " " + " ".join(project.get("tags", [])))
    return sum(len(_content_words(m) & blob) for m in missing)


def retry_could_help(missing: list[str], pool: list[dict],
                     current_ids: set[str]) -> list[str]:
    """Which missing terms does some UNUSED pool project actually evidence?"""
    if not missing:
        return []
    reachable = []
    for term in missing:
        for proj in pool:
            if proj["id"] in current_ids:
                continue
            blob = (proj["name"] + " " + " ".join(proj["bullets"])
                    + " " + " ".join(proj.get("tags", []))).lower()
            if term in blob:
                reachable.append(term)
                break
    return reachable
