"""
scoring.py  —  Deterministic résumé/JD coverage score. NO LLM.

Why deterministic: the retry loop in the graph uses this score to decide whether
to try again. If the LLM graded its own tailoring it would flatter itself and the
loop would exit immediately at a fake 95.

Why the vocabulary is built from the CANDIDATE'S OWN material: a naive keyword
scorer rewards stuffing JD terms into bullets, and a retry loop that feeds back
"you're missing Kubernetes" actively pushes the model to invent Kubernetes
experience. Here a term only counts if it appears somewhere in the candidate's
real pool or résumé variants, so the score can never be raised by inventing
something - only by surfacing something true that was left out.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

TARGET_SCORE = 85          # tailoring loop exits at or above this
MIN_SCORE_TO_APPLY = 70    # below this the job is a bad fit - skip it entirely
MAX_ATTEMPTS = 3           # hard cap - never loop unbounded


# --------------------------------------------------------------------------- #
# Vocabulary: every skill/tool the candidate can legitimately claim.
# --------------------------------------------------------------------------- #
_SPLIT = re.compile(r"[,/|;()]| and ")

# A term earns a place in the vocabulary if it looks like a technology rather
# than English: an internal capital (PySpark, FastAPI, XGBoost), a digit or
# hyphen (K-Means, SHA-256, PR-AUC), or a dot (scikit-learn is caught by the
# hyphen rule; Node.js by this one).
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
    """Every term the candidate may truthfully claim, taken from their own files.

    The curated `skills` sections are the authority. Bullets and project names
    contribute only things that look like technologies, never plain English -
    otherwise the JD's boilerplate ("teams", "compensation") scores as a skill.
    """
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


def resume_text(resume: dict) -> str:
    """Flatten a résumé dict to searchable text."""
    parts: list[str] = []
    for job in resume.get("experience", []):
        parts += [job.get("title", ""), job.get("company", "")] + job.get("bullets", [])
    for proj in resume.get("projects", []):
        parts += [proj.get("name", "")] + proj.get("bullets", [])
    skills = resume.get("skills", {})
    parts += list(skills.values()) if isinstance(skills, dict) else list(skills)
    return " ".join(parts).lower()


def score_resume(resume: dict, jd: str, vocab: set[str]) -> dict:
    """Weighted keyword coverage, 0-100, plus the named gaps.

    Returns {score, covered, missing, requirements} — `missing` is what the
    retry loop feeds back so selection can surface a better real project.
    """
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


def retry_could_help(missing: list[str], pool: list[dict],
                     current_ids: set[str]) -> list[str]:
    """Which missing terms does some UNUSED pool project actually evidence?

    Without this the retry loop spends its whole budget rediscovering that the
    candidate simply doesn't have BigQuery. Retrying is only worthwhile when a
    real project we didn't pick would close a real gap - anything else would
    require inventing, which is the one thing the system must never do.
    """
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
