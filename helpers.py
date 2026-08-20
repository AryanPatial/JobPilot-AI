"""
helpers.py  —  All the "tools" the agent uses, in one place.

Sections:
  1. Config & paths (+ Gemini model factory)
  2. Job search   (Greenhouse API + strict US-only location filter)
  3. Small brains (salary from JD, years of experience, start date)
  4. Resume PDF   (JSON content + HTML template -> PDF)
  5. Tracker      (append a row to Excel)

If you want to know "where does X happen?", it's one of these five sections.
"""
from __future__ import annotations
import html
import json
import re
from datetime import date, datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

# ========================================================================== #
# 1. CONFIG & PATHS
# ========================================================================== #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESUMES_DIR = DATA_DIR / "resumes"
TEMPLATES_DIR = ROOT / "templates"
OUTPUT_DIR = ROOT / "output"
OUTPUT_RESUMES_DIR = OUTPUT_DIR / "resumes"
TRACKER_PATH = OUTPUT_DIR / "applications.xlsx"
CHECKPOINT_DB = OUTPUT_DIR / "checkpoints.sqlite"
PROFILE_PATH = DATA_DIR / "candidate_profile.json"

OUTPUT_RESUMES_DIR.mkdir(parents=True, exist_ok=True)

import os
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def get_llm(temperature: float = 0.3, structured_schema=None):
    """One place to get the LLM. Swap providers here and nowhere else."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=temperature)
    return llm.with_structured_output(structured_schema) if structured_schema else llm


def invoke_llm(llm, prompt, *, attempts: int = 5, base_delay: float = 4.0):
    """Call the model, retrying transient failures.

    Gemini's free tier returns 503 UNAVAILABLE ("high demand") fairly often and
    429 RESOURCE_EXHAUSTED once you pass the daily quota. Without this, one blip
    kills the whole pipeline mid-run with a stack trace. Retries use exponential
    backoff; a genuine quota exhaustion is re-raised with a clear message.
    """
    import time
    last = None
    for i in range(attempts):
        try:
            return llm.invoke(prompt)
        except Exception as exc:
            last = exc
            msg = str(exc)
            transient = ("503" in msg or "UNAVAILABLE" in msg or "high demand" in msg
                         or "429" in msg or "RESOURCE_EXHAUSTED" in msg
                         or "deadline" in msg.lower() or "timeout" in msg.lower())
            if not transient or i == attempts - 1:
                break
            delay = base_delay * (2 ** i)
            print(f"  [llm] {msg[:70]}... retrying in {delay:.0f}s "
                  f"({i + 1}/{attempts - 1})")
            time.sleep(delay)
    if "RESOURCE_EXHAUSTED" in str(last) or "429" in str(last):
        raise RuntimeError(
            "Gemini daily free-tier quota is exhausted (20 requests/day/model). "
            "One pipeline run costs ~4 calls. Wait for the quota to reset."
        ) from last
    raise last


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_profile() -> dict:
    return load_json(PROFILE_PATH)


def load_resume_variants() -> list[dict]:
    return [load_json(p) for p in sorted(RESUMES_DIR.glob("*.json"))]


# ========================================================================== #
# 2. JOB SEARCH  (Greenhouse API + strict US-only filter)
# ========================================================================== #
US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine",
    "maryland", "massachusetts", "michigan", "minnesota", "mississippi",
    "missouri", "montana", "nebraska", "nevada", "new hampshire", "new jersey",
    "new mexico", "new york", "north carolina", "north dakota", "ohio",
    "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina",
    "south dakota", "tennessee", "texas", "utah", "vermont", "virginia",
    "washington", "west virginia", "wisconsin", "wyoming",
}
US_STATE_ABBR = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id",
    "il", "in", "ia", "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms",
    "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok",
    "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv",
    "wi", "wy", "dc",
}
US_MARKERS = {"us", "u.s.", "u.s.a.", "usa", "united states", "united states of america"}

# Non-US countries whose names (or 2-letter codes) collide with US state
# abbreviations: "Bengaluru, IN" is India, not Indiana; "Berlin, DE" is Germany,
# not Delaware; "Toronto, CA" is Canada, not California. Any hit here = drop.
NON_US_COUNTRIES = {
    "canada", "mexico", "india", "germany", "france", "spain", "portugal",
    "italy", "netherlands", "belgium", "switzerland", "austria", "poland",
    "czechia", "czech republic", "romania", "bulgaria", "greece", "turkey",
    "sweden", "norway", "denmark", "finland", "iceland", "ireland", "scotland",
    "wales", "england", "united kingdom", "u.k.", "great britain",
    "israel", "united arab emirates", "uae", "dubai", "saudi arabia", "qatar",
    "egypt", "nigeria", "kenya", "south africa", "ghana", "morocco",
    "china", "japan", "south korea", "korea", "singapore", "malaysia",
    "indonesia", "thailand", "vietnam", "philippines", "taiwan", "hong kong",
    "australia", "new zealand", "argentina", "brazil", "chile", "colombia",
    "peru", "uruguay", "costa rica", "panama", "guatemala", "moldova", "ukraine",
    "serbia", "croatia", "slovakia", "slovenia", "hungary", "lithuania",
    "latvia", "estonia", "luxembourg", "malta", "cyprus", "armenia", "georgia (country)",
}
# Major non-US cities. Needed because some ISO country codes ARE US state
# abbreviations: "Bengaluru, IN" (India/Indiana), "Berlin, DE" (Germany/Delaware),
# "Toronto, CA" (Canada/California), "Tel Aviv, IL" (Israel/Illinois). The city
# name disambiguates. Not exhaustive — add any hub your boards actually post.
NON_US_CITIES = {
    "toronto", "vancouver", "montreal", "ottawa", "calgary", "waterloo",
    "london", "manchester", "edinburgh", "dublin", "belfast", "cambridge, uk",
    "berlin", "munich", "hamburg", "frankfurt", "cologne", "paris", "lyon",
    "madrid", "barcelona", "lisbon", "porto", "milan", "rome", "amsterdam",
    "rotterdam", "brussels", "zurich", "geneva", "vienna", "prague", "warsaw",
    "krakow", "bucharest", "budapest", "sofia", "athens", "istanbul",
    "stockholm", "oslo", "copenhagen", "helsinki", "tallinn", "riga", "vilnius",
    "tel aviv", "jerusalem", "haifa", "dubai", "abu dhabi", "doha", "riyadh",
    "cairo", "lagos", "nairobi", "cape town", "johannesburg",
    "bengaluru", "bangalore", "hyderabad", "mumbai", "pune", "chennai",
    "gurugram", "gurgaon", "noida", "new delhi", "delhi", "kolkata",
    "beijing", "shanghai", "shenzhen", "tokyo", "osaka", "seoul", "singapore",
    "kuala lumpur", "jakarta", "bangkok", "hanoi", "ho chi minh", "manila",
    "taipei", "hong kong", "sydney", "melbourne", "brisbane", "perth",
    "auckland", "wellington", "sao paulo", "rio de janeiro", "buenos aires",
    "santiago", "bogota", "lima", "montevideo", "san jose, costa rica",
    "mexico city", "guadalajara", "monterrey", "panama city", "chisinau", "kyiv",
}

# US state abbreviations that are ALSO ISO country codes. Only for these does
# the city name need to break the tie — "Manchester, NH" and "Paris, TX" stay
# US because NH and TX are unambiguous.
AMBIGUOUS_ABBR = {"al", "ar", "ca", "co", "de", "ga", "id", "il", "in", "ky",
                  "la", "ma", "md", "me", "mn", "mo", "ms", "mt", "nc", "ne",
                  "pa", "sc", "sd", "tn", "va"}

# Bare 2-letter country codes that are NOT US states, but commonly appear.
NON_US_CODES = {"uk", "gb", "eu", "ie", "fr", "es", "pt", "nl", "be", "ch", "at",
                "pl", "cz", "ro", "gr", "tr", "dk", "fi", "no", "se", "br", "cn",
                "jp", "kr", "sg", "my", "th", "vn", "ph", "tw", "hk", "au", "nz",
                "ae", "za", "cr", "mx"}


def is_us_location(location: str) -> bool:
    """STRICT US-only. True only if the location clearly resolves to the US.

    Order matters:
      1. Any non-US country named/coded anywhere -> drop (a "New York or London"
         posting is ambiguous, and we drop ambiguous).
      2. An explicit US marker or a full state name -> keep.
      3. A state abbreviation -> keep, unless that abbreviation is also a country
         code AND the city is a known non-US hub ("Berlin, DE", "Toronto, CA").
    Bare 'Remote', 'n/a', and empty all return False.
    """
    if not location:
        return False
    text = location.lower().strip()
    tokens = set(re.split(r"[^a-z.]+", text)) - {""}

    # 1. Unambiguous non-US signal.
    for country in NON_US_COUNTRIES:
        if country in text:
            return False
    if tokens & NON_US_CODES:
        return False

    # 2. Unambiguous US signal.
    if tokens & {"us", "usa", "u.s.", "u.s.a."} or "united states" in text:
        return True
    for state in US_STATES:
        if state in text:
            return True

    # 3. State abbreviation as its own token, so "or" (Oregon) doesn't match
    #    inside "coordinator" and "in" doesn't match inside "engineering".
    hits = tokens & US_STATE_ABBR
    if not hits:
        return False
    if hits - AMBIGUOUS_ABBR:
        return True                      # e.g. NY, TX, WA — nothing to confuse
    return not any(city in text for city in NON_US_CITIES)


def search_jobs_greenhouse(board_tokens: list[str], titles: list[str]) -> list[dict]:
    """Fetch open roles from each company's Greenhouse board, keep only US roles
    whose title matches one of my target titles."""
    wanted = [t.lower() for t in titles]
    results: list[dict] = []

    for token in board_tokens:
        url = f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs"
        try:
            resp = requests.get(url, params={"content": "true"}, timeout=30)
            resp.raise_for_status()
        except Exception as exc:
            print(f"  [skip] {token}: {exc}")
            continue

        for j in resp.json().get("jobs", []):
            title = j.get("title", "")
            loc = (j.get("location") or {}).get("name", "")

            if wanted and not any(w in title.lower() for w in wanted):
                continue
            if not is_us_location(loc):
                continue

            results.append({
                "job_id": str(j.get("id")),
                "title": title,
                "company": j.get("company_name") or token,
                "location": loc,
                "url": j.get("absolute_url"),
                "description": _strip_html(j.get("content", "")),
            })
    return results


def _strip_html(raw: str) -> str:
    if not raw:
        return ""
    text = html.unescape(raw)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ========================================================================== #
# 3. SMALL BRAINS  (salary from JD, years of experience, start date)
# ========================================================================== #
def salary_expectation_from_jd(jd: str) -> str:
    """If the JD lists a pay range, aim at the TOP of it. Otherwise 'Negotiable'."""
    if not jd:
        return "Negotiable"
    # Find dollar amounts like $120,000 or $120K or $120k
    amounts = []
    for m in re.finditer(r"\$\s?(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?\s?[kK])", jd):
        raw = m.group(1).replace(",", "").lower().replace(" ", "")
        if raw.endswith("k"):
            amounts.append(int(float(raw[:-1]) * 1000))
        else:
            try:
                amounts.append(int(float(raw)))
            except ValueError:
                pass
    plausible = [a for a in amounts if 30000 <= a <= 800000]
    if plausible:
        top = max(plausible)
        return f"${top:,}"
    return "Negotiable"


_MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"], start=1)}


def _parse_month_year(text: str):
    """'August 2025' / 'Aug 2025' / 'Present' / '2025' -> (year, month)."""
    text = text.strip().lower()
    if "present" in text or "current" in text:
        today = date.today()
        return today.year, today.month
    m = re.search(r"([a-z]{3,})\s+(\d{4})", text)
    if m:
        prefix = m.group(1)[:3]
        for name, num in _MONTHS.items():
            if name.startswith(prefix):
                return int(m.group(2)), num
        return None
    m = re.search(r"\b(\d{4})\b", text)   # bare year -> assume January
    return (int(m.group(1)), 1) if m else None


def years_of_experience(resume: dict) -> float:
    """Sum the duration of each WORK stint in the resume (not study time)."""
    total_months = 0
    for job in resume.get("experience", []):
        dates = job.get("dates", "")
        parts = re.split(r"[–—\-]", dates)
        if len(parts) != 2:
            continue
        start = _parse_month_year(parts[0])
        end = _parse_month_year(parts[1])
        if not start or not end:
            continue
        months = (end[0] - start[0]) * 12 + (end[1] - start[1])
        if months > 0:
            total_months += months
    return round(total_months / 12, 1)


def earliest_start_date() -> str:
    return f"Available immediately (as of {date.today().strftime('%B %d, %Y')})"


# ========================================================================== #
# 4. RESUME PDF  (content + template -> PDF; code owns formatting)
# ========================================================================== #
def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def render_resume_pdf(resume_content: dict, company: str) -> str:
    from jinja2 import Template
    profile = load_profile()
    ident = profile["identity"]
    ctx = {
        "name": ident["full_name"], "location": ident["location"],
        "phone": ident["phone"], "email": ident["email"],
        "linkedin": ident["linkedin"], "github": ident["github"],
        "education": profile["education"],
        "experience": resume_content["experience"],
        "projects": resume_content["projects"],
        "skills": resume_content["skills"],
    }
    tmpl = Template((TEMPLATES_DIR / "resume.html").read_text(encoding="utf-8"))
    html_str = tmpl.render(**ctx)

    out_path = OUTPUT_RESUMES_DIR / f"Aryan_Patial_{_safe(company)}.pdf"
    from weasyprint import HTML   # lazy import; rest of app runs without it
    HTML(string=html_str).write_pdf(str(out_path))
    return str(out_path)


# ========================================================================== #
# 5. TRACKER  (append one row per application to Excel)
# ========================================================================== #
_HEADERS = ["Company", "Job Title", "Date", "Time", "Resume Used",
            "Match Score", "Job URL", "Status"]


def log_application(*, company, job_title, resume_used, match_score, job_url, status="Prepared"):
    from openpyxl import Workbook, load_workbook
    if TRACKER_PATH.exists():
        wb = load_workbook(TRACKER_PATH)
    else:
        wb = Workbook()
        wb.active.title = "Applications"
        wb["Applications"].append(_HEADERS)
    ws = wb["Applications"]
    now = datetime.now()
    ws.append([company, job_title, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"),
               resume_used, match_score, job_url, status])
    wb.save(TRACKER_PATH)
    return str(TRACKER_PATH)
