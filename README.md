# Job Application Agent

A LangGraph agent that finds US data/AI jobs, picks your best-matching resume,
tailors it to the job, generates a PDF, **auto-fills the application form**, then
pauses so **you** click Submit. Logs every application to Excel. Runs free on
Google Gemini.

## The whole project is 4 Python files

| File | What it is | "Where does X happen?" |
|------|------------|------------------------|
| **run.py** | Start button | Edit which companies + titles to search |
| **agent.py** | The brain | The state, the pipeline steps (nodes), the graph |
| **helpers.py** | The tools | Job search, US-only filter, salary/experience logic, PDF render, Excel |
| **apply.py** | The applying | Playwright fills the real form, stops at Submit |

Plus `data/` (your profile + two resumes), `templates/` (resume look).

## Flow

```
run.py → agent.build_graph()
   search_jobs      (helpers: Greenhouse API, US-only)
   match_resumes    (score both resumes vs JD, pick best)
   tailor_resume    (light honest edits to content only)
   render_resume    (helpers: content + template → PDF)
   prepare_application (map fields; salary from JD; start date + experience auto)
   ── PAUSE ──  apply.py fills the browser → you review + Submit
   track_application (helpers: append row to Excel)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Win: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env          # paste your free Gemini key (aistudio.google.com)
```

Then open `data/candidate_profile.json` and replace the test email at the top.
Your work-authorization / sponsorship / relocation answers are already filled in.

## Run

```bash
python run.py
```

Edit `COMPANY_BOARDS` and `TARGET_TITLES` at the top of `run.py` first.

## Known limits (verified live on 2026-08-19)

- **Only hosted Greenhouse forms can be auto-filled.** Verified: `robinhood` and
  `gitlab` serve real forms on `job-boards.greenhouse.io`. `databricks` redirects
  to its own careers site (no form — 0 fields fillable), and `airtable`/`discord`
  had no open US data/AI roles. `COMPANY_BOARDS` is set to the two that work.
- **Two fields still need you.** On a live Robinhood form 14 of 16 filled
  automatically. The city field is an async geocode autocomplete (we type it; you
  pick the suggestion), and veteran status is *deliberately* left blank — see below.
- **The matcher refuses to guess on attested questions.** Your profile says
  "I am not a protected veteran"; the form offers "I identify as a protected
  veteran" and "I have never served in the military". Since picking wrong would
  put a false statement on a legal form, `_choose_option()` is negation-aware and
  returns nothing when ambiguous. That blank is a safety feature, not a bug.
- **Same job every run.** `search_jobs` takes `jobs[0]`, so re-running the same
  boards re-processes the same posting. Nothing reads the tracker to skip it yet.
- **Gemini free tier: 20 requests/day/model, ~4 per run — about 5 runs a day.**
  Budget your rehearsal and recording takes accordingly. Transient 503s now retry
  automatically with backoff instead of killing the run.
- **US filter is list-based.** `is_us_location()` disambiguates "Berlin, DE" from
  "Dover, DE" using a list of major non-US cities. Add any hub your boards post from.

## What it does NOT do

It does not click Submit. It fills everything and stops so you do the final click
yourself (keeps you compliant with job-board terms; forms vary too much to trust a
blind auto-submit). Every tracker row says `Prepared`.

## Key design choices

- **Facts come from your file, prose comes from the LLM.** Work auth, sponsorship,
  salary, etc. are read from `candidate_profile.json` — the LLM never guesses them.
  It only writes the "why this role" paragraph.
- **The LLM edits content; code owns formatting.** Resumes are JSON; a fixed HTML
  template renders them, so tailoring can't wreck your layout.
- **US-only, strict.** `is_us_location()` keeps a job only if it names a US state,
  state abbreviation, or "US"/"USA"/"United States". Everything else is dropped.
- **Salary aims high.** If the JD lists a range, expectation = top of range; else
  "Negotiable".

## Notes

- WeasyPrint needs system libs on some OSes (macOS: `brew install pango`; Ubuntu:
  `sudo apt-get install libpango-1.0-0 libpangocairo-1.0-0`).
- First live run on a new company may skip a custom field — the terminal prints
  what it filled vs skipped. `apply.py` matches fields by their **label text**
  (see `FIELD_PATTERNS`), so adding a company's wording is a one-line change.
- Stripe's careers page is custom (not a hosted Greenhouse form) and is the hardest
  to auto-fill — start testing with a company whose `absolute_url` is on `greenhouse.io`.
