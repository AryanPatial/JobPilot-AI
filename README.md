# Job Application Agent

A LangGraph agent that finds US data/AI jobs, picks the best-matching résumé,
tailors it to the job description, renders a PDF, **auto-fills the real
application form in a browser**, then stops so a human clicks Submit. Every
application is logged to Excel.

Runs on OpenAI (`gpt-4o-mini`) or Google Gemini — one setting switches between them.

---

## What it actually does

```
run.py  — you give it an INTENT, e.g. "gen AI roles"
  │
  ├─ 1. search_jobs        LLM expands the intent into ~28 real title variants,
  │                        keyword + strict-US prefilter, then ONE batched LLM
  │                        call judges every candidate for intent AND seniority
  ├─ 2. match_resumes      score the two base résumé variants against the JD
  ├─ 3. select_projects    pick 3 real projects from the 12-project pool
  ├─ 4. tailor_resume      reword existing bullets only — never invent
  ├─ 5. grade_resume       deterministic JD coverage, 0-100  (NO LLM)
  │       ├── < target, and an unused project would help → back to 3
  │       └── < 70 after retries → next_job, skip this posting entirely
  ├─ 6. render_resume      JSON + HTML template → Aryan_Patial_<Company>.pdf
  ├─ 7. prepare_application map profile → form fields; compute salary/start/years
  │
  ├─ ⏸  PAUSE  ─────────── apply.py opens Chrome and fills the real form.
  │                        GPT picks each dropdown and writes the open answers.
  │                        YOU review, tick consent, click Submit.
  │
  ├─ 8. human_review       (resumes when you press Enter)
  └─ 9. track_application  append a row to output/applications.xlsx
```

Two cycles: tailoring (capped at 3 attempts) and job-skipping. State is
checkpointed to SQLite, and the pause is a real LangGraph `interrupt_before`.

---

## Setup

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/playwright install chromium
cp .env.example .env
```

Then put an API key in `.env`. The default provider is OpenAI
(`gpt-4o-mini`); set `LLM_PROVIDER=gemini` to use Google instead — that switch
is the only change needed, since every model call goes through `call_llm()`.

WeasyPrint needs system libraries:
`brew install pango` (macOS) · `apt-get install libpango-1.0-0 libpangocairo-1.0-0` (Ubuntu)

Finally, edit `data/candidate_profile.json` — your name, email, and every
application answer live there.

## Run

```bash
./venv/bin/python run.py
```

Edit `INTENT` and `COMPANY_BOARDS` at the top of `run.py`. `INTENT` is plain
English — an LLM expands it into the title variants real postings use.

---

## The files

| File | Contains |
|---|---|
| **run.py** | Entry point. `INTENT`, `COMPANY_BOARDS`, `LLM_FIRST`, `SLOW_MO_MS`. Invokes the graph, hands fields to `apply`, waits on `input()`, resumes to log. |
| **agent.py** | `State` · Pydantic schemas · 10 node functions · `build_graph()` with both cycles, the interrupt and the SQLite checkpointer. |
| **helpers.py** | Tools: provider-agnostic `call_llm()` with model/key failover · Greenhouse search + US filter · salary/experience/date logic · PDF render · Excel tracker. |
| **llm_tasks.py** | Every decision the LLM is allowed to make: `expand_query`, `judge_jobs`, `select_content`, `llm_choose`, `llm_answer`. All batched, all structured output. |
| **scoring.py** | Deterministic JD coverage. No LLM — a model grading its own tailoring flatters itself. |
| **apply.py** | Browser automation. Label→field routing, `_decide()`, react-select driver, résumé upload. Imports only `re`, `time`, `pathlib`. |
| `data/candidate_profile.json` | Identity, `application_answers`, `screening_answers`. Source of truth for every fact. |
| `data/experience_pool.json` | 12 real projects. `select_projects` chooses from these per job. |
| `data/resumes/*.json` | Two base variants (experience + skills). |
| `templates/resume.html` | Jinja + print CSS. Code owns layout. |
| `output/` | PDFs, `applications.xlsx`, `checkpoints.sqlite`, `choices.json`, `last_fill_report.txt`. Gitignored. |

---

## Design decisions

**Facts come from the file; only prose comes from the LLM.**
Work authorization, sponsorship, visa status, EEO answers — all read from
`candidate_profile.json`. The model writes exactly one thing: the "why this
role" paragraph. It never touches a factual field, so it cannot hallucinate one.

**The LLM edits content; code owns formatting.**
Résumés are JSON. A fixed HTML template renders them. Tailoring can reword a
bullet but can never change the layout.

**The form matcher refuses to guess on legal questions.**
The profile says *"I am not a protected veteran."* Robinhood's form offers
*"I identify as a protected veteran"* and *"I have never served in the
military."* Naive substring matching would select the opposite and put a false
statement on a legal document. `_choose_option()` is negation-aware: it compares
polarity, and when more than one candidate survives it returns `None` and leaves
the field blank for the human. A blank there is a safety feature, not a bug.

**429 and 503 are different failures and get different fixes.**
A 429 is *this key's* daily quota → rotate to the next API key immediately (no
backoff; quota doesn't clear for 24h). A 503 is *this model* being overloaded →
rotate to the next model, since every key hits the same overloaded model.
`call_llm()` walks `models × keys`; `invoke_llm()` handles transient backoff
within one pair.

**Every network call has a timeout.**
Without one, a stalled TCP connection blocks forever inside `SSL_read` and no
exception is ever raised — so retry logic never fires. Observed live: a run sat
at 0% CPU for five minutes. `LLM_TIMEOUT_SECONDS = 150`.

**US-only, strictly.**
`is_us_location()` drops anything ambiguous. It handles the cases that break
naive filters: `"Bengaluru, IN"` is India not Indiana, `"Berlin, DE"` is Germany
not Delaware, `"Toronto, CA"` is Canada not California — while `"Manchester, NH"`
and `"Paris, TX"` correctly stay. A multi-country posting like
`"New York, NY or London, UK"` is dropped as ambiguous.

**It never clicks Submit.**
It fills everything it confidently can and stops. Checkboxes and radios are
never touched, so the consent box is always yours to tick.

---

## Known limits

- **Only hosted Greenhouse forms can be auto-filled.** If a job's
  `absolute_url` is on `greenhouse.io` / `job-boards.greenhouse.io`, it works.
  Companies that redirect to their own careers site (Databricks, Stripe) open
  in the browser but the fields won't match.
- **Only the first 40 candidates are judged** (`JUDGE_BATCH_LIMIT`). On a wide
  search the rest are silently dropped.
- **Multi-page applications aren't handled.** One page only — Workday-style
  flows need the page-navigation loop that isn't built yet.
- **No automated tests.** Verification has been manual throughout.
- **The pool is the ceiling.** `select_projects` can only surface what's in
  `experience_pool.json`; if a JD wants BigQuery and no project shows it, the
  score stays low and the job gets skipped rather than the gap being invented.

---

## Rebuilding from scratch

Dependency order — each layer only needs what's above it, so every step is
testable the moment you write it:

```
1. requirements.txt, .gitignore, .env       nothing runs without these
2. data/candidate_profile.json              your facts
3. data/resumes/*.json                      résumé content
4. templates/resume.html                    layout (needs the JSON shape)
5. helpers.py                               tools (needs data + template)
6. agent.py                                 state → schemas → nodes → graph
7. apply.py                                 browser (independent — any time)
8. run.py                                   entry point (needs agent + apply)
```

Two things worth preserving if you rebuild:

- **Write nodes before the graph.** You can't wire steps together until the
  steps exist, and steps need their tools first.
- **Keep `apply.py` decoupled.** It imports only `re`, `time`, and `playwright`
  — no project modules. It receives a finished dict and knows nothing about
  profiles, LangGraph, or Gemini. That boundary is why the browser layer can be
  tested and rewritten on its own.
