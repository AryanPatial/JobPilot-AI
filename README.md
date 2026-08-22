# Job Application Agent

A LangGraph agent that finds US data/AI jobs, picks the best-matching résumé,
tailors it to the job description, renders a PDF, **auto-fills the real
application form in a browser**, then stops so a human clicks Submit. Every
application is logged to Excel.

Runs on Google Gemini's free tier.

---

## What it actually does

```
run.py
  │
  ├─ 1. search_jobs         Greenhouse API → title match → strict US-only filter
  ├─ 2. match_resumes       score both résumé variants against the JD (LLM)
  ├─ 3. tailor_resume       reword existing bullets only — never invent (LLM)
  ├─ 4. render_resume       JSON + HTML template → Aryan_Patial_<Company>.pdf
  ├─ 5. prepare_application map profile → form fields; compute salary/start/years
  │
  ├─ ⏸  PAUSE  ──────────── apply.py opens Chrome and fills the real form
  │                          YOU review, tick consent, click Submit
  │
  ├─ 6. human_review        (resumes when you press Enter)
  └─ 7. track_application   append a row to output/applications.xlsx
```

One job per run. The graph pauses via LangGraph's `interrupt_before`, and state
is checkpointed to SQLite so a crash resumes instead of starting over.

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

Edit `COMPANY_BOARDS` and `TARGET_TITLES` at the top of `run.py` to change what
it searches.

---

## The files

| File | Lines | Contains |
|---|---|---|
| **run.py** | 94 | Entry point + the only config you edit. Invokes the graph, hands fields to `apply`, waits on `input()`, resumes the graph to log. |
| **agent.py** | 242 | `State` TypedDict · 4 Pydantic output schemas · the 7 node functions · `build_graph()` with the interrupt and SQLite checkpointer. |
| **helpers.py** | 471 | Every tool, in 5 sections: LLM factory + failover · Greenhouse search + US filter · salary/experience/date logic · PDF render · Excel tracker. |
| **apply.py** | 490 | Browser automation. Label→field routing, the option matcher, react-select combobox driver, résumé upload. Imports nothing from this project. |
| `data/candidate_profile.json` | — | Identity, `application_answers`, `screening_answers`. The source of truth for every fact. |
| `data/resumes/*.json` | — | Two résumé variants as structured content, not PDFs. |
| `templates/resume.html` | — | Jinja + print CSS. Code owns layout so the LLM can't wreck it. |
| `output/` | — | Generated PDFs, `applications.xlsx`, `checkpoints.sqlite`. Gitignored. |

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
  Companies that redirect to their own careers site (Databricks, Stripe) will
  open in the browser but the fields won't match.
- **Same job every run.** `search_jobs` takes `jobs[0]`, so re-running the same
  boards re-processes the same posting. Nothing reads the tracker to skip it.
- **Free-tier Gemini is unreliable at peak.** 20 requests/day/key and frequent
  503s during US afternoons. One run costs ~4 calls. Multiple keys can be listed
  in `GOOGLE_API_KEYS` (comma-separated) and are rotated on quota exhaustion.
- **No automated tests.** Verification has been manual.
- **Company-specific answer text.** `screening_answers.worked_here_before` names
  a company. The matcher's "never" rule generalises it to other employers, but
  the stored string is literal.

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
