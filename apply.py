"""
apply.py  —  The applying part. Opens the real application form in a browser,
fills every field it can from your prepared data, uploads your tailored PDF,
then STOPS at the submit button and hands control to you.

It never clicks submit. That's on purpose: you do the final click, which keeps
this on the right side of job-board rules and lets you eyeball everything first.

How it finds fields
-------------------
Greenhouse has two form flavours (classic `boards.greenhouse.io` and the newer
React `job-boards.greenhouse.io`), and every company adds its own custom
questions. So rather than hard-coding one company's ids, we:

  1. ask the page for EVERY input/select/textarea plus its visible label text,
  2. match each label against the patterns in FIELD_PATTERNS below,
  3. fill text inputs directly, and pick the closest option for <select>s.

Anything we can't confidently match is printed as "skipped" so you can set it
during review. Custom React comboboxes (a button + popup listbox instead of a
real <select>) are reported, not guessed at.
"""
from __future__ import annotations
import re


# --------------------------------------------------------------------------- #
# Which prepared field goes with which question label.
# Order matters: the first pattern that matches a label wins that control.
# --------------------------------------------------------------------------- #
FIELD_PATTERNS: list[tuple[str, list[str]]] = [
    ("first_name",           [r"^first\s*name", r"\bgiven name\b"]),
    ("last_name",            [r"^last\s*name", r"\bfamily name\b", r"\bsurname\b"]),
    ("email",                [r"^e-?mail"]),
    ("phone",                [r"^phone", r"\bmobile\b", r"\btelephone\b"]),
    ("location",             [r"^location", r"\bcity\b", r"where are you (currently )?based"]),
    ("country",              [r"^country"]),
    ("linkedin",             [r"linked\s*-?in"]),
    ("github",               [r"git\s*hub", r"\bportfolio\b", r"personal website", r"^website$"]),
    ("work_authorization",   [r"legally authorized", r"authorized to work", r"work authorization",
                              r"eligible to work"]),
    ("requires_sponsorship", [r"sponsor", r"visa support", r"immigration support"]),
    ("visa_status",          [r"visa status", r"current visa", r"work permit"]),
    ("willing_to_relocate",  [r"relocat"]),
    ("salary_expectation",   [r"salary", r"compensation expectation", r"expected pay", r"desired pay"]),
    ("earliest_start_date",  [r"start date", r"when (can|could) you start", r"availab"]),
    ("years_experience",     [r"years of (relevant )?experience", r"how many years"]),
    ("gender",               [r"^gender", r"gender identity"]),
    ("race_ethnicity",       [r"race", r"ethnic"]),
    ("military_status",      [r"military status", r"military service"]),
    ("veteran_status",       [r"veteran"]),
    ("pronouns",             [r"pronoun"]),
    ("lgbtq_status",         [r"lgbtq", r"\blgbt\b"]),
    ("used_product",         [r"have you used\b", r"are you a (customer|user) of"]),
    ("worked_here_before",   [r"have you ever worked (for|at)\b",
                              r"as an employee,? intern,? or contractor"]),
    ("preferred_office_location", [r"preferred office location", r"which office",
                                   r"office location"]),
    ("relationships_disclosure", [r"personal[/ ]?familial", r"familial relation",
                                  r"outside business", r"business activit",
                                  r"intellectual property ownership"]),
    ("government_official",  [r"government official", r"public official",
                              r"bribery", r"corruption risk"]),
    ("willing_to_work_from_office", [r"willing to work from the office",
                                     r"\bonsite\b", r"\bon-site\b"]),
    ("disability_status",    [r"disab"]),
    ("how_did_you_hear",     [r"how did you (hear|find)", r"referral source"]),
]

# Labels we never want to touch (site search boxes, consent checkboxes, etc.)
IGNORE_LABEL = re.compile(r"search|newsletter|subscribe|password|confirm email", re.I)

# Free-text follow-ups to the compliance questions. Since the answers below are
# "No", these boxes should stay empty — and a model/regex should never invent
# prose for a legal attestation. Consent checkboxes are separately never ticked
# (the fill loop skips every checkbox and radio unconditionally).
NEVER_ANSWER = re.compile(
    # Free-text follow-ups to compliance questions
    r"if you answered|please provide additional information|"
    r"please (explain|describe|elaborate)|"
    # Anything the candidate would be legally BOUND by. GPT agreed to a binding
    # arbitration clause on a live Anthropic form before this was added; a
    # signature or a waiver of rights is never the model's to give.
    r"arbitrat|agree to the terms|terms and conditions|"
    r"i (agree|consent|acknowledge|understand and agree)|"
    r"waiv(e|er)|binding|legally bound|"
    r"electronic signature|e-signature|\bsign\b|initials?\b|"
    r"do you certify|i certify|i attest|under penalty of perjury|"
    r"background check|drug (test|screen)|authorize .* to (verify|contact)",
    re.I,
)


# --------------------------------------------------------------------------- #
# Page introspection: every control on the page, with its human-readable label.
# --------------------------------------------------------------------------- #
_SCAN_JS = """
() => {
  const out = [];
  const ctrls = document.querySelectorAll('input, select, textarea');
  ctrls.forEach((el, i) => {
    el.setAttribute('data-jaa', String(i));
    let label = '';
    if (el.id) {
      const l = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
      if (l) label = l.innerText;
    }
    if (!label) { const l = el.closest('label'); if (l) label = l.innerText; }
    if (!label && el.getAttribute('aria-labelledby')) {
      const l = document.getElementById(el.getAttribute('aria-labelledby'));
      if (l) label = l.innerText;
    }
    if (!label) label = el.getAttribute('aria-label') || el.placeholder || el.name || '';
    const style = window.getComputedStyle(el);
    out.push({
      idx: i,
      tag: el.tagName.toLowerCase(),
      type: (el.type || '').toLowerCase(),
      label: label.replace(/\\s+/g, ' ').trim().slice(0, 120),
      options: el.tagName === 'SELECT' ? Array.from(el.options).map(o => o.text.trim()) : [],
      combo: el.getAttribute('role') === 'combobox'
             || (el.className || '').toString().includes('select__input'),
      hidden: style.display === 'none' || style.visibility === 'hidden' || el.type === 'hidden',
    });
  });
  return out;
}
"""


def _scan_controls(page) -> list[dict]:
    try:
        return page.evaluate(_SCAN_JS)
    except Exception:
        return []


def _match_key(label: str) -> str | None:
    """Which prepared field does this label want? None if we can't tell, or if
    it's a legal attestation we refuse to answer on the candidate's behalf."""
    if not label or IGNORE_LABEL.search(label) or NEVER_ANSWER.search(label):
        return None
    low = label.lower()
    for key, patterns in FIELD_PATTERNS:
        for pat in patterns:
            if re.search(pat, low):
                return key
    return None


# --------------------------------------------------------------------------- #
# Filling
# --------------------------------------------------------------------------- #
def _fill_text(page, idx: int, value) -> bool:
    try:
        page.fill(f"[data-jaa='{idx}']", str(value))
        return True
    except Exception:
        return False


# Greenhouse's newer forms render dropdowns as react-select: a text input with
# role="combobox" plus a popup menu of divs. There is no <select> to set, so we
# have to click it open, read the menu, and click the right option.
_SCOPED_OPTIONS_JS = """(el) => {
  document.querySelectorAll('[data-jaaopt]').forEach(o => o.removeAttribute('data-jaaopt'));
  let n = el;
  while (n && !n.querySelector('[class*="select__menu"]')) n = n.parentElement;
  if (!n) return [];
  const opts = n.querySelectorAll('[class*="select__option"]');
  return Array.from(opts).map((o, i) => { o.setAttribute('data-jaaopt', String(i));
                                          return o.innerText.trim(); });
}"""


def _read_options(page, el):
    """Read the options of THIS combobox's own menu, without touching it.
    Scoped via the input's ancestor chain so we never pick up the phone
    country-code list, which is always present elsewhere in the DOM."""
    try:
        return page.evaluate(_SCOPED_OPTIONS_JS, el)
    except Exception:
        return []


def _combobox_options(page, idx: int):
    """Click the combobox open and return (element, its own options)."""
    try:
        el = page.query_selector(f"[data-jaa='{idx}']")
        if not el:
            return None, []
        el.click()
        page.wait_for_timeout(700)
        return el, _read_options(page, el)
    except Exception:
        return None, []


from pathlib import Path

# Where the per-run fill report is written, so the terminal scrolling away
# doesn't lose it. Kept as a literal path to preserve this module's rule of
# importing nothing from the rest of the project.
_REPORT_PATH = Path(__file__).resolve().parent / "output" / "last_fill_report.txt"


# How many times to re-scan the page. Conditional questions appear only after
# an earlier answer is given, so one scan can never see the whole form.
MAX_SWEEPS = 3


# Typed straight from candidate_profile.json — no model needed to enter a
# phone number. Everything else on the page goes to the whole-form call.
DIRECT_FIELDS = {"first_name", "last_name", "email", "phone",
                 "linkedin", "github", "years_experience"}


def _upload_resume(page, resume_path) -> bool:
    try:
        for el in page.query_selector_all("input[type='file']"):
            el.set_input_files(resume_path)   # first file input = resume
            return True
    except Exception:
        pass
    return False


def fill_greenhouse_form(url: str, fields: dict, open_questions: list,
                         chooser=None, answerer=None, slow_mo: int = 0,
                         llm_first: bool = False, form_answerer=None) -> dict:
    """Open the form, fill what we can, upload the resume, and leave the browser
    OPEN for you to review + submit. Returns a report of filled/skipped fields.

    chooser:  callback (label, options) -> index|None, consulted ONLY when the
              deterministic rules find no confident match, and never for a
              field in LLM_FORBIDDEN.
    answerer: callback (label) -> text|None, for free-text questions the rules
              have no value for. Never called for a NEVER_ANSWER label.
    slow_mo:  milliseconds to pause between browser actions, so you can watch
              each field being filled instead of the form snapping to done.
    form_answerer: callback (questions) -> {idx: {option|text}}. Given every
              remaining question on the page WITH its real options, in one
              call, it returns what to put in each. This is the main path:
              the model sees the whole form at once, and the page is then
              filled in a single top-to-bottom pass.
    llm_first / chooser / answerer: retained for the older per-field path.

    Both are passed in rather than imported, which keeps this module free of any
    LLM dependency - it still imports nothing from the rest of the project.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {"error": "Playwright not installed. Run: pip install playwright && playwright install chromium"}

    pw = sync_playwright().start()
    try:
        browser = pw.chromium.launch(headless=False, slow_mo=slow_mo)  # visible
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded")
        page.wait_for_timeout(2500)                    # let React forms mount
    except Exception as exc:
        pw.stop()
        return {"error": f"Could not open the form: {exc}"}

    try:
        return _fill(page, fields, open_questions, chooser, answerer,
                     url, browser, pw, llm_first, form_answerer)
    except Exception as exc:
        # Whatever went wrong, the browser MUST stay open and the caller MUST
        # get a handle back - otherwise run.py dies here and the window closes
        # before the human ever sees it.
        import traceback
        print(f"\n[apply] fill failed part-way: {type(exc).__name__}: {str(exc)[:160]}")
        print("[apply] the browser is still open — finish the form by hand.")
        traceback.print_exc()
        return {"filled": [], "skipped": ["<fill crashed>"], "needs_confirm": [],
                "untouched": [], "browser": browser, "pw": pw,
                "crashed": f"{type(exc).__name__}: {exc}"}


def collect_questions(page) -> list[dict]:
    """Everything on the page a human would have to answer.

    Walks the DOM in order and returns one entry per still-empty control,
    including a dropdown's REAL options (read by opening it). Legally binding
    items are dropped here so they are never shown to the model at all.
    """
    out = []
    for ctrl in _scan_controls(page):
        label = (ctrl["label"] or "").strip()
        if (ctrl["hidden"] or not label or len(label) < 6
                or ctrl["type"] in ("hidden", "submit", "button", "file")
                or IGNORE_LABEL.search(label)):
            continue
        if ctrl["type"] in ("checkbox", "radio"):
            continue                       # never auto-ticked, so never asked
        if NEVER_ANSWER.search(label):
            continue                       # arbitration, signatures, consent

        if ctrl.get("combo"):
            if _combobox_value(page, ctrl["idx"]):
                continue                   # already answered
            _, options = _combobox_options(page, ctrl["idx"])
            try:
                page.keyboard.press("Escape")
            except Exception:
                pass
            if not options:
                continue
            out.append({"idx": ctrl["idx"], "label": label,
                        "kind": "dropdown", "options": options})
        elif ctrl["tag"] == "select":
            out.append({"idx": ctrl["idx"], "label": label,
                        "kind": "dropdown", "options": ctrl["options"]})
        else:
            try:
                el = page.query_selector(f"[data-jaa='{ctrl['idx']}']")
                if el is None or (el.input_value() or "").strip():
                    continue
            except Exception:
                continue
            out.append({"idx": ctrl["idx"], "label": label,
                        "kind": "text", "options": []})
    return out


def _combobox_value(page, idx: int) -> str:
    """What a react-select box currently shows, if anything."""
    try:
        return page.evaluate(
            """(i)=>{const e=document.querySelector(`[data-jaa="${i}"]`);
               const c=e&&e.closest('[class*=select__control]');
               const v=c&&c.querySelector('[class*=single-value]');
               return v?v.innerText.trim():''}""", idx) or ""
    except Exception:
        return ""


def _fill(page, fields, open_questions, chooser, answerer, url, browser, pw,
          llm_first=False, form_answerer=None) -> dict:
    """Fill the form in ONE ordered pass.

    1. Known facts straight from the profile (name, email, phone, links).
    2. Everything else: collect every remaining question with its real
       options, hand the whole lot to the model in a single call, then fill
       the page top to bottom with what comes back.
    """
    filled: list[str] = []
    skipped: list[str] = []
    untouched: list[tuple] = []

    # 1) Resume upload first — some forms re-render (and clear) after parsing it.
    if fields.get("resume_file"):
        (filled if _upload_resume(page, fields["resume_file"]) else skipped).append("resume_file")
        page.wait_for_timeout(2000)

    # 2) Plain identity text boxes. No model needed to type a phone number.
    for ctrl in _scan_controls(page):
        if ctrl["hidden"] or ctrl.get("combo") or ctrl["tag"] != "input":
            continue
        if ctrl["type"] in ("file", "hidden", "submit", "button", "checkbox", "radio"):
            continue
        key = _match_key(ctrl["label"])
        if key not in DIRECT_FIELDS:
            continue
        value = fields.get(key)
        if value in (None, ""):
            continue
        value = value[0] if isinstance(value, (list, tuple)) else value
        if _fill_text(page, ctrl["idx"], value):
            print(f"    OK  {ctrl['label'][:40]:<42}{str(value)[:30]:<32}profile")
            filled.append(f"{key} ({ctrl['label'][:38]})")

    # 3) Collect -> one call -> fill, repeated until nothing new appears.
    #    Some questions are conditional: "Please identify your race" only exists
    #    once "Are you Hispanic/Latino?" has been answered, so a single scan can
    #    never see them. Capped at MAX_SWEEPS so this can't spin.
    untouched_by_label: dict[str, str] = {}
    seen_labels: set[str] = set()

    for sweep in range(MAX_SWEEPS):
        questions = [q for q in collect_questions(page)
                     if q["label"] not in seen_labels]
        if not questions or not form_answerer:
            break
        if sweep:
            print(f"\n  sweep {sweep + 1}: {len(questions)} newly-revealed "
                  f"question(s)")
        seen_labels.update(q["label"] for q in questions)
        answers = form_answerer(questions)
        _apply_answers(page, questions, answers, filled, untouched_by_label)

    untouched = list(untouched_by_label.items())
    return _finish(url, browser, pw, filled, skipped, untouched)


def _apply_answers(page, questions, answers, filled, untouched_by_label):
    """Write one batch of answers onto the page, in DOM order."""
    for q in questions:
        a = answers.get(q["idx"])
        if not a:
            untouched_by_label[q["label"][:58]] = "model returned no answer"
            continue
        try:
            if q["kind"] == "dropdown":
                el, opts = _combobox_options(page, q["idx"])
                if not opts:
                    page.select_option(f"[data-jaa='{q['idx']}']",
                                       label=q["options"][a["option"]])
                else:
                    target = q["options"][a["option"]]
                    hit = opts.index(target) if target in opts else a["option"]
                    page.click(f"[data-jaaopt='{hit}']")
                page.wait_for_timeout(250)
                chosen = q["options"][a["option"]]
            else:
                page.fill(f"[data-jaa='{q['idx']}']", a["text"])
                chosen = a["text"][:44]
            print(f"    OK  {q['label'][:40]:<42}{str(chosen)[:30]:<32}gpt")
            filled.append(f"{q['label'][:40]} -> {str(chosen)[:24]} (gpt)")
            untouched_by_label.pop(q["label"][:58], None)
        except Exception as exc:
            untouched_by_label[q["label"][:58]] = f"could not set it: {type(exc).__name__}"


def _finish(url, browser, pw, filled, skipped, untouched):
    """Report what happened and hand back the still-open browser."""
    for lab, why in untouched:
        skipped.append(f"{lab} — {why}")
    if untouched:
        print(f"\n  LEFT EMPTY ({len(untouched)}):")
        for lab, why in untouched:
            print(f"    ·  {lab:<60} {why}")
    print(f"\n  Filled {len(filled)} fields.")
    print("[apply] The submit button is NOT clicked. That part is yours.")

    try:
        rep = ["FORM FILL REPORT", url, "", f"FILLED ({len(filled)}):"]
        rep += [f"  + {x}" for x in filled]
        rep += ["", f"LEFT EMPTY ({len(untouched)}):"]
        rep += [f"  - {lab}\n      {why}" for lab, why in untouched]
        _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _REPORT_PATH.write_text("\n".join(rep), encoding="utf-8")
        print(f"[apply] Full report saved to {_REPORT_PATH}")
    except Exception:
        pass

    return {"filled": filled, "skipped": skipped, "untouched": untouched,
            "browser": browser, "pw": pw}

