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
import time


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
    r"if you answered|please provide additional information|"
    r"please (explain|describe|elaborate)|"
    r"do you certify|i certify|i attest",
    re.I,
)

# "Prefer not to say" shows up under many different wordings.
DECLINE_WORDS = ["decline", "prefer not", "do not wish", "dont wish", "not to disclose",
                 "prefer to not", "choose not", "wish to answer", "rather not"]

_CONTRACTIONS = [("don't", "do not"), ("doesn't", "does not"), ("didn't", "did not"),
                 ("isn't", "is not"), ("aren't", "are not"), ("wasn't", "was not"),
                 ("haven't", "have not"), ("hasn't", "has not"), ("won't", "will not"),
                 ("i'm", "i am"), ("i've", "i have")]

_NEGATIVE = re.compile(r"\bnot\b|\bnever\b|\bno\b|\bnon-|\bnone\b")


def _norm(text: str) -> str:
    """Lowercase, expand contractions, drop punctuation. So 'No, I don't have a
    disability' and 'No, I do not have a disability' compare equal."""
    t = str(text).lower().strip()
    for a, b in _CONTRACTIONS:
        t = t.replace(a, b)
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


_STATE_BY_ABBR = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island",
    "sc": "south carolina", "sd": "south dakota", "tn": "tennessee", "tx": "texas",
    "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming",
    "dc": "district of columbia",
}


def _expand_state(normalised: str) -> str:
    """'dallas tx' -> 'dallas texas', so a city value matches a form option
    that spells the state out in full."""
    parts = normalised.split()
    if parts and parts[-1] in _STATE_BY_ABBR:
        return " ".join(parts[:-1] + [_STATE_BY_ABBR[parts[-1]]])
    return normalised


def _is_negative(text: str) -> bool:
    return bool(_NEGATIVE.search(_norm(text)))


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


def _choose_option(options: list[str], desired) -> str | None:
    """Accepts a string, or a list meaning 'try these in order, first match
    wins' (e.g. ["South Asian", "Asian"] for forms that only offer the broader
    category)."""
    if isinstance(desired, (list, tuple)):
        for candidate in desired:
            hit = _choose_one_option(options, candidate)
            if hit is not None:
                return hit
        return None
    return _choose_one_option(options, desired)


def _choose_one_option(options: list[str], desired: str) -> str | None:
    """Pick the option that genuinely expresses `desired`. Returns None rather
    than guessing.

    This runs on legally-attested questions (work authorization, veteran status,
    disability). Picking the *opposite* option would put a false statement on a
    real application, so anything ambiguous is left for the human to answer.
    """
    if not desired:
        return None
    want = _norm(desired)
    real = [o for o in options
            if o.strip() and not re.match(r"^(select|choose|--)", o.strip(), re.I)]
    if not want or not real:
        return None

    # 1. Exact match after normalising contractions/punctuation.
    for o in real:
        if _norm(o) == want:
            return o

    # 2. "Never" answers. "I have never worked at Robinhood" must still match
    #    when another company words it "I have never worked at Stripe". Only
    #    fires when exactly one option is a "never", so it can't pick between
    #    two of them.
    if "never" in want:
        nevers = [o for o in real if "never" in _norm(o)]
        if len(nevers) == 1:
            return nevers[0]

    # 3. "Prefer not to say" family -> the form's own decline option.
    if any(w in want for w in DECLINE_WORDS):
        for o in real:
            if any(w in _norm(o) for w in DECLINE_WORDS):
                return o
        return None

    # 4. Plain yes/no questions: match the leading token only.
    if want in ("yes", "no"):
        for o in real:
            if _norm(o).split()[:1] == [want]:
                return o
        return None

    # 5. Location-style values: "Dallas, TX" should find "Dallas, Texas,
    #    United States". Expand the state abbreviation and prefer a prefix hit
    #    so we take "Dallas, Texas" over "Lake Dallas, Texas".
    for variant in (want, _expand_state(want)):
        if not variant:
            continue
        pref = [o for o in real if _norm(o).startswith(variant)]
        if pref:
            return pref[0]

    # 6. Substring match — but only where the polarity agrees, so
    #    "I am not a protected veteran" can never select "I identify as a
    #    protected veteran". If more than one option survives, it's ambiguous.
    want_neg = _is_negative(desired)
    cands = [o for o in real
             if (want in _norm(o) or _norm(o) in want) and _is_negative(o) == want_neg]
    return cands[0] if len(cands) == 1 else None


# --------------------------------------------------------------------------- #
# Filling
# --------------------------------------------------------------------------- #
def _fill_text(page, idx: int, value) -> bool:
    try:
        page.fill(f"[data-jaa='{idx}']", str(value))
        return True
    except Exception:
        return False


def _select_option(page, idx: int, option_text: str) -> bool:
    try:
        page.select_option(f"[data-jaa='{idx}']", label=option_text)
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


# Attested / legal / EEO answers. These come from candidate_profile.json and
# are resolved by the deterministic rules ONLY - the LLM chooser is never
# consulted for them, no matter how confused the rules get. A blank here is a
# safe outcome; a model-invented legal declaration is not.
LLM_FORBIDDEN = {
    "work_authorization", "requires_sponsorship", "visa_status",
    "veteran_status", "military_status", "disability_status",
    "gender", "race_ethnicity", "lgbtq_status",
    "relationships_disclosure", "government_official",
}


from pathlib import Path

# Where the per-run fill report is written, so the terminal scrolling away
# doesn't lose it. Kept as a literal path to preserve this module's rule of
# importing nothing from the rest of the project.
_REPORT_PATH = Path(__file__).resolve().parent / "output" / "last_fill_report.txt"


# Fields where taking the first offered option is acceptable when nothing
# matches confidently. Deliberately excludes every attested question.
FIRST_OPTION_OK = {"preferred_office_location"}


def _fill_combobox(page, idx: int, desired, *, allow_first: bool = False,
                   chooser=None, label: str = "", llm_first: bool = False):
    """Drive a react-select combobox. Returns (filled?, who_decided)."""
    el, options = _combobox_options(page, idx)

    if not options:
        # Async autocomplete (the city field geocodes server-side): no options
        # until you type. Type, then RE-READ - clicking again would close the
        # menu we just opened.
        try:
            query = str(desired).split(",")[0].strip() or str(desired)
            el.type(query, delay=120)
            late = []
            for _ in range(4):
                page.wait_for_timeout(1000)
                late = _read_options(page, el)
                if late:
                    break
            if late:
                choice, how = _decide(late, desired, label, chooser, llm_first)
                if choice is None and allow_first:
                    choice, how = late[0], "first-option"
                if choice is not None:
                    page.click(f"[data-jaaopt='{late.index(choice)}']")
                    page.wait_for_timeout(400)
                    return True, how
            page.keyboard.press("Escape")
            return False, "—"
        except Exception:
            return False, "—"

    choice, how = _decide(options, desired, label, chooser, llm_first)
    if choice is None and allow_first:
        real = [o for o in options
                if o.strip() and not re.match(r"^(select|choose|--)", o.strip(), re.I)]
        choice, how = (real[0], "first-option") if real else (None, "—")
    if choice is None:
        try:
            page.keyboard.press("Escape")      # leave it untouched for the human
        except Exception:
            pass
        return False, "—"
    try:
        page.click(f"[data-jaaopt='{options.index(choice)}']")
        page.wait_for_timeout(350)
        return True, how
    except Exception:
        return False, "—"


def _upload_resume(page, resume_path) -> bool:
    try:
        for el in page.query_selector_all("input[type='file']"):
            el.set_input_files(resume_path)   # first file input = resume
            return True
    except Exception:
        pass
    return False


def _decide(options: list[str], desired, label: str, chooser, llm_first: bool):
    """Return (chosen_option_text | None, who_decided).

    With llm_first the model is asked before the rules, and is handed the
    profile value so it is matching a known fact to this form's wording rather
    than inventing one. The rules remain the fallback, and the rules alone can
    still decline - a blank beats a wrong answer on an application.
    """
    want = desired[0] if isinstance(desired, (list, tuple)) else desired

    if llm_first and chooser:
        try:
            idx = chooser(label, options, want)
            if idx is not None and 0 <= idx < len(options):
                return options[idx], "gpt"
        except Exception:
            pass

    option = _choose_option(options, desired)
    if option is not None:
        return option, "rule"

    if not llm_first and chooser:                # fallback mode
        try:
            idx = chooser(label, options, want)
            if idx is not None and 0 <= idx < len(options):
                return options[idx], "gpt"
        except Exception:
            pass
    return None, "—"


def _chooser_for(key: str, chooser):
    """The LLM chooser, unless this field is a legal attestation."""
    return None if (chooser is None or key in LLM_FORBIDDEN) else chooser


def fill_greenhouse_form(url: str, fields: dict, open_questions: list,
                         chooser=None, answerer=None, slow_mo: int = 0,
                         llm_first: bool = False) -> dict:
    """Open the form, fill what we can, upload the resume, and leave the browser
    OPEN for you to review + submit. Returns a report of filled/skipped fields.

    chooser:  callback (label, options) -> index|None, consulted ONLY when the
              deterministic rules find no confident match, and never for a
              field in LLM_FORBIDDEN.
    answerer: callback (label) -> text|None, for free-text questions the rules
              have no value for. Never called for a NEVER_ANSWER label.
    slow_mo:  milliseconds to pause between browser actions, so you can watch
              each field being filled instead of the form snapping to done.
    llm_first: ask the chooser for EVERY dropdown, using the deterministic
              rules only when it declines. The chooser is handed the profile
              value as ground truth, so the model is matching a known fact to
              this form's wording rather than deciding the fact.

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
                     url, browser, pw, llm_first)
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


def _fill(page, fields, open_questions, chooser, answerer, url, browser, pw,
          llm_first=False) -> dict:
    """The actual filling. Split out so fill_greenhouse_form can guarantee it
    always returns a live browser handle even when this raises."""
    filled: list[str] = []
    skipped: list[str] = []
    needs_confirm: list[str] = []
    seen: set[str] = set()

    # 1) Resume upload first — some forms re-render (and clear) after parsing it.
    if fields.get("resume_file"):
        (filled if _upload_resume(page, fields["resume_file"]) else skipped).append("resume_file")
        page.wait_for_timeout(2000)

    # 2) Every labelled control we can confidently map.
    for ctrl in _scan_controls(page):
        if ctrl["hidden"] or ctrl["type"] in ("file", "hidden", "submit", "button"):
            continue
        key = _match_key(ctrl["label"])
        if not key or key in seen:
            continue
        value = fields.get(key)
        if value in (None, ""):
            continue

        if ctrl["tag"] == "select":
            option, how = _decide(ctrl["options"], value, ctrl["label"],
                                  chooser, llm_first)
            ok = _select_option(page, ctrl["idx"], option) if option else False
            how = how if ok else "—"
        elif ctrl.get("combo"):
            ok, how = _fill_combobox(page, ctrl["idx"], value,
                                     allow_first=key in FIRST_OPTION_OK,
                                     chooser=chooser, label=ctrl["label"],
                                     llm_first=llm_first)
        elif ctrl["type"] in ("radio", "checkbox"):
            ok, how = False, "—"                # never auto-tick consent boxes
        else:
            # A list value is an ordered preference for dropdowns; a plain text
            # box just gets the first choice.
            ok = _fill_text(page, ctrl["idx"],
                            value[0] if isinstance(value, (list, tuple)) else value)
            how = "profile"

        shown = value[0] if isinstance(value, (list, tuple)) else value
        source = (how if ok else "—")
        print(f"    {'OK ' if ok else '   '} {key:<28}{str(shown)[:32]:<34}"
              f"{source if ok else 'left blank for you'}")
        (filled if ok else skipped).append(f"{key} ({ctrl['label'][:40]})")
        if ok:
            seen.add(key)

    # 3) Free-text questions. The pre-drafted "why this role" goes in first,
    #    then the answerer writes anything else that is still blank.
    answered: list[str] = []
    if open_questions:
        draft = open_questions[0].get("drafted_answer", "")
        placed = False
        for ta in page.query_selector_all("textarea"):
            try:
                if not (ta.input_value() or "").strip():
                    ta.fill(draft)
                    answered.append("why_this_role")
                    placed = True
                    break
            except Exception:
                continue
        if not placed:
            skipped.append("why_this_role")

    if answerer:
        for ctrl in _scan_controls(page):
            label = ctrl["label"]
            if (ctrl["hidden"] or ctrl.get("combo")
                    or ctrl["tag"] not in ("textarea", "input")
                    or ctrl["type"] in ("file", "hidden", "submit", "button",
                                        "checkbox", "radio", "tel", "email")
                    or not label or len(label) < 12
                    or IGNORE_LABEL.search(label) or NEVER_ANSWER.search(label)
                    or _match_key(label)):        # rules already own this field
                continue
            try:
                el = page.query_selector(f"[data-jaa='{ctrl['idx']}']")
                if el is None or (el.input_value() or "").strip():
                    continue                      # already filled - leave it
                text = answerer(label)
                if text:
                    el.fill(text)
                    answered.append(f"{label[:42]} (written)")
            except Exception:
                continue

    filled.extend(answered)

    # 3b) Everything on the page we did NOT touch, and why. Without this the
    #     unmatched controls are invisible - they never enter the fill loop at
    #     all, so a blank box on the form has no explanation in the terminal.
    untouched = []
    for ctrl in _scan_controls(page):
        label = (ctrl["label"] or "").strip()
        if (ctrl["hidden"] or ctrl["type"] in ("hidden", "submit", "button")
                or not label or IGNORE_LABEL.search(label)):
            continue
        try:
            el = page.query_selector(f"[data-jaa='{ctrl['idx']}']")
            if el is None:
                continue
            if ctrl["type"] in ("checkbox", "radio"):
                if el.is_checked():
                    continue
                why = "checkbox - never auto-ticked"
            else:
                cur = (el.input_value() or "").strip()
                if not cur and ctrl.get("combo"):
                    ctl = page.evaluate(
                        """(i)=>{const e=document.querySelector(`[data-jaa="${i}"]`);
                           const c=e&&e.closest('[class*=select__control]');
                           const v=c&&c.querySelector('[class*=single-value]');
                           return v?v.innerText.trim():''}""", ctrl["idx"])
                    cur = ctl or ""
                if cur:
                    continue
                key = _match_key(label)
                why = ("legal question - answered from your profile only"
                       if key in LLM_FORBIDDEN else
                       "no rule and no confident match" if key else
                       "not recognised - no rule for this question")
                if NEVER_ANSWER.search(label):
                    why = "compliance follow-up - deliberately left for you"
            untouched.append((label[:58], why))
        except Exception:
            continue

    if untouched:
        print(f"\n  LEFT EMPTY ({len(untouched)}) — fill these yourself before submitting:")
        for label, why in untouched:
            print(f"    ·  {label:<60} {why}")
    else:
        print("\n  LEFT EMPTY (0) — every visible field on the page has a value.")

    # 4) Report. Anything not confidently matched is YOUR job during review.
    print("\n[apply] Filled:")
    for f in filled:
        print(f"          + {f}")
    if not filled:
        print("          (nothing — the page may not be a hosted Greenhouse form)")
    if needs_confirm:
        print("[apply] Typed for you — pick the dropdown suggestion to lock it in:")
        for n in needs_confirm:
            print(f"          ~ {n}")
    print("[apply] Set these manually during review:")
    for s in skipped:
        print(f"          - {s}")
    if not skipped:
        print("          (none)")
    print("[apply] Confirm these values from your profile file:")
    for k in ["work_authorization", "requires_sponsorship", "visa_status", "willing_to_relocate",
              "salary_expectation", "earliest_start_date", "years_experience"]:
        print(f"          {k}: {fields.get(k)}")
    print("[apply] The submit button is NOT clicked. That part is yours.")

    # Terminal output scrolls away; keep a copy you can read at your own pace.
    try:
        report = ["FORM FILL REPORT", url, "",
                  f"FILLED ({len(filled)}):"] + [f"  + {x}" for x in filled]
        report += ["", f"LEFT EMPTY ({len(untouched)}):"]
        report += [f"  - {lab}\n      {why}" for lab, why in untouched]
        report += ["", "Values taken from your profile:"]
        report += [f"  {k}: {fields.get(k)}" for k in
                   ("work_authorization", "requires_sponsorship", "visa_status",
                    "willing_to_relocate", "salary_expectation",
                    "earliest_start_date", "years_experience")]
        path = _REPORT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(report), encoding="utf-8")
        print(f"[apply] Full report saved to {path}")
    except Exception:
        pass

    return {"filled": filled, "skipped": skipped, "needs_confirm": needs_confirm,
            "untouched": untouched, "browser": browser, "pw": pw}
