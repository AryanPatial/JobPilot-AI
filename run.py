"""
run.py  —  Start here.  Just run:  python run.py

What happens:
  1. Searches your target US jobs across the company boards below
  2. Picks the best-matching resume, tailors it, makes the PDF
  3. Opens the application form in a browser and fills it in
  4. PAUSES — you review the browser and click Submit yourself
  5. Logs the application to output/applications.xlsx
"""
import json
from datetime import datetime

import agent
import apply
import llm_tasks

# ======================= EDIT THESE ======================================== #
# Say what you want in plain English. An LLM expands this into the title
# variants real postings actually use — no hardcoded synonym list.
INTENT = "gen AI and machine learning engineer roles"

# Company Greenhouse boards to search.
# Only companies on a HOSTED Greenhouse form can be auto-filled — if a board's
# absolute_url redirects to the company's own careers site (Databricks, Stripe),
# the browser opens but the fields won't match.
COMPANY_BOARDS = ["reddit", "anthropic", "scaleai", "chime", "gusto",
                  "figma", "twilio", "affirm"]

AUTO_FILL_BROWSER = True   # False = prepare only, no browser
SLOW_MO_MS = 400           # pause between browser actions so you can watch it
                           # fill. 0 = instant.
USE_LLM_CHOOSER = True     # LLM picks dropdowns the rules can't resolve AND
                           # writes free-text answers. Never consulted for
                           # legal/EEO fields — see apply.LLM_FORBIDDEN.

# A fresh thread id per run. Reusing one makes LangGraph resume the PREVIOUS
# run's checkpoint instead of starting a new search.
THREAD = {"configurable": {"thread_id": f"run-{datetime.now():%Y%m%d-%H%M%S}"}}
# =========================================================================== #


def main():
    app = agent.build_graph()

    print("\n=== SEARCHING + PREPARING (will pause before you submit) ===\n")
    state = app.invoke({"board_tokens": COMPANY_BOARDS, "intent": INTENT}, THREAD)

    if state.get("error") in ("no_jobs", "no_suitable_jobs"):
        print(f"\nStopped: {state.get('status')}")
        for sk in state.get("skipped_jobs", []):
            print(f"   skipped {sk['title'][:46]} — {sk['score']}/100")
        print("Try a different INTENT, more boards, or add projects to the pool.")
        return

    job = state["current_job"]
    print("\n=== JOB SELECTED ===")
    print(f"  {job['title']} @ {job['company']} — {job['location']}")
    print(f"  {job['url']}")
    print(f"  Resume: {state['resume_pdf_path']}")
    print("\n  Match scores:")
    for sc in state["resume_scores"]:
        print(f"    {sc['resume_id']:<15} {sc['score']}/100")
    cov = state.get("coverage") or {}
    print(f"  Projects: {', '.join(p['name'][:34] for p in state['selected_projects'])}")
    print(f"  JD coverage: {cov.get('score')}/100 after {state.get('attempts')} "
          f"tailoring pass(es)")
    for sk in state.get("skipped_jobs", []):
        print(f"  (skipped {sk['title'][:42]} — {sk['score']}/100)")

    # ---- Auto-fill the real form (browser stays open for you) ----
    handle = None
    if AUTO_FILL_BROWSER:
        print("\n=== OPENING + FILLING THE APPLICATION FORM ===")
        chooser = answerer = None
        if USE_LLM_CHOOSER:
            summary = llm_tasks.profile_summary()
            context = f"{job['title']} at {job['company']}. {state['job_description'][:400]}"
            chooser = lambda label, options: llm_tasks.llm_choose(label, options, summary)
            answerer = lambda label: llm_tasks.llm_answer(label, summary, context)
        before = llm_tasks.CALLS
        handle = apply.fill_greenhouse_form(job["url"], state["application_fields"],
                                            state["open_questions"],
                                            chooser=chooser, answerer=answerer,
                                            slow_mo=SLOW_MO_MS)
        used = llm_tasks.CALLS - before
        print(f"\n  Form filling used {used} LLM call(s) — everything else came "
              f"from rules + your profile file.")
        if handle.get("error"):
            print("  ", handle["error"])
    else:
        print("\n(Auto-fill off. Prepared fields:)")
        print(json.dumps(state["application_fields"], indent=2))

    # ---- Human checkpoint ----
    print("\n" + "=" * 60)
    print("REVIEW THE BROWSER, FIX ANY SKIPPED FIELDS, THEN CLICK SUBMIT YOURSELF.")
    print("=" * 60)
    input(">>> Press Enter AFTER you've submitted, to log it... ")

    if handle and handle.get("browser"):
        try:
            handle["browser"].close(); handle["pw"].stop()
        except Exception:
            pass

    # ---- Resume graph to record the application ----
    final = app.invoke(None, THREAD)
    print("\n=== DONE ===")
    print("STATUS: logged to your tracker. (You are the one who clicked Submit.)")
    print(final.get("status"))


if __name__ == "__main__":
    main()
