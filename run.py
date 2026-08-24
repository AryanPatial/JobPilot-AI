"""Entry point. Run with: python run.py"""
import json
import sys
from datetime import datetime

import agent
import apply
import llm_tasks

# ======================= EDIT THESE ======================================== #
# Company Greenhouse boards to search
COMPANY_BOARDS = ["reddit", "anthropic", "scaleai", "chime", "gusto",
                  "figma", "twilio", "affirm"]

# Used when you press Enter at the prompt instead of typing something.
DEFAULT_INTENT = "gen AI and machine learning engineer roles"

AUTO_FILL_BROWSER = True   # False = prepare only, no browser
SLOW_MO_MS = 300           # pause between browser actions so you can watch it
# =========================================================================== #


def ask_intent() -> str:
    """What kind of role are you after?"""
    print("=" * 66)
    print("  WHAT ARE YOU LOOKING FOR?  (US roles only)")
    print("=" * 66)
    print(f"  e.g. 'gen AI roles' · 'data analyst jobs' · 'ML engineer, new grad'")
    print(f"  Press Enter for: {DEFAULT_INTENT!r}\n")
    typed = input("  > ").strip()
    intent = typed or DEFAULT_INTENT
    print(f"\n  Searching US jobs for: {intent!r}")
    print(f"  Boards: {', '.join(COMPANY_BOARDS)}\n")
    return intent


def confirm_submitted() -> bool:
    """The human checkpoint."""
    if not sys.stdin.isatty():
        print("\n  stdin is not a terminal, so there is nobody here to approve"
              "\n  this. Run it directly in a terminal to use the checkpoint.")
        return False
    while True:
        try:
            reply = input(">>> Did you submit it? [y = log it / n = discard]: ").strip().lower()
        except EOFError:
            print("\n  No input available - treating as NOT submitted.")
            return False
        if reply in ("y", "yes"):
            return True
        if reply in ("n", "no"):
            return False
        print("    please type y or n")


def main():
    intent = ask_intent()
    app = agent.build_graph()

    # A fresh thread id per run
    thread = {"configurable": {"thread_id": f"run-{datetime.now():%Y%m%d-%H%M%S}"}}

    print("=== SEARCHING + PREPARING (will pause before you submit) ===\n")
    state = app.invoke({"board_tokens": COMPANY_BOARDS, "intent": intent}, thread)

    if state.get("error") in ("no_jobs", "no_suitable_jobs"):
        print(f"\nStopped: {state.get('status')}")
        for sk in state.get("skipped_jobs", []):
            print(f"   skipped {sk['title'][:46]} - {sk['score']}/100")
        print("Try a different search, more boards, or add projects to the pool.")
        return

    job = state["current_job"]
    coverage = state.get("coverage") or {}

    print("\n=== JOB SELECTED ===")
    print(f"  {job['title']} @ {job['company']} - {job['location']}")
    print(f"  {job['url']}")
    print(f"  Resume: {state['resume_pdf_path']}")
    print("\n  Resume variant scores:")
    for sc in state["resume_scores"]:
        print(f"    {sc['resume_id']:<16} {sc['score']}/100")
    print(f"  Projects chosen: "
          f"{', '.join(p['name'][:32] for p in state['selected_projects'])}")
    print(f"  Best JD coverage: {coverage.get('score')}/100 "
          f"(best of {state.get('attempts')} project combination(s))")
    for sk in state.get("skipped_jobs", []):
        print(f"  (skipped {sk['title'][:42]} - {sk['score']}/100)")

    # ---- Auto-fill the real form (browser stays open for you) ----
    handle = None
    if AUTO_FILL_BROWSER:
        print("\n=== OPENING + FILLING THE APPLICATION FORM ===")
        summary = llm_tasks.profile_summary()
        context = f"{job['title']} at {job['company']}. {state['job_description'][:400]}"

        def form_answerer(questions):
            """Playwright hands over every question it found, with the real
            options; one call answers them all."""
            print(f"\n  {len(questions)} question(s) found - asking GPT in one call\n")
            return llm_tasks.answer_form(questions, summary, context)

        before = llm_tasks.CALLS
        try:
            handle = apply.fill_greenhouse_form(
                job["url"], state["application_fields"], state["open_questions"],
                slow_mo=SLOW_MO_MS, form_answerer=form_answerer)
        except Exception as exc:
            # Even if the browser layer fails outright we must still reach the human checkpoint below
            import traceback
            traceback.print_exc()
            print(f"\n  Auto-fill failed: {type(exc).__name__}. "
                  f"Fill the form by hand if a window is open.")
        print(f"\n  Form filling used {llm_tasks.CALLS - before} LLM call(s).")
    else:
        print("\n(Auto-fill off. Prepared fields:)")
        print(json.dumps(state["application_fields"], indent=2))

    # ---- Human checkpoint ----
    print("\n" + "=" * 66)
    print("REVIEW THE BROWSER, FIX ANY SKIPPED FIELDS, THEN CLICK SUBMIT YOURSELF.")
    print("=" * 66)
    submitted = confirm_submitted()
    if not submitted:
        print("\nNot logged. The application was NOT recorded as submitted.")
        return

    if handle and handle.get("browser"):
        try:
            handle["browser"].close()
            handle["pw"].stop()
        except Exception:
            pass

    # ---- Resume the graph to record the application ----
    # You confirmed you submitted it, so it goes in the sheet as Applied.
    app.update_state(thread, {"submitted_status": "Applied"})
    final = app.invoke(None, thread)
    print("\n=== DONE ===")
    print("STATUS: logged to your tracker. (You are the one who clicked Submit.)")
    print(final.get("status"))


if __name__ == "__main__":
    main()
