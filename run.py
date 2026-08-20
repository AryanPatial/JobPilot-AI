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

# ======================= EDIT THESE ======================================== #
# Company Greenhouse boards to search (add/remove freely).
#
# IMPORTANT: only companies that use a HOSTED Greenhouse form can be auto-filled.
# Verified 2026-08-19:
#   robinhood, gitlab  -> job-boards.greenhouse.io  (auto-fill works)
#   databricks         -> redirects to databricks.com (own careers site, NO form)
#   airtable, discord  -> no US data/AI roles open right now
# If a board's absolute_url isn't on greenhouse.io, the browser still opens it
# but the fields won't match. Keep hosted-form companies first.
COMPANY_BOARDS = ["robinhood", "gitlab"]

# Titles you care about (matched loosely against job titles).
TARGET_TITLES = [
    "Data Analyst", "Data Scientist", "Data Engineer", "Analytics Engineer",
    "Business Analyst", "Business Intelligence", "AI Engineer", "ML Engineer",
    "Machine Learning", "Generative AI", "LLM",
]

AUTO_FILL_BROWSER = True   # set False to only prepare (no browser), like before

# A fresh thread id per run. Reusing one id makes LangGraph resume the PREVIOUS
# run's checkpoint instead of starting a new search.
THREAD = {"configurable": {"thread_id": f"run-{datetime.now():%Y%m%d-%H%M%S}"}}
# =========================================================================== #


def main():
    app = agent.build_graph()

    print("\n=== SEARCHING + PREPARING (will pause before you submit) ===\n")
    state = app.invoke({"board_tokens": COMPANY_BOARDS, "titles": TARGET_TITLES}, THREAD)

    if state.get("error") == "no_jobs":
        print("No US jobs matched. Try different company boards or titles.")
        return

    job = state["current_job"]
    print("\n=== JOB SELECTED ===")
    print(f"  {job['title']} @ {job['company']} — {job['location']}")
    print(f"  {job['url']}")
    print(f"  Resume: {state['resume_pdf_path']}")
    print("\n  Match scores:")
    for s in state["resume_scores"]:
        print(f"    {s['resume_id']:<15} {s['score']}/100")

    # ---- Auto-fill the real form (browser stays open for you) ----
    handle = None
    if AUTO_FILL_BROWSER:
        print("\n=== OPENING + FILLING THE APPLICATION FORM ===")
        handle = apply.fill_greenhouse_form(job["url"], state["application_fields"], state["open_questions"])
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
