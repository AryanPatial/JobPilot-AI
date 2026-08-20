"""The shared state that flows through the graph.

Every node receives this whole dict, reads what it needs, and returns a partial
update. Keys are written by exactly one node each in V1, so no reducers are
needed yet. (When you parallelize resume matching in V2, that's where an
Annotated[..., operator.add] reducer comes in — see README.)
"""
from typing import TypedDict, Optional, Any


class JobApplicationState(TypedDict, total=False):
    # ---- inputs (set at invoke time) ----
    search_criteria: dict            # {"titles": [...], "location": "...", "board_token": "..."}

    # ---- job discovery ----
    jobs: list                       # candidate jobs pulled from the source
    current_job: Optional[dict]      # the one job we chose to process
    job_description: Optional[str]   # plain-text JD for current_job

    # ---- resume matching ----
    resume_scores: list              # [{resume_id, score, reasoning, matched, missing}, ...]
    selected_resume_id: Optional[str]

    # ---- tailoring + rendering ----
    tailored_resume: Optional[dict]  # resume content after light, honest edits
    resume_pdf_path: Optional[str]   # path to the generated PDF

    # ---- application prep ----
    application_fields: Optional[dict]   # mapped standard fields (name, email, ...)
    open_questions: Optional[list]       # [{question, drafted_answer}, ...]

    # ---- bookkeeping ----
    status: str                      # human-readable pipeline status
    error: Optional[str]
