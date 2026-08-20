"""Wire the nodes into a LangGraph StateGraph.

Flow (linear V1, one job):
    START
      -> search_jobs
      -> match_resumes
      -> tailor_resume
      -> render_resume
      -> prepare_application
      -> [PAUSE for human review]      <- interrupt_before
      -> human_review
      -> track_application
      -> END

The interrupt is real human-in-the-loop: the graph stops before human_review,
you inspect the prepared package + PDF, submit the application yourself, then
resume the graph to record it. Checkpointing (SQLite) means a crash mid-run
resumes from the last completed node instead of starting over.
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from .state import JobApplicationState
from . import nodes
from . import config


def build_graph():
    g = StateGraph(JobApplicationState)

    g.add_node("search_jobs", nodes.search_jobs)
    g.add_node("match_resumes", nodes.match_resumes)
    g.add_node("tailor_resume", nodes.tailor_resume)
    g.add_node("render_resume", nodes.render_resume)
    g.add_node("prepare_application", nodes.prepare_application)
    g.add_node("human_review", nodes.human_review)
    g.add_node("track_application", nodes.track_application)

    g.add_edge(START, "search_jobs")
    g.add_edge("search_jobs", "match_resumes")
    g.add_edge("match_resumes", "tailor_resume")
    g.add_edge("tailor_resume", "render_resume")
    g.add_edge("render_resume", "prepare_application")
    g.add_edge("prepare_application", "human_review")
    g.add_edge("human_review", "track_application")
    g.add_edge("track_application", END)

    checkpointer = SqliteSaver.from_conn_string(str(config.CHECKPOINT_DB))
    # Pause before the human_review node so you can inspect + apply manually.
    return g.compile(checkpointer=checkpointer, interrupt_before=["human_review"])
