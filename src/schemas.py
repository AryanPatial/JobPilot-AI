"""Pydantic schemas that force reliable, parseable LLM output.

Using with_structured_output(Schema) means the model must return exactly these
fields every time — no "eight" vs 8 surprises.
"""
from typing import List
from pydantic import BaseModel, Field


class ResumeMatch(BaseModel):
    """How well one resume fits a given job description."""
    score: int = Field(description="Fit score 0-100 for this resume against the JD", ge=0, le=100)
    reasoning: str = Field(description="One or two sentences on why this score")
    matched_keywords: List[str] = Field(default_factory=list, description="Important JD skills this resume already covers")
    missing_keywords: List[str] = Field(default_factory=list, description="Important JD skills this resume does NOT show")


class TailoredBullet(BaseModel):
    original: str = Field(description="The original bullet text, unchanged")
    revised: str = Field(description="The reworded bullet. Same facts, no invented experience. May re-emphasize to match the JD.")


class TailoringPlan(BaseModel):
    """A conservative plan to align a resume to a JD WITHOUT inventing anything."""
    emphasize_skills: List[str] = Field(default_factory=list, description="Existing skills to surface more prominently for this JD")
    revised_bullets: List[TailoredBullet] = Field(default_factory=list, description="Light rewrites of a FEW existing bullets. Never add new experience.")
    summary_of_changes: str = Field(description="Plain-English note of what was changed and why")


class DraftedAnswer(BaseModel):
    question: str
    answer: str = Field(description="A concise answer grounded ONLY in the candidate profile. If unknown, say it needs the candidate's input.")
    confidence: str = Field(description="one of: high, medium, low")
