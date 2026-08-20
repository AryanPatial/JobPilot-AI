"""Central configuration: paths, env loading, and the LLM factory.

All model access goes through get_llm() so you can swap providers in one place.
Default is Google Gemini (free tier). Nothing here reads your API key directly –
it comes from the environment (.env), which is gitignored.
"""
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ---- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESUMES_DIR = DATA_DIR / "resumes"
TEMPLATES_DIR = ROOT / "templates"
OUTPUT_DIR = ROOT / "output"
OUTPUT_RESUMES_DIR = OUTPUT_DIR / "resumes"
TRACKER_PATH = OUTPUT_DIR / "applications.xlsx"
CHECKPOINT_DB = OUTPUT_DIR / "checkpoints.sqlite"

OUTPUT_RESUMES_DIR.mkdir(parents=True, exist_ok=True)

# ---- Model -----------------------------------------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def get_llm(temperature: float = 0.3, structured_schema=None):
    """Return a LangChain chat model. Pass a Pydantic class to force structured output.

    Swapping providers later (Groq, OpenAI) means changing only this function.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=temperature)
    if structured_schema is not None:
        return llm.with_structured_output(structured_schema)
    return llm
