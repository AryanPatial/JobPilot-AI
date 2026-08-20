"""Job source: Greenhouse public boards API.

Greenhouse exposes a company's open roles as JSON with no auth:
    https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs?content=true

board_token is the company's slug on their Greenhouse-hosted careers page
(e.g. many companies use their name). This is a legitimate, structured source —
no scraping, no ToS gray area.

The JobSource base class lets you add LeverSource / AshbySource / AdzunaSource
later without changing any graph code.
"""
from __future__ import annotations
import html
import re
import requests


def _strip_html(raw: str) -> str:
    """Greenhouse returns JD content as escaped HTML. Turn it into plain text."""
    if not raw:
        return ""
    text = html.unescape(raw)
    text = re.sub(r"<[^>]+>", " ", text)          # drop tags
    text = re.sub(r"\s+", " ", text).strip()       # collapse whitespace
    return text


class JobSource:
    """Interface every source implements. The graph only knows about this."""
    def search(self, criteria: dict) -> list[dict]:
        raise NotImplementedError


class GreenhouseSource(JobSource):
    BASE = "https://boards-api.greenhouse.io/v1/boards/{token}/jobs"

    def search(self, criteria: dict) -> list[dict]:
        token = criteria.get("board_token")
        if not token:
            raise ValueError("search_criteria needs a 'board_token' (the company's Greenhouse slug)")

        url = self.BASE.format(token=token)
        resp = requests.get(url, params={"content": "true"}, timeout=30)
        resp.raise_for_status()
        raw_jobs = resp.json().get("jobs", [])

        titles = [t.lower() for t in criteria.get("titles", [])]
        location_filter = (criteria.get("location") or "").lower()

        results = []
        for j in raw_jobs:
            title = j.get("title", "")
            loc = (j.get("location") or {}).get("name", "")

            if titles and not any(t in title.lower() for t in titles):
                continue
            if location_filter and location_filter not in loc.lower():
                continue

            results.append({
                "job_id": str(j.get("id")),
                "title": title,
                "company": token,
                "location": loc,
                "url": j.get("absolute_url"),
                "description": _strip_html(j.get("content", "")),
            })
        return results


# Registry so run.py / config can pick a source by name.
SOURCES = {
    "greenhouse": GreenhouseSource,
}


def get_source(name: str) -> JobSource:
    if name not in SOURCES:
        raise ValueError(f"Unknown source '{name}'. Available: {list(SOURCES)}")
    return SOURCES[name]()
