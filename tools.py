"""Host-side source tools for the deep research agents.

Network access and API credentials stay on the host; only returned research text
is passed into the sandbox-backed agents.
"""
import json
import os
import random
import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import quote

import httpx
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_ARXIV_MIN_INTERVAL = 3.0
_arxiv_lock = threading.Lock()
_last_arxiv_request = 0.0


class RetryableError(Exception):
    """A transient failure that should be retried; retry_after is seconds, if provided."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn with bounded exponential backoff, jitter, and Retry-After support."""
    attempts = int(attempts)
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            try:
                retry_after = float(exc.retry_after) if exc.retry_after is not None else None
            except (TypeError, ValueError):
                retry_after = None

            if retry_after is None:
                exponential = max(0.0, float(base)) * (2 ** attempt)
                delay = exponential + random.uniform(0.0, min(1.0, exponential * 0.25))
            else:
                delay = max(0.0, retry_after)
            time.sleep(min(max(0.0, float(cap)), delay))


def _retry_after_seconds(value):
    """Parse Retry-After as seconds or an HTTP date."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(value))
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _http_attempt(url, *, params=None, headers=None, timeout=30.0):
    """Perform one HTTP GET and translate only transient failures for with_retry."""
    try:
        response = httpx.get(url, params=params, headers=headers, timeout=timeout, follow_redirects=True)
    except httpx.TransportError as exc:
        raise RetryableError(f"network error: {exc}") from exc

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if response.status_code in RETRYABLE_STATUS_CODES:
            retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
            raise RetryableError(f"HTTP {response.status_code}", retry_after=retry_after) from exc
        raise
    return response


def _http_post_attempt(url, *, json_body, params=None, headers=None, timeout=30.0):
    """Perform one HTTP POST and translate only transient failures for with_retry."""
    try:
        response = httpx.post(url, params=params, headers=headers, json=json_body,
                              timeout=timeout, follow_redirects=True)
    except httpx.TransportError as exc:
        raise RetryableError(f"network error: {exc}") from exc

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if response.status_code in RETRYABLE_STATUS_CODES:
            retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
            raise RetryableError(f"HTTP {response.status_code}", retry_after=retry_after) from exc
        raise
    return response


def _get(url, *, params=None, headers=None, timeout=30.0, attempts=5, base=1.0, cap=30.0):
    return with_retry(lambda: _http_attempt(url, params=params, headers=headers, timeout=timeout),
                      attempts=attempts, base=base, cap=cap)


def _collapse(value, limit=None):
    text = " ".join(str(value or "").split())
    return text if limit is None else text[:limit]


def _bounded_int(value, default, low, high):
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError, OverflowError):
        return default


def _error(exc, secret=None):
    message = f"{type(exc).__name__}: {exc}"
    if secret:
        for secret_form in {secret, quote(secret, safe="")}:
            if secret_form:
                message = message.replace(secret_form, "[REDACTED]")
    return f"ERROR: {message}"


def _arxiv_get(params):
    """Serialize calls and all their retries so arXiv requests stay at least 3s apart."""
    def request_with_spacing():
        global _last_arxiv_request
        with _arxiv_lock:
            wait = _ARXIV_MIN_INTERVAL - (time.monotonic() - _last_arxiv_request)
            if _last_arxiv_request and wait > 0:
                time.sleep(wait)
            _last_arxiv_request = time.monotonic()
            return _http_attempt(ARXIV_URL, params=params, timeout=20.0)

    return with_retry(request_with_spacing, attempts=5, base=2.0, cap=20.0)


@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns JSON records with id, URL, date, title, and summary."""
    try:
        terms = re.findall(r"[\w-]+", str(query or ""), flags=re.UNICODE)
        if not terms:
            return "NO RESULTS"
        search_query = " AND ".join(f"all:{term}" for term in terms)
        response = _arxiv_get({
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": _bounded_int(max_results, 10, 1, 30),
            "start": 0,
        })

        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(response.text)
        records = []
        for entry in root.findall("atom:entry", namespace):
            def value(name):
                node = entry.find(f"atom:{name}", namespace)
                return node.text if node is not None else ""

            paper_url = value("id").strip()
            paper_id = paper_url.rsplit("/abs/", 1)[-1].split("?", 1)[0]
            paper_id = re.sub(r"v\d+$", "", paper_id)
            if not paper_id:
                continue
            records.append({
                "id": paper_id,
                "url": f"https://arxiv.org/abs/{paper_id}",
                "published": _collapse(value("published"))[:10],
                "title": _collapse(value("title")),
                "summary": _collapse(value("summary"), 600),
            })
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # tools must return errors to the agent instead of raising
        return _error(exc)


def _hf_items(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("papers", "results", "items"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload]
    return []


def _hf_record(item, *, prefer_ai_summary=False):
    if not isinstance(item, dict):
        return None
    paper = item.get("paper") if isinstance(item.get("paper"), dict) else item
    paper_id = paper.get("id") or item.get("id")
    if not paper_id:
        return None

    summary = ""
    if prefer_ai_summary:
        summary = item.get("ai_summary") or paper.get("ai_summary")
    summary = summary or item.get("summary") or paper.get("summary") or ""
    title = item.get("title") or paper.get("title") or ""
    published = (item.get("publishedAt") or paper.get("publishedAt") or
                 item.get("published") or paper.get("published") or "")
    github = paper.get("githubRepo") or item.get("githubRepo") or item.get("github") or ""
    stars = paper.get("githubStars") or item.get("githubStars") or item.get("stars") or 0
    upvotes = paper.get("upvotes") or item.get("upvotes") or 0
    try:
        upvotes = int(upvotes)
    except (TypeError, ValueError, OverflowError):
        upvotes = 0
    try:
        stars = int(stars)
    except (TypeError, ValueError, OverflowError):
        stars = 0

    paper_id = str(paper_id).strip()
    return {
        "id": paper_id,
        "url": f"https://huggingface.co/papers/{paper_id}",
        "published": _collapse(published)[:10],
        "title": _collapse(title),
        "summary": _collapse(summary, 600),
        "upvotes": upvotes,
        "github": _collapse(github),
        "stars": stars,
    }


@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """List trending Hugging Face Daily Papers, optionally for a date and filtered by keyword. Returns JSON paper records sorted by upvotes; use hf_search_papers for topic search."""
    try:
        params = {"limit": _bounded_int(limit, 30, 1, 100)}
        if str(date or "").strip():
            params["date"] = str(date).strip()
        response = _get(HF_DAILY_URL, params=params)
        records = [record for item in _hf_items(response.json())
                   if (record := _hf_record(item)) is not None]
        keyword = str(keyword or "").strip().casefold()
        if keyword:
            records = [record for record in records
                       if keyword in f"{record['title']} {record['summary']}".casefold()]
        records.sort(key=lambda record: record["upvotes"], reverse=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns JSON records with date, title, summary, upvotes, GitHub repository, and stars."""
    try:
        query = str(query or "").strip()
        if not query:
            return "NO RESULTS"
        response = _get(HF_SEARCH_URL, params={"q": query, "limit": _bounded_int(limit, 10, 1, 50)})
        records = [record for item in _hf_items(response.json())
                   if (record := _hf_record(item, prefer_ai_summary=True)) is not None]
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


def _exa_retry_after(meta):
    if not isinstance(meta, dict):
        return None
    for key, value in meta.items():
        if "retry" in str(key).lower() and "after" in str(key).lower():
            parsed = _retry_after_seconds(value)
            if parsed is not None:
                return parsed
        if isinstance(value, dict):
            parsed = _exa_retry_after(value)
            if parsed is not None:
                return parsed
    return None


def _meta_marks_rate_limited(meta):
    if isinstance(meta, dict):
        for key, value in meta.items():
            normalized = re.sub(r"[^a-z]", "", str(key).lower())
            if ("rate" in normalized and "limit" in normalized) or "throttl" in normalized:
                if (value is True or value == 429 or
                        ("limited" in normalized and bool(value)) or
                        (isinstance(value, str) and any(marker in value.lower() for marker in (
                            "true", "yes", "limited", "exceed", "throttl", "429",
                        )))):
                    return True
            if isinstance(value, (dict, list)) and _meta_marks_rate_limited(value):
                return True
    elif isinstance(meta, list):
        return any(_meta_marks_rate_limited(item) for item in meta)
    return False


def _decode_exa_response(response):
    text = response.text.strip()
    messages = []
    for line in text.splitlines():
        if line.startswith("data:"):
            data = line[5:].strip()
            if data and data != "[DONE]":
                try:
                    messages.append(json.loads(data))
                except json.JSONDecodeError:
                    continue
    if not messages:
        try:
            messages = [json.loads(text)]
        except json.JSONDecodeError as exc:
            raise RuntimeError("Exa returned an unreadable JSON-RPC response") from exc

    payload = next((item for item in reversed(messages)
                    if isinstance(item, dict) and ("result" in item or "error" in item)), messages[-1])
    if not isinstance(payload, dict):
        raise RuntimeError("Exa returned an unexpected JSON-RPC response")
    if payload.get("error"):
        raise RuntimeError(f"Exa JSON-RPC error: {payload['error']}")
    result = payload.get("result", payload)
    if not isinstance(result, dict):
        raise RuntimeError("Exa returned an unexpected result")
    output_text = _exa_text(result)
    meta = result.get("_meta") or payload.get("_meta")
    if _meta_marks_rate_limited(meta) or _text_marks_rate_limited(output_text):
        raise RetryableError("Exa rate limit reported by MCP", retry_after=_exa_retry_after(meta))
    if result.get("isError"):
        raise RuntimeError("Exa tool returned an error: " + output_text)
    return output_text


def _exa_text(result):
    content = result.get("content", []) if isinstance(result, dict) else []
    pieces = []
    for block in content if isinstance(content, list) else []:
        if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
            pieces.append(str(block["text"]))
    if pieces:
        return "\n".join(pieces).strip()
    structured = result.get("structuredContent") if isinstance(result, dict) else None
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False)
    return ""


def _text_marks_rate_limited(text):
    lowered = str(text or "").lower()
    return any(marker in lowered for marker in (
        "rate limit exceeded", "rate limited", "rate-limited", "too many requests",
        "quota exceeded", "temporarily throttled", "try again later",
    ))


def _exa_call(tool_name, arguments):
    api_key = (os.getenv("EXA_API_KEY") or "").strip()
    params = {"exaApiKey": api_key} if api_key else None
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }

    def attempt():
        response = _http_post_attempt(EXA_URL, json_body=payload, params=params, headers=headers, timeout=30.0)
        return _decode_exa_response(response)

    try:
        return with_retry(attempt, attempts=5, base=2.0, cap=20.0), api_key
    except Exception as exc:
        raise RuntimeError(_error(exc, secret=api_key).removeprefix("ERROR: ")) from exc


@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web through Exa for pages matching a query and objective. Returns text for top results with URLs."""
    api_key = (os.getenv("EXA_API_KEY") or "").strip()
    try:
        query = str(query or "").strip()
        if not query:
            return "NO RESULTS"
        objective = str(objective or "").strip() or f"Find reliable, relevant sources about: {query}"
        result, api_key = _exa_call("web_search_exa", {
            "query": query,
            "objective": objective,
            "numResults": _bounded_int(num_results, 5, 1, 10),
        })
        result = _collapse(result)
        if api_key:
            for secret_form in {api_key, quote(api_key, safe="")}:
                if secret_form:
                    result = result.replace(secret_form, "[REDACTED]")
        return result if result else "NO RESULTS"
    except Exception as exc:
        return _error(exc, secret=api_key)


@tool
def web_fetch(url: str) -> str:
    """Fetch one web page through Exa and return its text or markdown, truncated to about 12,000 characters."""
    api_key = (os.getenv("EXA_API_KEY") or "").strip()
    try:
        url = str(url or "").strip()
        if not url:
            return "NO RESULTS"
        result, api_key = _exa_call("web_fetch_exa", {"urls": [url]})
        result = result.strip()
        if not result:
            return "NO RESULTS"
        result = result[:12000]
        if api_key:
            for secret_form in {api_key, quote(api_key, safe="")}:
                if secret_form:
                    result = result.replace(secret_form, "[REDACTED]")
        return result
    except Exception as exc:
        return _error(exc, secret=api_key)


# ---- Source registry: exactly these tools are available to researchers. ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        print(f"== {name}\n{fn.invoke(args)[:400]}\n")
