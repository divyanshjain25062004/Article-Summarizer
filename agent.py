# agent.py
import time
import concurrent.futures
from pathlib import Path
import json

from tools import multi_source_search, rank_candidates, fetch_page, extract_main, is_probably_article
from prompts import SUMM_SYSTEM, make_user_prompt
from llm_client import summarize_structured
# thread pool to enforce timeouts on blocking ops (httpx/LLM)
_EXEC = concurrent.futures.ThreadPoolExecutor(max_workers=6)

def run_with_timeout(fn, timeout, *args, **kwargs):
    fut = _EXEC.submit(fn, *args, **kwargs)
    try:
        return fut.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        raise TimeoutError(f"{fn.__name__} timed out after {timeout}s")


# def agent_run(query: str, top_k: int = 3, domain_only: str | None = None) -> dict:
#     trace = {"query": query, "steps": []}

#     # 1) Search
#     try:
#         results = web_search(query, max_results=30, domain_only=domain_only)
#         trace["steps"].append({"name": "web_search", "found": len(results)})
#     except Exception as e:
#         results = []
#         trace["steps"].append({"name": "web_search", "found": 0, "error": f"{type(e).__name__}: {e}"})

#     if not results:
#         Path("logs").mkdir(exist_ok=True)
#         Path("logs/last_trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
#         return {"query": query, "articles": [], "trace_path": "logs/last_trace.json"}

#     # 2) Rank, then keep a few spares
#     try:
#         ranked = rank_candidates(query, results, top_n=10)
#     except Exception:
#         ranked = results[:10]
#     trace["steps"].append({"name": "rank_candidates", "selected": [
#         {"title": r.get("title"), "url": r.get("url"), "score": r.get("score", None)} for r in ranked[:top_k]
#     ]})
def agent_run(query: str, top_k: int = 3, domain_only: str | None = None, sources: list[str] | None = None) -> dict:
    trace = {"query": query, "steps": []}

    # ---- 0) Time budget for whole request ----
    BUDGET_S = 32            # total wall-clock budget
    FETCH_TIMEOUT_S = 10     # per-page fetch
    SUMM_TIMEOUT_S = 18      # per-summary
    start = time.time()
    deadline = start + BUDGET_S

    def time_left():
        return max(0.0, deadline - time.time())

    # ---- 1) Multi-source search ----
    try:
        results = multi_source_search(query, max_results_per_source=30, domain_only=domain_only, sources=sources)
        trace["steps"].append({"name": "multi_source_search", "found": len(results), "sources": sources or ["web","news","reddit","twitter"]})
    except Exception as e:
        results = []
        trace["steps"].append({"name": "multi_source_search", "found": 0, "error": f"{type(e).__name__}: {e}"})

    if not results:
        Path("logs").mkdir(exist_ok=True)
        Path("logs/last_trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
        return {"query": query, "articles": [], "trace_path": "logs/last_trace.json"}

    # ---- 2) Rank and keep spares ----
    ranked = rank_candidates(query, results, top_n=12)
    trace["steps"].append({"name": "rank_candidates", "selected": [
        {"title": r.get("title"), "url": r.get("url"), "source": r.get("source"), "score": r.get("score")} for r in ranked[:top_k]
    ]})

    # ---- 3) Summarize with strict per-step timeouts ----
    summaries: list[dict] = []

    def fallback_summary(title: str, snippet: str, url: str) -> dict:
        snippet_text = (snippet or "").strip()
        if snippet_text:
            body = f"{snippet_text}\n\nVisit the source link for the full context."
        else:
            body = (
                "We couldn’t reliably fetch the full article right now. Open the source "
                "to read the complete details."
            )
        return {"title": title or "Untitled", "url": url, "summary": body}

    def build_summary_input(article_text: str, snippet: str) -> tuple[str, bool]:
        primary = (article_text or "").strip()
        snippet_text = (snippet or "").strip()

        if primary and len(primary) >= 320:
            return primary, True

        pieces: list[str] = []
        if primary:
            pieces.append(primary)
        if snippet_text and snippet_text not in primary:
            pieces.append(snippet_text)

        combined = "\n\n".join(pieces).strip()
        return combined, len(combined) >= 220


    for cand in ranked:
        if len(summaries) >= top_k:
            break
        if time.time() > deadline:
            break  # out of time; return what we have

        url = cand["url"]
        title_guess = (cand.get("title") or "Untitled").strip()
        snippet = (cand.get("snippet") or "").strip()
        if not is_probably_article(url, title_guess):
            continue

        fstep = {"name": "fetch_extract", "url": url}
        try:
            # if almost out of time, fallback directly
            if time_left() < 3.0:
                summaries.append(fallback_summary(title_guess, snippet, url))
                continue

            # fetch with timeout
            html = run_with_timeout(fetch_page, min(FETCH_TIMEOUT_S, time_left()), url)
            title, text = extract_main(html, base_url=url)
            combined_text, has_enough = build_summary_input(text, snippet)
            fstep["ok"] = True; fstep["chars"] = len(combined_text)
            trace["steps"].append(fstep)

            # too short even after enrichment? fallback
            if not has_enough or not combined_text:
                summaries.append(fallback_summary(title or title_guess, snippet, url))
                continue

            # summarize with timeout
            user_prompt = make_user_prompt(query, title or title_guess, url, mode="long")
            try:
                summ = run_with_timeout(
                    summarize_structured,
                    min(SUMM_TIMEOUT_S, time_left()),
                    SUMM_SYSTEM,
                    user_prompt,
                    combined_text,
                )
            except Exception as e:
                # timed out or failed → fallback summary
                trace["steps"].append({"name": "summarize", "url": url, "ok": False, "error": str(e)})
                summaries.append(fallback_summary(title or title_guess, snippet, url))
                continue

            trace["steps"].append({"name": "summarize", "url": url, "ok": True})
            summaries.append({"title": title or title_guess, "url": url, "summary": summ})

        except Exception as e:
            fstep["ok"] = False; fstep["error"] = str(e); fstep["chars"] = 0
            trace["steps"].append(fstep)
            summaries.append(fallback_summary(title_guess, snippet, url))

    # If still short (budget ran out early), pad with quick fallbacks
    for cand in ranked:
        if len(summaries) >= top_k: break
        summaries.append(fallback_summary(cand.get("title","Untitled"), cand.get("snippet",""), cand["url"]))

    articles = summaries[:top_k]
    Path("logs").mkdir(exist_ok=True)
    Path("logs/last_trace.json").write_text(json.dumps(trace, indent=2), encoding="utf-8")
    return {"query": query, "articles": articles, "trace_path": "logs/last_trace.json"}
