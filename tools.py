from __future__ import annotations
import math, time, re
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse
import httpx
from bs4 import BeautifulSoup
import trafilatura
from ddgs import DDGS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

JUNK_PATTERNS = [
    r"/sitemap", r"/login", r"/signin", r"/subscribe", r"/privacy",
    r"/terms", r"/category/", r"/tag/", r"/topics/", r"/about", r"#",
]
WIKI_DOMAINS = {"wikipedia.org", "m.wikipedia.org", "simple.wikipedia.org"}

def hostname(url: str) -> str:
    try:
        h = urlparse(url).hostname or ""
        return h.replace("www.", "")
    except Exception:
        return ""

def is_probably_article(url: str, title: str = "") -> bool:
    u = (url or "").lower()
    t = (title or "").lower()
    if not u.startswith("http"):
        return False
    if any(p in u for p in JUNK_PATTERNS):
        return False
    if "sitemap" in t or "login" in t or "privacy" in t:
        return False
    return True
def is_social(url: str) -> bool:
    u = (url or "").lower()
    return "twitter.com" in u or "reddit.com" in u

def dedupe_by_url(items: List[Dict], key: str = "url") -> List[Dict]:
    seen, out = set(), []
    for it in items:
        u = it.get(key)
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(it)
    return out

_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
]

def fetch_page(url: str, timeout: int = 12) -> str:
    base_headers = {
        "User-Agent": _UAS[0],
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Referer": "https://www.google.com/",
    }
    tmo = httpx.Timeout(connect=6.0, read=15.0, write=6.0, pool=6.0)
    last_err = None
    for attempt in range(3):
        headers = dict(base_headers)
        headers["User-Agent"] = _UAS[attempt % len(_UAS)]
        try:
            with httpx.Client(follow_redirects=True, headers=headers, timeout=tmo, http2=True) as client:
                resp = client.get(url)
                if resp.status_code in (403, 429, 503):
                    last_err = RuntimeError(f"HTTP {resp.status_code}")
                    time.sleep(0.6 * (2 ** attempt))
                    continue
                resp.raise_for_status()
                text = resp.text or ""
                if len(text) < 400:
                    last_err = RuntimeError("tiny response")
                    time.sleep(0.4 * (2 ** attempt))
                    continue
                return text
        except Exception as e:
            last_err = e
            time.sleep(0.4 * (2 ** attempt))
            continue
    try:
        alt = trafilatura.fetch_url(url)
        if alt:
            return alt
    except Exception:
        pass
    raise last_err or RuntimeError("failed to fetch after retries")

def _og_fallback_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = ""
    for sel in [
        "meta[property='og:title']",
        "meta[name='twitter:title']",
    ]:
        m = soup.select_one(sel)
        if m and m.get("content"):
            title = m["content"].strip(); break
    if not title and soup.title and soup.title.string:
        title = soup.title.string.strip()
    desc = ""
    for sel in [
        "meta[property='og:description']",
        "meta[name='description']",
        "meta[name='twitter:description']",
    ]:
        m = soup.select_one(sel)
        if m and m.get("content"):
            desc = m["content"].strip(); break
    paras = []
    for p in soup.find_all("p"):
        txt = (p.get_text(" ", strip=True) or "").strip()
        if len(txt) >= 60:
            paras.append(txt)
        if len(" ".join(paras)) > 3000:
            break
    text = "\n\n".join([desc] + paras).strip()
    return title, text

def extract_main(html: str, base_url: Optional[str] = None) -> Tuple[str, str]:
    try:
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_formatting=False,
            favor_recall=True,
            with_metadata=True,
            output="txt",
            url=base_url,
        ) or ""
    except TypeError:
        try:
            extracted = trafilatura.extract(
                html,
                favor_recall=True,
                url=base_url,
            ) or ""
        except Exception:
            extracted = ""
    title = ""
    try:
        meta = trafilatura.bare_extraction(html, favor_recall=True)
        if meta:
            if isinstance(meta, dict):
                title = (meta.get("title") or "").strip()
            else:
                title = getattr(meta, "title", "") or ""
    except Exception:
        pass
    if len(extracted) < 700:
        og_title, og_text = _og_fallback_text(html)
        title = title or og_title
        text = og_text if len(og_text) > len(extracted) else extracted
    else:
        text = extracted
    return title, text[:60000]

def _ddg_text(query: str, max_results: int = 40) -> List[Dict]:
    out = []
    with DDGS(timeout=20) as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            url = r.get("href") or r.get("url") or ""
            title = r.get("title") or ""
            if not url or not is_probably_article(url, title):
                continue
            out.append({
                "title": title,
                "url": url,
                "snippet": r.get("body", ""),
                "source": "web",
                "ts": None,
            })
    return out

def _ddg_news(query: str, max_results: int = 40) -> List[Dict]:
    out = []
    with DDGS(timeout=20) as ddgs:
        for r in ddgs.news(query, max_results=max_results):
            url = r.get("url") or ""
            title = r.get("title") or ""
            if not url or not is_probably_article(url, title):
                continue
            ts = None
            dt = r.get("date") or r.get("published")
            if isinstance(dt, (int, float)):
                ts = int(dt)
            out.append({
                "title": title,
                "url": url,
                "snippet": r.get("excerpt", ""),
                "source": "news",
                "ts": ts,
            })
    return out

def _ddg_site(query: str, site: str, max_results: int = 30) -> List[Dict]:
    items = _ddg_text(f"{query} site:{site}", max_results=max_results)
    for it in items:
        it["source"] = site
    return items

def web_search(query: str, max_results: int = 25, domain_only: str | None = None) -> List[Dict]:
    dom = (domain_only or "").strip().lower() or None
    q = f"{query} site:{dom}" if dom and "." in dom else query
    items = _ddg_text(q, max_results=max_results)
    if dom and "." in dom:
        items = [i for i in items if dom in i["url"].lower()]
    return dedupe_by_url(items)

def multi_source_search(
    query: str,
    max_results_per_source: int = 30,
    domain_only: str | None = None,
    sources: List[str] | None = None
) -> List[Dict]:
    if not sources:
        sources = ["web", "news", "reddit", "twitter"]
    dom = (domain_only or "").strip().lower() or None
    has_domain = dom and "." in dom
    results: List[Dict] = []
    if has_domain:
        results += _ddg_text(f"{query} site:{dom}", max_results_per_source)
        results += _ddg_news(f"{query} site:{dom}", max_results_per_source // 2)
    else:
        if "web" in sources:
            results += _ddg_text(query, max_results_per_source)
        if "news" in sources:
            results += _ddg_news(query, max_results_per_source // 2)
        if "reddit" in sources:
            results += _ddg_site(query, "reddit.com", max_results_per_source // 2)
        if "twitter" in sources:
            results += _ddg_site(query, "twitter.com", max_results_per_source // 2)
    if has_domain:
        results = [r for r in results if dom in r["url"].lower()]
    selected = set(sources)
    allow_social = ("twitter" in selected) or ("reddit" in selected)
    if "web" in selected and not allow_social:
        results = [r for r in results if not is_social(r["url"])]
    results = [r for r in results if is_probably_article(r["url"], r["title"])]
    results = dedupe_by_url(results)
    return results

def rank_candidates(query: str, items: List[Dict], top_n: int = 10) -> List[Dict]:
    if not items:
        return []
    texts = [f"{it.get('title','')} {it.get('snippet','')}" for it in items]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=8000)
    X = vectorizer.fit_transform([query] + texts)
    qv, M = X[0], X[1:]
    rel = cosine_similarity(qv, M).ravel()
    now = time.time()
    def rec_boost(ts):
        if not ts:
            return 0.0
        days = max(0.0, (now - float(ts)) / 86400.0)
        return math.exp(-days / 30.0)
    scores = []
    for i, it in enumerate(items):
        s = 0.7 * rel[i] + 0.3 * rec_boost(it.get("ts"))
        host = hostname(it.get("url",""))
        if host in WIKI_DOMAINS:
            s *= 0.7
        if it.get("source") == "news":
            s *= 1.05
        it["score"] = float(s)
        scores.append(s)
    order = sorted(range(len(items)), key=lambda i: scores[i], reverse=True)
    picked, seen_hosts = [], set()
    for idx in order:
        host = hostname(items[idx]["url"])
        if host in seen_hosts:
            continue
        seen_hosts.add(host)
        picked.append(items[idx])
        if len(picked) >= top_n:
            break
    return picked
