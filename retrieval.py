import os
import requests

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
TAVILY_URL = "https://api.tavily.com/search"

WIKI_HEADERS = {
    "User-Agent": "HallucinationDetectorProject/1.0 (student capstone project; contact: none)"
}


def search_claim_snippets(claim, max_results=5):
    if not TAVILY_API_KEY:
        return []

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": claim,
        "search_depth": "advanced",
        "max_results": max_results,
        "include_answer": True,
        "include_raw_content": False,
    }

    try:
        response = requests.post(TAVILY_URL, json=payload, timeout=25)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Tavily search failed for claim '{claim}': {e}")
        return []

    snippets = []
    if data.get("answer"):
        snippets.append(data["answer"])
    for r in data.get("results", []):
        content = r.get("content", "")
        if content:
            snippets.append(content)

    return snippets


def search_wikipedia_snippets(claim, limit=3):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": claim,
        "format": "json",
        "srlimit": limit,
    }
    try:
        response = requests.get(url, params=params, headers=WIKI_HEADERS, timeout=15)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"Wikipedia search failed for claim '{claim}': {e}")
        return []

    page_titles = [item["title"] for item in data.get("query", {}).get("search", [])]

    snippets = []
    for title in page_titles:
        extract_params = {
            "action": "query",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "titles": title,
            "format": "json",
        }
        try:
            extract_res = requests.get(url, params=extract_params, headers=WIKI_HEADERS, timeout=15)
            extract_res.raise_for_status()
            pages = extract_res.json().get("query", {}).get("pages", {})
            for page in pages.values():
                extract = page.get("extract", "")
                if extract:
                    snippets.append(extract[:1200])
        except Exception:
            continue

    return snippets


def build_live_snippets(claim, use_tavily=True, use_wikipedia=True):
    snippets = []
    if use_tavily and TAVILY_API_KEY:
        for s in search_claim_snippets(claim):
            snippets.append({"text": s, "source": "web"})
    if use_wikipedia:
        for s in search_wikipedia_snippets(claim):
            snippets.append({"text": s, "source": "wikipedia"})
    return snippets
