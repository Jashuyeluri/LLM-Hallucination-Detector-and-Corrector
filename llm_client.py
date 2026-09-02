import os
import re
import time

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

_KNOWN_GROQ_MODELS = {
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
}


def chat(prompt, model=None):
    if LLM_PROVIDER == "groq":
        groq_model = model if model in _KNOWN_GROQ_MODELS else GROQ_MODEL
        return _chat_groq(prompt, groq_model)
    else:
        return _chat_ollama(prompt, model or OLLAMA_MODEL)


def _chat_ollama(prompt, model):
    import ollama
    response = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
    return response['message']['content'].strip()


def _chat_groq(prompt, model, max_retries=5):
    from groq import Groq, RateLimitError

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys")

    client = Groq(api_key=GROQ_API_KEY)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content.strip()
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise
            wait = _extract_retry_seconds(str(e)) or (2 ** attempt)
            time.sleep(wait + 0.5)


def _extract_retry_seconds(error_message):
    match = re.search(r"try again in ([\d.]+)s", error_message)
    if match:
        return float(match.group(1))
    return None
