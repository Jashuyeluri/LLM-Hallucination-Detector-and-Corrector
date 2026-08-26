import os

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


def _chat_groq(prompt, model):
    from groq import Groq
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys")
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()
