import os

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-70b-8192")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")


def chat(prompt, model=None):
    """Sends a single-turn prompt to whichever LLM provider is configured
    and returns the response text. Set LLM_PROVIDER=groq (with GROQ_API_KEY)
    for cloud/hosted deployments, or leave as 'ollama' for local use."""
    if LLM_PROVIDER == "groq":
        return _chat_groq(prompt, model or GROQ_MODEL)
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
