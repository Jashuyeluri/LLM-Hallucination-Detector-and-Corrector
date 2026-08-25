# LLM Hallucination Detector — Verification Lab

A custom FastAPI + hand-built HTML/CSS/JS dashboard that checks whether an AI-generated response is factually grounded in a source document (or in live web search), then rewrites it to fix what's wrong.

## Structure

```
llm_client.py           unified LLM client — Ollama (local) or Groq (cloud), picked via LLM_PROVIDER
claim_extraction.py     Llama 3 extracts atomic claims from the response
entailment_check.py     DeBERTa-v3-MNLI checks each claim against the source
scoring.py               faithfulness score aggregation
correction.py            Llama 3 rewrites the response using verified facts
retrieval.py             live mode — Tavily + Wikipedia search per claim
pipeline.py              orchestrates document mode and live mode
file_utils.py            reads uploaded .txt/.pdf files
server.py                FastAPI backend + serves the frontend
static/                  dashboard (index.html, style.css, app.js)
Dockerfile               for hosting (Hugging Face Spaces, Render, Railway, etc.)
requirements.txt
```

## Running locally (with Ollama — free, no API key needed)

```bash
pip install -r requirements.txt
ollama pull llama3
ollama serve
uvicorn server:app --reload
```
Open http://127.0.0.1:8000

For live web mode, also set a free Tavily key (https://tavily.com):
```powershell
set TAVILY_API_KEY=your_key_here     # cmd
$env:TAVILY_API_KEY="your_key_here"  # PowerShell
```

## Deploying it as a public website

Ollama cannot run on almost any free hosting platform (no persistent background service allowed), so deployment uses **Groq** instead — same Llama 3 model, called over an API instead of localhost. This is already wired in via `llm_client.py`.

### 1. Get free API keys
- **Groq** (required for deployment): https://console.groq.com/keys
- **Tavily** (optional, only for live web mode): https://tavily.com

### 2. Push this project to GitHub
```bash
cd hallucination_web
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```
(Create the empty repo on github.com first, then run the commands above.)

### 3. Deploy on Hugging Face Spaces (recommended — free, supports Docker, enough RAM for the models)
1. Go to https://huggingface.co/new-space
2. Choose **Docker** as the Space SDK
3. Either connect your GitHub repo, or push directly to the Space's own git remote (HF gives you one on creation)
4. In the Space's **Settings → Variables and secrets**, add:
   - `LLM_PROVIDER` = `groq`
   - `GROQ_API_KEY` = your Groq key
   - `TAVILY_API_KEY` = your Tavily key (optional)
5. The Space will build the `Dockerfile` automatically and give you a public URL like `https://huggingface.co/spaces/<you>/<space-name>`

### Alternative hosts
Render, Railway, and Fly.io all support Dockerfile-based deployment the same way — create a new Web Service, point it at your GitHub repo, and set the same three environment variables in their dashboard's secrets/env settings.

## Notes
- First run downloads `MoritzLaurer/DeBERTa-v3-base-mnli` (~370MB) automatically.
- Free hosting tiers run on CPU, so entailment checking will be slower than on your local machine — this is expected.
- Never commit API keys to GitHub — they're read from environment variables / host secrets only.
