import os
import re
import json
from llm_client import chat

ENTAILMENT_PROVIDER = os.environ.get("ENTAILMENT_PROVIDER", "llm_judge").lower()
LOCAL_NLI_MODEL = os.environ.get("NLI_MODEL_NAME", "cross-encoder/nli-deberta-v3-small")


def check_claim(source_text, claim):
    if ENTAILMENT_PROVIDER == "local":
        return _check_claim_local(source_text, claim)
    else:
        return _check_claim_llm_judge(source_text, claim)


def _check_claim_llm_judge(source_text, claim):
    prompt = f"""You are a strict fact-checking judge. Given a SOURCE text and a CLAIM, decide the relationship between them.

SOURCE:
{source_text[:3000]}

CLAIM:
{claim}

Respond with ONLY a single-line JSON object and nothing else - no explanations, no comments, no text before or after it, in exactly this format:
{{"label": "entailment", "entailment_score": 0.95, "contradiction_score": 0.02, "neutral_score": 0.03}}

Rules:
- "label" must be exactly one of: "entailment", "contradiction", or "neutral".
- The three scores must be numbers between 0 and 1, summing to approximately 1.0.
- Base this ONLY on what the source text actually says, not on outside knowledge.
- Output ONLY the JSON object on a single line. Do not add any commentary anywhere."""

    raw = chat(prompt).strip()
    entail_score, contra_score, neutral_score = _parse_scores(raw)

    if entail_score > contra_score and entail_score > neutral_score:
        label = "supported"
    elif contra_score > entail_score and contra_score > neutral_score:
        label = "contradicted"
    else:
        label = "unsupported"

    return {
        "claim": claim,
        "label": label,
        "entailment_score": round(entail_score, 4),
        "contradiction_score": round(contra_score, 4),
        "neutral_score": round(neutral_score, 4),
    }


def _parse_scores(raw):
    """Tries strict JSON parsing first, then falls back to regex extraction
    if the model's output has stray text or minor syntax issues breaking
    strict JSON parsing."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]

    try:
        data = json.loads(cleaned)
        return (
            float(data.get("entailment_score", 0.0)),
            float(data.get("contradiction_score", 0.0)),
            float(data.get("neutral_score", 0.0)),
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    try:
        start = cleaned.find('{')
        end = cleaned.rfind('}') + 1
        data = json.loads(cleaned[start:end])
        return (
            float(data.get("entailment_score", 0.0)),
            float(data.get("contradiction_score", 0.0)),
            float(data.get("neutral_score", 0.0)),
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # last resort: pull each score out directly with regex, ignoring
    # anything else the model may have written around them
    def find_score(key):
        m = re.search(rf'"{key}"\s*:\s*([\d.]+)', raw)
        return float(m.group(1)) if m else 0.0

    entail = find_score("entailment_score")
    contra = find_score("contradiction_score")
    neutral = find_score("neutral_score")

    if entail == 0.0 and contra == 0.0 and neutral == 0.0:
        # nothing usable was found at all - treat as neutral/unclear rather
        # than crashing the whole request
        return (0.0, 0.0, 1.0)

    return (entail, contra, neutral)


_local_tokenizer = None
_local_model = None
_local_label2id = None


def _check_claim_local(source_text, claim):
    global _local_tokenizer, _local_model, _local_label2id
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if _local_model is None:
        _local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_NLI_MODEL)
        _local_model = AutoModelForSequenceClassification.from_pretrained(LOCAL_NLI_MODEL)
        _local_model.eval()
        _local_label2id = {v.lower(): k for k, v in _local_model.config.id2label.items()}

    inputs = _local_tokenizer.encode(
        source_text, claim, return_tensors="pt", truncation=True, max_length=512,
    )
    with torch.no_grad():
        logits = _local_model(inputs)[0]
    probs = torch.softmax(logits, dim=1)[0]

    contra_score = probs[_local_label2id["contradiction"]].item()
    neutral_score = probs[_local_label2id["neutral"]].item()
    entail_score = probs[_local_label2id["entailment"]].item()

    if entail_score > contra_score and entail_score > neutral_score:
        label = "supported"
    elif contra_score > entail_score and contra_score > neutral_score:
        label = "contradicted"
    else:
        label = "unsupported"

    return {
        "claim": claim,
        "label": label,
        "entailment_score": round(entail_score, 4),
        "contradiction_score": round(contra_score, 4),
        "neutral_score": round(neutral_score, 4),
    }


def check_all_claims(source_text, claims):
    return [check_claim(source_text, c) for c in claims]


def check_claim_against_snippets(snippets, claim):
    if not snippets:
        return {
            "claim": claim, "label": "unverified",
            "entailment_score": 0.0, "contradiction_score": 0.0, "neutral_score": 0.0,
            "best_snippet": None,
        }

    per_snippet_results = []
    for snip in snippets:
        text = snip["text"] if isinstance(snip, dict) else snip
        if not text or not text.strip():
            continue
        r = check_claim(text, claim)
        r["snippet_text"] = text
        per_snippet_results.append(r)

    if not per_snippet_results:
        return {
            "claim": claim, "label": "unverified",
            "entailment_score": 0.0, "contradiction_score": 0.0, "neutral_score": 0.0,
            "best_snippet": None,
        }

    best = max(per_snippet_results, key=lambda r: max(r["entailment_score"], r["contradiction_score"]))

    return {
        "claim": best["claim"],
        "label": best["label"],
        "entailment_score": best["entailment_score"],
        "contradiction_score": best["contradiction_score"],
        "neutral_score": best["neutral_score"],
        "best_snippet": best["snippet_text"][:300],
    }
