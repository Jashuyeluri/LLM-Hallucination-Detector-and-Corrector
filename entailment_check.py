bash

cat /home/claude/hallucination_web/entailment_check.py
Output

import os
import time
import requests

# 'hf_api' calls Hugging Face's hosted Inference API remotely - the model
# runs on HF's servers, not this one, so torch/transformers never load
# into this process's memory. This is the deployed/production mode.
# 'local' loads the model in-process via transformers - only use this
# locally where you have enough RAM; torch/transformers are imported
# lazily inside _check_claim_local so they're never touched at all when
# running in hf_api mode.
ENTAILMENT_PROVIDER = os.environ.get("ENTAILMENT_PROVIDER", "local").lower()
HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
HF_NLI_MODEL = os.environ.get("HF_NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
LOCAL_NLI_MODEL = os.environ.get("NLI_MODEL_NAME", "cross-encoder/nli-deberta-v3-small")

HF_API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_NLI_MODEL}"


def check_claim(source_text, claim):
    if ENTAILMENT_PROVIDER == "hf_api":
        return _check_claim_hf_api(source_text, claim)
    else:
        return _check_claim_local(source_text, claim)


def _check_claim_hf_api(source_text, claim, max_retries=4):
    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

    payload = {"inputs": {"text": source_text[:2000], "text_pair": claim}}

    for attempt in range(max_retries):
        try:
            response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(3)
            continue

        if response.status_code == 200:
            data = response.json()
            # response is normally [[{label, score}, ...]] or [{label, score}, ...]
            if isinstance(data, list) and data and isinstance(data[0], list):
                data = data[0]
            scores = {item["label"].lower(): item["score"] for item in data}
            entail_score = scores.get("entailment", 0.0)
            contra_score = scores.get("contradiction", 0.0)
            neutral_score = scores.get("neutral", 0.0)

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

        # model cold-starting on HF's side - wait and retry
        if response.status_code == 503:
            wait = 5
            try:
                wait = min(response.json().get("estimated_time", 5), 20)
            except Exception:
                pass
            time.sleep(wait)
            continue

        response.raise_for_status()

    raise RuntimeError(f"HF Inference API did not respond successfully after {max_retries} attempts")


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
    """Checks a claim against each evidence snippet separately and returns
    the strongest, most decisive result — rather than concatenating all
    snippets into one premise, which can confuse the model when snippets
    describe similar-but-different real events."""
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
