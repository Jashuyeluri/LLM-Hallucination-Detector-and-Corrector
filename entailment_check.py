import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_tokenizer = None
_model = None
_label2id = None
MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli"


def _load():
    global _tokenizer, _model, _label2id
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.eval()
        _label2id = {v.lower(): k for k, v in _model.config.id2label.items()}
    return _tokenizer, _model, _label2id


def check_claim(source_text, claim):
    tokenizer, model, label2id = _load()

    inputs = tokenizer.encode(
        source_text, claim,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    with torch.no_grad():
        logits = model(inputs)[0]

    probs = torch.softmax(logits, dim=1)[0]
    contra_score = probs[label2id["contradiction"]].item()
    neutral_score = probs[label2id["neutral"]].item()
    entail_score = probs[label2id["entailment"]].item()

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
    describe similar-but-different real events (e.g. two versions of a
    product, or the same company's two different launches)."""
    if not snippets:
        return {
            "claim": claim,
            "label": "unverified",
            "entailment_score": 0.0,
            "contradiction_score": 0.0,
            "neutral_score": 0.0,
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
            "claim": claim,
            "label": "unverified",
            "entailment_score": 0.0,
            "contradiction_score": 0.0,
            "neutral_score": 0.0,
            "best_snippet": None,
        }

    # pick whichever snippet gave the single most decisive signal in either
    # direction (highest entailment OR highest contradiction), since that's
    # the piece of evidence that actually speaks to this specific claim
    best = max(per_snippet_results, key=lambda r: max(r["entailment_score"], r["contradiction_score"]))

    return {
        "claim": best["claim"],
        "label": best["label"],
        "entailment_score": best["entailment_score"],
        "contradiction_score": best["contradiction_score"],
        "neutral_score": best["neutral_score"],
        "best_snippet": best["snippet_text"][:300],
    }
