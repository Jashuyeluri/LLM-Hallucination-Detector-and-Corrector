from claim_extraction import extract_claims
from entailment_check import check_all_claims, check_claim_against_snippets
from scoring import compute_faithfulness_score
from correction import correct_response
from retrieval import build_live_snippets


def run_pipeline(source_text, llm_response, model='llama3', auto_correct=True):
    claims = extract_claims(llm_response, model=model)
    results = check_all_claims(source_text, claims)
    score = compute_faithfulness_score(results)

    corrected_response = None
    if auto_correct and (score["contradicted"] > 0 or score["unsupported"] > 0):
        corrected_response = correct_response(source_text, llm_response, results, model=model)

    return {
        "claims": claims,
        "results": results,
        "score": score,
        "corrected_response": corrected_response,
    }


def run_pipeline_live(llm_response, model='llama3', auto_correct=True):
    """Same as run_pipeline, but fetches live evidence per claim from the web
    (Tavily + Wikipedia) instead of using a document you provide.

    Each claim is checked against its evidence snippets INDIVIDUALLY rather
    than one merged blob of all snippets — this matters when search returns
    genuinely relevant results about a similar-but-different real event
    (e.g. a company's two different product launches), since merging them
    into one premise can confuse both the entailment check and the
    correction rewrite. Only the single strongest/most decisive snippet per
    claim is kept and used going forward.

    Claims with no evidence found at all are labelled 'unverified' rather
    than 'unsupported' — absence of search results is not evidence the
    claim is wrong, so these are kept as-is during correction instead of
    being deleted."""
    claims = extract_claims(llm_response, model=model)

    results = []
    best_snippets = []
    for claim in claims:
        snippets = build_live_snippets(claim)
        result = check_claim_against_snippets(snippets, claim)
        results.append(result)
        if result.get("best_snippet"):
            best_snippets.append(f"Regarding \"{claim}\": {result['best_snippet']}")

    score = compute_faithfulness_score(results)
    # distilled, claim-labelled fact list instead of one giant raw blob —
    # much less likely to confuse the correction model with mixed context
    distilled_source = "\n".join(best_snippets)

    corrected_response = None
    if auto_correct and (score["contradicted"] > 0 or score["unsupported"] > 0):
        corrected_response = correct_response(distilled_source, llm_response, results, model=model)

    return {
        "claims": claims,
        "results": results,
        "score": score,
        "corrected_response": corrected_response,
        "retrieved_source": distilled_source,
    }
