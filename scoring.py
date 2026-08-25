def compute_faithfulness_score(results):
    if not results:
        return {
            "faithfulness_score": 0.0,
            "total_claims": 0,
            "supported": 0,
            "contradicted": 0,
            "unsupported": 0,
            "unverified": 0,
        }

    total = len(results)
    supported = sum(1 for r in results if r["label"] == "supported")
    contradicted = sum(1 for r in results if r["label"] == "contradicted")
    unsupported = sum(1 for r in results if r["label"] == "unsupported")
    unverified = sum(1 for r in results if r["label"] == "unverified")

    # unverified claims (no evidence found either way) are excluded from
    # the faithfulness calculation entirely, since there's nothing to
    # judge them against - only claims that were actually checked count.
    checked = supported + contradicted + unsupported
    score = round((supported / checked) * 100, 2) if checked else 0.0

    return {
        "faithfulness_score": score,
        "total_claims": total,
        "supported": supported,
        "contradicted": contradicted,
        "unsupported": unsupported,
        "unverified": unverified,
    }
