from llm_client import chat


def correct_response(source_text, llm_response, results, model=None):
    supported = [r["claim"] for r in results if r["label"] == "supported"]
    contradicted = [r["claim"] for r in results if r["label"] == "contradicted"]
    unsupported = [r["claim"] for r in results if r["label"] == "unsupported"]
    unverified = [r["claim"] for r in results if r["label"] == "unverified"]

    prompt = f"""You are correcting a factually flawed summary using verified source material.

Source material (ground truth — use the specific facts stated here):
{source_text}

Original response (may contain errors):
{llm_response}

The following claims from the response were checked against the source:

SUPPORTED (keep exactly as-is):
{chr(10).join('- ' + c for c in supported) if supported else '(none)'}

CONTRADICTED — factually wrong. The source material above contains the correct fact for each of these; find it and use it exactly, do not soften it into a vague or hedged statement:
{chr(10).join('- ' + c for c in contradicted) if contradicted else '(none)'}

UNSUPPORTED — the source actively conflicts with or does not back this claim, remove it:
{chr(10).join('- ' + c for c in unsupported) if unsupported else '(none)'}

UNVERIFIED — no evidence was found either way; this is NOT proven wrong, so keep it in the response as originally written, unchanged:
{chr(10).join('- ' + c for c in unverified) if unverified else '(none)'}

Rewrite the response as ONE corrected paragraph. Rules:
1. Keep all SUPPORTED and UNVERIFIED claims, close to their original wording.
2. For each CONTRADICTED claim, replace it with the specific, concrete correct fact found in the source material above (a specific date, name, or number) — never write phrases like "a yet-unknown date" or "an unspecified location." If the source material genuinely gives no specific correct value for a contradicted claim, state only what it does confirm, without inventing vague filler.
3. Only remove UNSUPPORTED claims — do not remove or shorten anything else.
4. Preserve as much of the original response's detail, structure, and length as possible. Do not summarize or compress beyond what correcting the wrong facts requires.
5. Write it as natural, connected prose.

Return ONLY the corrected paragraph, no explanations, no headers."""

    return chat(prompt, model=model)
