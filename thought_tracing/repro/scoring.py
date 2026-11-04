"""Conservative final-output-only grading fixed before model execution."""
import re

AMBIGUOUS = re.compile(r"\b(or|either|and|but|whereas|however|maybe|might|perhaps|probably|uncertain|unknown|not|cannot|can't|don't|doesn't)\b", re.I)


def score_final(text, row, truncated=False):
    official = row["answer"].lower() in text.lower()
    surface = re.sub(r"^\s*Answer\s*:\s*", "", text.strip(), flags=re.I).replace("**", "").replace("__", "")
    masked = surface.lower()
    mentions = []
    for candidate in sorted(row["candidate_answer_locations"], key=lambda s: (-len(s), s)):
        pattern = re.compile(r"(?<!\w)" + re.escape(candidate.lower()) + r"(?!\w)")
        if pattern.search(masked):
            mentions.append(candidate)
            masked = pattern.sub(lambda m: " " * len(m.group()), masked)
    sentences = [s for s in re.split(r"[.!?]+(?:\s+|$)", surface) if s.strip()]
    reasons = []
    if not surface:
        reasons.append("empty_final")
    if len(mentions) != 1:
        reasons.append("multiple_locations" if len(mentions) > 1 else "no_known_location")
    if len(sentences) > 1:
        reasons.append("multiple_sentences")
    if AMBIGUOUS.search(surface):
        reasons.append("ambiguous_or_negated")
    if re.search(r"[,;]", surface):
        reasons.append("coordination_or_list_requires_review")
    if truncated:
        reasons.append("final_output_at_token_limit")
    strict = None if reasons else mentions[0].lower() == row["answer"].lower()
    return {"official_substring": official, "strict_final_only": strict, "candidate_mentions": mentions, "human_review_required": bool(reasons), "review_reasons": reasons}
