"""Pure CPU rules for adaptive-A; no model, GPU, labels or external services."""
import math
import re

SCORE_MAPPING = {"a": 3.0, "b": 2.5, "c": 2.0, "d": 1.0, "e": 0.5, "f": 0.001}
OPTIONS = """(a) Very Likely (Around 90%)
(b) Likely (Around 70%)
(c) Somewhat Likely (Around 60%)
(d) Somewhat Unlikely (Around 25%)
(e) Unlikely (Around 20%)
(f) Very Unlikely (Below 10%)"""


class FormatError(ValueError):
    pass


def parse_initial_list(raw):
    items, numbers = [], []
    for line in raw.strip().splitlines():
        if not line.strip():
            continue
        match = re.match(r"^\s*(\d+)[.)]\s+(.+?)\s*$", line)
        if match:
            numbers.append(int(match[1]))
            items.append(match[2].strip())
        elif items and line[:1].isspace() and not re.match(r"^\s*\d+[.)]", line):
            items[-1] += "\n" + line.strip()
        else:
            raise FormatError("Expected only a numbered list, starting at 1, with nonempty items")
    if not 1 <= len(items) <= 5 or numbers != list(range(1, len(items) + 1)):
        raise FormatError(f"Expected 1..5 consecutive numbered hypotheses; got {len(items)}")
    return items


def parse_option(raw):
    # Same Answer: split and first matching option test as fixed-4 RecordingTracer.
    answer = raw.split("Answer:")[-1].strip()
    for option in "abcdef":
        if answer.startswith(f"({option})") or answer.startswith(f"{option})") or answer.startswith(f"{option}.") or answer.startswith(f"{option} ") or f"({option})" in answer or f" {option})" in answer or f" {option}." in answer or option == answer:
            return option
    raise FormatError("Unparseable six-option evaluation")


def softmax_scores(scores, expected_n):
    if len(scores) != expected_n or len(scores) == 0:
        raise ValueError("Score/ID length mismatch or empty candidate set")
    values = [float(value) for value in scores]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Non-finite hypothesis score")
    # The fixed score range is bounded. This is identical to its np.exp normalization.
    exp = [math.exp(value) for value in values]
    return [value / sum(exp) for value in exp]


def prune_once(ids, weights, threshold=0.10):
    if not ids or len(ids) != len(weights) or len(set(ids)) != len(ids):
        raise ValueError("Invalid IDs or weight/ID length mismatch")
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("Invalid pruning threshold")
    if not all(math.isfinite(float(w)) and w >= 0 for w in weights) or abs(sum(weights) - 1) > 1e-8:
        raise ValueError("Weights must be finite, nonnegative and normalized")
    keep = [i for i, w in enumerate(weights) if w >= threshold]
    guarded = not keep
    if guarded:
        keep = [max(range(len(weights)), key=lambda i: weights[i])]
    mass = sum(weights[i] for i in keep)
    if mass <= 0:
        raise ValueError("Surviving mass is zero")
    return {"keep_indices": keep, "kept_ids": [ids[i] for i in keep], "deleted_ids": [h for i, h in enumerate(ids) if i not in keep], "weights_after": [weights[i] / mass for i in keep], "minimum_one_guard": guarded}


def summary_indices(weights, top_p):
    ordered = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
    chosen, mass = [], 0.0
    for i in ordered:
        chosen.append(i)
        mass += weights[i]
        if mass >= top_p:
            break
    return chosen
