"""Independent stdlib-only, read-only audit of saved adaptive-A execution logs."""
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "records/publication_20261104_chunked_v1"
RUN = ROOT / "experiments/adaptive_a_v2/runs/adaptive_a_v2/cohort"
SCORES = dict(zip("abcdef", [3, 2.5, 2, 1, .5, .001]))


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def near(a, b):
    assert len(a) == len(b)
    assert all(math.isclose(x, y, rel_tol=0, abs_tol=1e-12) for x, y in zip(a, b))


def option(raw):
    answer = raw.split("Answer:")[-1].strip()
    for o in "abcdef":
        if (answer.startswith((f"({o})", f"{o})", f"{o}.", f"{o} "))
                or any(s in answer for s in [f"({o})", f" {o})", f" {o}."])
                or o == answer):
            return o
    return None


def main():
    original = read(ROOT / "experiments/adaptive_a_v2/records/fixed4_protection_before.json")
    protected = original["protected_sha256"]
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in protected.items())
    counts, evals, deleted, first_prune = Counter(), Counter(), Counter(), Counter()
    initial_sizes, last_sizes, successful_sizes, summaries, failed = [], [], [], [], []
    pruned, single = set(), set()
    for d in sorted(RUN.glob("row-*")):
        if not (d / "result.json").exists():
            continue
        result = read(d / "result.json")
        rid = result["row_id"]
        counts["result_records"] += 1
        counts["status:" + result["status"]] += 1
        events = [e for e in lines(d / "trace_events.jsonl") if e.get("case_attempt_id") == result["case_attempt_id"]]
        assert not any(e["event"] in ["resample", "rejuvenate"] for e in events)
        if result.get("tt_bypassed"):
            counts["bypassed"] += 1
            assert not any(e["event"] == "adaptive_step" for e in events)
            continue
        initials = [e for e in events if e["event"] == "adaptive_initial_format" and e["valid"]]
        assert len(initials) == 1
        init = initials[0]
        n = init["initial_count"]
        assert 1 <= n <= 5 and len(init["ids"]) == n and len(set(init["ids"])) == n
        assert len(init["parsed"]) == n and all(isinstance(t, str) and t.strip() for t in init["parsed"])
        near(init["initial_weights"], [1/n] * n)
        initial_sizes.append(n)
        counts["initialized"] += 1
        counts["initial_format_repairs"] += sum(e["event"] == "adaptive_initial_format" and e.get("attempt", 0) > 1 for e in events)
        steps = [e for e in events if e["event"] == "adaptive_step"]
        before_ids, previous_texts, previous_weights = init["ids"], None, init["initial_weights"]
        dead = set()
        for index, s in enumerate(steps):
            ids, after = s["ids_before"], s["ids_after"]
            assert s["step"] == index and ids == before_ids and not dead.intersection(ids)
            assert len(ids) == len(s["texts_before"]) == len(s["raw_scores"]) == len(s["raw_evaluations"])
            assert all(isinstance(t, str) and t.strip() for t in s["texts_before"])
            assert len(ids) == s["count_before_evaluation"] == s["hypothesis_evaluations"]
            assert len(after) == s["count_after_pruning"] >= 1 and len(after) <= len(ids)
            if previous_texts is not None:
                assert s["previous_texts_by_id"] == dict(zip(ids, previous_texts))
            near(s["previous_weights"], previous_weights)
            expected_type = "action_likelihood" if s["state_action"]["action"] else "state_only_plausibility"
            assert s["evaluator_type"] == expected_type
            opts = [option(raw) for raw in s["raw_evaluations"]]
            assert opts == s["parsed_options"] and all(o in SCORES for o in opts)
            scores = [SCORES[o] for o in opts]
            near(scores, s["raw_scores"])
            exponentials = [math.exp(x - max(scores)) for x in scores]
            weights = [v / sum(exponentials) for v in exponentials]
            near(weights, s["weights_before_pruning"])
            assert s["prune_threshold"] == .10 and s["prune_comparison"] == "strictly_less_than"
            assert s["prune_passes"] == 1 and s["resampling"] is False and s["rejuvenation"] is False
            keep = [i for i, w in enumerate(s["weights_before_pruning"]) if w >= .10]
            guard = not keep
            if guard:
                keep = [max(range(len(ids)), key=lambda i: s["weights_before_pruning"][i])]
            assert s["minimum_one_guard"] == guard
            assert after == [ids[i] for i in keep]
            assert s["texts_after"] == [s["texts_before"][i] for i in keep]
            removed = [v for i, v in enumerate(ids) if i not in keep]
            assert s["deleted_ids"] == removed
            denom = sum(s["weights_before_pruning"][i] for i in keep)
            near(s["weights_after_pruning"], [s["weights_before_pruning"][i] / denom for i in keep])
            if removed:
                if rid not in pruned:
                    first_prune[index] += 1
                pruned.add(rid)
            if len(after) == 1:
                single.add(rid)
            dead.update(removed)
            before_ids, previous_texts, previous_weights = after, s["texts_after"], s["weights_after_pruning"]
            counts["completed_step_records"] += 1
            counts["logical_hypothesis_evaluations"] += len(ids)
            evals[expected_type] += 1
            deleted[expected_type] += len(removed)
        if steps:
            last_sizes.append(len(before_ids))
            if result["status"] == "ok":
                successful_sizes.append(len(before_ids))
        for s in [e for e in events if e["event"] == "adaptive_summary_selection"]:
            previous = steps[s["summary_for_step"]]
            assert s["survivor_ids"] == previous["ids_after"]
            assert s["summary_top_p"] == .9
            ranked = sorted(range(len(s["survivor_ids"])), key=lambda i: previous["weights_after_pruning"][i], reverse=True)
            chosen, total = [], 0
            for i in ranked:
                chosen.append(s["survivor_ids"][i])
                total += previous["weights_after_pruning"][i]
                if total >= .9:
                    break
            assert s["summary_ids"] == chosen and s["summary_count"] == len(chosen)
            summaries.append(s["summary_count"])
        if result["status"] != "ok":
            calls = lines(d / "calls.jsonl")
            end_calls = [c for c in calls if c["event"] == "call_end"]
            last_stage = end_calls[-1]["stage"]
            tail = []
            for c in reversed(end_calls):
                if c["stage"] != last_stage:
                    break
                tail.append(c)
            bad = [c for c in reversed(tail) if option(c["raw_output"]) is None]
            failed.append({"row_id": rid, "error": result.get("error"), "last_stage": last_stage,
                           "unparseable_responses": [{"ordinal": c["ordinal"], "raw_output": c["raw_output"], "truncated": c["truncated"]} for c in bad],
                           "last_recorded_survivors": len(before_ids), "completed_steps_before_failure": len(steps)})
    assert counts["result_records"] == 600 and counts["bypassed"] == 200 and counts["initialized"] == 400
    assert counts["status:ok"] == 595 and counts["status:error"] == 5
    value = {"audit_status": "passed", "generated_at": datetime.now().astimezone().isoformat(),
             "scope": "Read-only persisted-log invariants for latest case_attempt_id; no model inference or score retuning",
             "counts": dict(counts), "completed_steps_by_evaluator": dict(evals), "deleted_by_evaluator": dict(deleted),
             "initial_count_distribution": dict(Counter(initial_sizes)), "mean_initial_count_all_initialized": mean(initial_sizes),
             "last_recorded_count_examples": len(last_sizes), "mean_last_recorded_count_including_partial_errors": mean(last_sizes),
             "completed_trace_count": len(successful_sizes), "mean_final_count_successful_traces_only": mean(successful_sizes),
             "summary_records": len(summaries), "mean_summary_count": mean(summaries),
             "pruned_examples": len(pruned), "first_prune_step_distribution_zero_based": dict(first_prune),
             "started_or_became_single": len(single), "failed_examples": failed,
             "protected_original_files_verified": len(protected),
             "empty_initial_or_recorded_step_hypotheses": 0,
             "limits": ["Persisted logs cannot prove object immutability; existing CPU mock test covers that property",
                        "No semantic validation of target, action labels, perceptions or hypothesis correctness",
                        "Error examples may contain partial trajectories; reported last count is not a completed final count",
                        "Snapshot logs do not by themselves prove absence of all unlogged operations"]}
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "adaptive_log_audit.json"
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in value.items() if k != "failed_examples"}, ensure_ascii=False, indent=2))
    print("Failure summaries:")
    for e in failed:
        print(json.dumps({**e, "unparseable_responses": [{**c, "raw_output": c["raw_output"].splitlines()[0]} for c in e["unparseable_responses"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
