"""Read-only comparison of fixed results; writes only adaptive reports."""
from collections import Counter
from pathlib import Path
import json
from statistics import mean
import time

from .common import ROOT, FIXED, call_metrics, canonical, digest, lines, read, save


def normal(result):
    return bool(result and result.get("status") == "ok" and not result.get("blocking_format_errors") and not result.get("metrics", {}).get("truncated_calls"))


def score_group(rows, results):
    found = [results[row["row_id"]] for row in rows if row["row_id"] in results]
    correct = sum(r.get("grading", {}).get("official_substring") is True for r in found)
    strict = sum(r.get("grading", {}).get("strict_final_only") is True for r in found)
    return {"planned": len(rows), "attempted": len(found), "incomplete": len(rows) - len(found), "execution_failures": sum(r["status"] != "ok" for r in found), "format_failures": sum(bool(r.get("blocking_format_errors")) for r in found), "truncated_examples": sum(bool(r.get("metrics", {}).get("truncated_calls")) for r in found), "official_correct": correct, "official_accuracy": correct / len(rows) if rows and len(found) == len(rows) else None, "strict_correct": strict, "strict_accuracy": strict / len(rows) if rows and len(found) == len(rows) else None, "human_final_review": sum(r.get("grading", {}).get("human_review_required", False) for r in found)}


def aggregate_compute(directories):
    metrics = [call_metrics(path) for path in directories]
    keys = ["calls", "failed_calls", "input_tokens", "output_tokens", "total_tokens", "call_wall_seconds", "truncated_calls", "partial_output_tokens_resource_interruptions"]
    result = {key: sum(m[key] for m in metrics) for key in keys}
    stages = Counter()
    for path in directories:
        for call in lines(path / "calls.jsonl"):
            if call["event"] == "call_end":
                stages[call["stage"]] += 1
    result["actual_inference_requests_by_stage_all_attempts"] = dict(stages)
    result["actual_evaluation_inference_requests_all_attempts"] = stages["tt.likelihood"] + stages["adaptive.state_only_plausibility"]
    result["batch_interface_calls_all_attempts"] = sum(e["event"] == "batch_interface_end" for path in directories for e in lines(path / "batch_events.jsonl"))
    result["failed_generation_tokens_may_be_unknown"] = any(m["failed_generated_token_counts_may_be_unknown"] for m in metrics)
    return result


def analyze(limit=None, output_directory=None):
    manifest_path = ROOT / "manifests/adaptive_cohort.json"
    if not manifest_path.exists():
        return {"status": "awaiting_cohort_confirmation"}
    manifest = read(manifest_path)
    rows = manifest["rows"]
    total_planned = len(rows)
    if limit is not None:
        if not 1 <= limit <= total_planned:
            raise ValueError("Invalid report prefix size")
        rows = rows[:limit]
    frozen = read(ROOT / "records/adaptive_frozen_identity.json")
    run = ROOT / "runs" / frozen["config"]["run_version"]
    results = {"fixed-4": {}, "adaptive-A": {}}
    paths = {"fixed-4": [], "adaptive-A": []}
    step_events, initial_counts, final_counts, first_prunes, deleted_by_type = [], [], [], [], Counter()
    pruned_ids, single_ids, initialized_ids = [], [], []
    summary_sizes, predictions, manual = [], [], []
    for row in rows:
        rid = row["row_id"]
        for method, directory in (("fixed-4", FIXED / "runs/bf16_official600_v1/official600" / f"row-{rid:04d}" / "TT"), ("adaptive-A", run / "cohort" / f"row-{rid:04d}")):
            if (directory / "calls.jsonl").exists():
                paths[method].append(directory)
            if not (directory / "result.json").exists():
                continue
            value = read(directory / "result.json")
            assert value["row_id"] == rid
            if method == "fixed-4":
                assert value["manifest_sha256"] == manifest["fixed_manifest_sha256"]
            else:
                assert value["run_identity_sha256"] == canonical(frozen)
            results[method][rid] = value
            predictions.append({"row_id": rid, "method": method, "qtype": row["qtype"], "stratum": row["stratum"], "gold": row["answer"], "prediction": value.get("prediction"), "status": value["status"], "grading": value.get("grading")})
            if value.get("grading", {}).get("human_review_required") or not normal(value):
                manual.append({"row_id": rid, "method": method, "prediction": value.get("prediction"), "reasons": value.get("grading", {}).get("review_reasons", []), "intermediate_review_completed": False})
            if method != "adaptive-A":
                continue
            events = [e for e in lines(directory / "trace_events.jsonl") if e.get("case_attempt_id") == value["case_attempt_id"]]
            steps = [e for e in events if e["event"] == "adaptive_step"]
            initials = [e for e in events if e["event"] == "adaptive_initial_format" and e["valid"]]
            if initials:
                initial_counts.append(initials[0]["initial_count"])
                initialized_ids.append(rid)
            if steps:
                final_counts.append(steps[-1]["count_after_pruning"])
                prunes = [e for e in steps if e["deleted_ids"]]
                if prunes:
                    pruned_ids.append(rid)
                    first_prunes.append(prunes[0]["step"])
                if any(e["count_after_pruning"] == 1 for e in steps):
                    single_ids.append(rid)
                for e in steps:
                    deleted_by_type[e["evaluator_type"]] += len(e["deleted_ids"])
                step_events.extend(steps)
            summary_sizes.extend(e["summary_count"] for e in events if e["event"] == "adaptive_summary_selection")
    groups = {"all": rows}
    for name in sorted({r["stratum"] for r in rows}):
        groups[name] = [r for r in rows if r["stratum"] == name]
    for name, predicate in (("belief", lambda r: bool(r.get("order"))), ("false_belief", lambda r: r.get("belief") == "false"), ("true_belief", lambda r: r.get("belief") == "true"), ("order1", lambda r: r.get("order") == 1), ("order2", lambda r: r.get("order") == 2)):
        groups[name] = [r for r in rows if predicate(r)]
    for qtype in sorted({r["qtype"] for r in rows}):
        groups[f"qtype:{qtype}"] = [r for r in rows if r["qtype"] == qtype]
    group_names = [name for name in groups if name != "all"]
    methods = {method: {name: score_group(selection, values) for name, selection in groups.items()} for method, values in results.items()}
    paired = [row for row in rows if all(normal(results[m].get(row["row_id"])) for m in results)]
    only_fixed, only_adaptive = [], []
    for row in paired:
        rid = row["row_id"]
        f = results["fixed-4"][rid]["grading"]["official_substring"]
        a = results["adaptive-A"][rid]["grading"]["official_substring"]
        if f and not a: only_fixed.append(rid)
        if a and not f: only_adaptive.append(rid)
    resources = lines(run / "resource_samples.jsonl")
    complete = all(len(values) == len(rows) for values in results.values())
    adaptive_stats = {"initialized_examples": len(initial_counts), "initial_count_distribution": dict(Counter(initial_counts)), "mean_initial_count": mean(initial_counts) if initial_counts else None, "mean_final_count": mean(final_counts) if final_counts else None, "mean_pre_prune_count_over_steps": mean([e["count_before_evaluation"] for e in step_events]) if step_events else None, "mean_survivors_over_steps": mean([e["count_after_pruning"] for e in step_events]) if step_events else None, "mean_summary_hypothesis_count": mean(summary_sizes) if summary_sizes else None, "pruned_examples": len(pruned_ids), "pruned_example_ids": pruned_ids, "pruned_fraction_of_initialized": len(pruned_ids) / len(initial_counts) if initial_counts else None, "pruned_fraction_of_all_attempted": len(pruned_ids) / len(results["adaptive-A"]) if results["adaptive-A"] else None, "first_prune_step_distribution_zero_based": dict(Counter(first_prunes)), "became_or_started_single_examples": len(single_ids), "became_or_started_single_fraction_of_initialized": len(single_ids) / len(initial_counts) if initial_counts else None, "single_example_ids": single_ids, "deleted_hypotheses_by_evaluator_type": dict(deleted_by_type), "logical_completed_step_hypothesis_evaluations": sum(e["hypothesis_evaluations"] for e in step_events), "logical_completed_step_batch_interface_calls": len(step_events), "trajectory_step_count": len(step_events), "bypassed_examples": sum(r.get("tt_bypassed", False) for r in results["adaptive-A"].values()), "summary_top_p_distinct_from_pruning": True}
    value = {"generated_unix": time.time(), "status": "complete" if complete else "incomplete", "cohort_size": len(rows), "cohort_scope": manifest["selection_policy"], "fixed4_results_read_only": True, "model": frozen["config"]["model_id"], "model_revision": frozen["config"]["model_revision"], "manifest_sha256": digest(manifest_path), "methods_all_attempts": methods, "paired_normal_completed_count": len(paired), "paired_metrics": {m: score_group(paired, results[m]) for m in results}, "paired_only_fixed_correct_ids": only_fixed, "paired_only_adaptive_correct_ids": only_adaptive, "adaptive_stats": adaptive_stats, "compute_cohort_all_attempts": {m: aggregate_compute(paths[m]) for m in paths}, "compute_adaptive_smoke_and_connectivity": aggregate_compute([p.parent for p in run.glob("smoke/row-*/calls.jsonl")] + ([run / "connectivity"] if (run / "connectivity/calls.jsonl").exists() else [])), "fixed_overlap_detected": any(e.get("fixed_overlap") for e in resources), "resource_sample_count": len(resources), "gpu_used_peak_mib_whole_device": max((g["used_mib"] for e in resources for g in e.get("gpu", [])), default=None), "manual_review_queue": manual, "intermediate_semantic_review": "pending; final correctness does not verify intermediate hypotheses", "limits": ["variable initial count + pruning + disabled resampling/rejuvenation + state-only evaluation are a bundled intervention", "deleted hypotheses cannot return", "one remaining hypothesis always has normalized weight 1", "relative weight depends on candidate count and cannot detect all candidates being poor", "action likelihood and state-only plausibility share labels but are not calibrated to the same probability", "same seed does not guarantee identical outputs with different RNG consumption", "local Qwen7B cohort is not reproduction of all paper benchmark/model conditions", "overlapping GPU workloads invalidate wall-clock-only speedup claims; whole-device samples can include other applications"]}
    by_step = {}
    for step in sorted({e["step"] for e in step_events}):
        selected = [e for e in step_events if e["step"] == step]
        by_step[str(step)] = {"examples_reaching_step": len(selected), "mean_before_evaluation": mean(e["count_before_evaluation"] for e in selected), "mean_after_pruning": mean(e["count_after_pruning"] for e in selected)}
    adaptive_stats["mean_counts_by_trajectory_step"] = by_step
    lifecycle = lines(run / "run_lifecycle.jsonl")
    starts = [e for e in lifecycle if e["event"] == "run_start"]
    ends = [e for e in lifecycle if e["event"] == "run_end"]
    active_seconds = sum(max(0, end["time_unix"] - next((start["time_unix"] for start in reversed(starts) if start["time_unix"] <= end["time_unix"]), end["time_unix"])) for end in ends)
    latest_start = max((e["time_unix"] for e in starts), default=None)
    latest_end = max((e["time_unix"] for e in ends), default=None)
    open_attempt_seconds = max(0, value["generated_unix"] - latest_start) if latest_start is not None and (latest_end is None or latest_start > latest_end) else 0
    value["adaptive_worker_time"] = {"closed_attempt_active_wall_seconds": active_seconds, "first_start_unix": min((e["time_unix"] for e in starts), default=None), "last_end_unix": max((e["time_unix"] for e in ends), default=None), "model_load_seconds_all_attempts": sum(e["load_seconds"] for e in lines(run / "model_loads.jsonl")), "includes_smoke_and_loading": True, "queue_wait_time_excluded": True}
    value["adaptive_worker_time"].update(open_attempt_elapsed_at_report_seconds=open_attempt_seconds, total_elapsed_at_report_seconds=active_seconds + open_attempt_seconds, as_of_unix=value["generated_unix"])
    value["process_memory_peak_bytes"] = max((e.get("process_peak_wset_bytes", 0) or 0 for e in resources), default=None)
    value["system_ram_available_min_bytes"] = min((e["system_ram_available_bytes"] for e in resources if "system_ram_available_bytes" in e), default=None)
    value["full_experiment_planned_items"] = total_planned
    value["report_scope"] = "first_prefix_draft" if limit is not None else "full_cohort"
    value["reported_ids_in_manifest_order"] = [row["row_id"] for row in rows]
    output = Path(output_directory) if output_directory is not None else ROOT / "reports"
    output.mkdir(parents=True, exist_ok=True)
    save(output / "comparison.json", value)
    save(output / "HUMAN_REVIEW.json", {"final_items": manual, "intermediate_traces_pending": len(initial_counts)})
    text = ["# fixed-4 / adaptive-A 비교", "", f"상태: {value['status']}. 현재 로컬 Qwen 7B의 고정 {len(rows)}문항이며 논문 전체 평가를 재현한 것이 아니다.", "", f"표본: {manifest['selection_policy']}", "", "| 조건 | 시도/전체 | 공식 정답 | 공식 정확도 | 별도 정답 | 실행 실패 | 형식 실패 | 미완료 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for method in results:
        metric = methods[method]["all"]
        accuracy = "미완료" if metric["official_accuracy"] is None else f"{metric['official_accuracy']:.2%}"
        text.append(f"| {method} | {metric['attempted']}/{len(rows)} | {metric['official_correct']}/{len(rows)} | {accuracy} | {metric['strict_correct']}/{len(rows)} | {metric['execution_failures']} | {metric['format_failures']} | {metric['incomplete']} |")
    text += ["", "| 하위 유형 | fixed-4 공식/별도 | adaptive-A 공식/별도 |", "|---|---|---|"]
    for name in group_names:
        fm, am = methods["fixed-4"][name], methods["adaptive-A"][name]
        text.append(f"| {name} | {fm['official_correct']}/{fm['planned']} · {fm['strict_correct']}/{fm['planned']} | {am['official_correct']}/{am['planned']} · {am['strict_correct']}/{am['planned']} |")
    text += ["", f"양쪽 정상 완료 paired: {len(paired)}개. fixed-4만 정답 {len(only_fixed)}개 {only_fixed}; adaptive-A만 정답 {len(only_adaptive)}개 {only_adaptive}.", "", "가설 통계(초기화가 실제 수행된 사례 기준; 사실/기억 우회는 제외):", "", "```json", json.dumps(adaptive_stats, ensure_ascii=False, indent=2), "```", "", "계산량(재시도 포함; smoke/연결 확인은 JSON에서 별도 집계):", "", "```json", json.dumps(value["compute_cohort_all_attempts"], ensure_ascii=False, indent=2), "```", "", f"fixed GPU 겹침 관측: {value['fixed_overlap_detected']}. 총 GPU 메모리 표본은 다른 앱을 포함할 수 있다. 호출 시간은 논리적 가설 평가 횟수와 구분한다.", "", f"최종 답변 검토 항목 {len(manual)}개. TT 중간 의미 검토는 미완료다. 정확도만으로 중간 추론 타당성을 보증하지 않는다.", "", "이 비교는 가변 초기 개수·pruning·재표집/재표현 제거·state-only 평가 추가의 묶음 차이다. pruning 하나의 인과 효과나 통계적 우월성을 주장하지 않는다. 삭제 후보는 복구할 수 없고, 후보 1개이면 상대 가중치는 항상 1이며, 모든 후보가 부적절해도 상대 가중치만으로 알아낼 수 없다."]
    text += ["", f"Adaptive worker 경과 시간: {active_seconds + open_attempt_seconds:.1f}초(보고서 생성 시점 기준, 대기 제외, 로딩·smoke 포함). 호출 시간·토큰 및 모델 로딩 합은 comparison.json에 따로 보존한다."]
    (output / "COMPARISON_KO.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return value


if __name__ == "__main__":
    print(json.dumps({"status": analyze()["status"]}))
