"""Adaptive-A on the copied, actually running fixed-4 implementation."""
from copy import deepcopy
import json
from pathlib import Path
import sys

from .common import ROOT
from .kernel import FormatError, OPTIONS, SCORE_MAPPING, parse_initial_list, parse_option, prune_once, softmax_scores, summary_indices
from repro import bootstrap  # telemetry/DLL settings before ML imports

sys.path.insert(0, str(ROOT / "official"))
import numpy as np
from hypothesis import HypothesesSetV3
from utils import softmax as fixed_softmax
from repro.tracing import RecordingTracer


def prompt(name, **values):
    return (ROOT / "adaptive/prompts" / name).read_text(encoding="utf-8").format(**values).strip()


class AdaptiveTracer(RecordingTracer):
    def __init__(self, args, directory, method_config):
        super().__init__(args, directory)
        self.method_config = method_config
        self.history_snapshots = []
        self.initial_count = None
        self.summary_cursor = 0

    def emit(self, event, **values):
        super().emit(event, method="adaptive-A", example_id=self.args.example_id, **values)

    def initialize(self, state_action, perceptions):
        self.step = 0
        state, action = state_action["state"], state_action["action"]
        context = ""
        # Same initial information scope and fact-extraction call as fixed-4.
        if state:
            if self.args.input_is_chat:
                context += f"{state}\n"
            else:
                agent_state = self.get_agent_state(self.target_agent, state)
                context += f"<state>\n{agent_state.strip()}\n</state>\n"
            context += f"<note>{perceptions['state']}</note>\n\n"
        if action:
            context += f"<action>\n{action}\n</action>\n"
            if perceptions["action"]:
                context += f"<note>{perceptions['action']}</note>\n"
        query = prompt("initialize_action.txt" if action else "initialize_state.txt", context=context.strip(), assumption=self.assumption, agent=self.target_agent)
        original_query = query
        for attempt in range(2):
            with self.stage("tt.initialize" if attempt == 0 else "tt.initialize.format_repair"):
                raw = self.tracer_model.interact(query, temperature=0, max_tokens=1024)
            try:
                texts = parse_initial_list(raw)
            except FormatError as exc:
                self.emit("adaptive_initial_format", attempt=attempt + 1, raw=raw, error=str(exc), valid=False)
                if attempt == 1:
                    raise
                query = prompt("repair_list.txt", original_prompt=original_query, raw=raw)
            else:
                self.initial_count = len(texts)
                ids = [f"{self.args.example_id}:h{i + 1:02d}" for i in range(len(texts))]
                result = HypothesesSetV3(self.target_agent, [deepcopy(state_action)], [deepcopy(perceptions)], list(texts), np.ones(len(texts)) / len(texts))
                result.ids = ids
                self.emit("adaptive_initial_format", attempt=attempt + 1, raw=raw, parsed=texts, ids=ids, initial_count=len(texts), valid=True, initial_weights=result.weights)
                return result

    def state_only_evaluation(self, current, previous_by_id):
        perspective = self.interleave_context_and_perception(current.contexts, current.perceptions)
        queries = [prompt("state_only_user.txt", agent=self.target_agent, previous=previous_by_id.get(hid, "No previous hypothesis; this is the initial step."), updated=text, perspective=perspective, perception=json.dumps(current.perceptions[-1], ensure_ascii=False), options=OPTIONS) for hid, text in zip(current.ids, current.texts)]
        system = prompt("state_only_system.txt")
        with self.stage("adaptive.state_only_plausibility"):
            outputs = self.tracer_model.batch_interact(queries, temperature=0, max_tokens=512, system_prompts=system)
        options = [parse_option(raw) for raw in outputs]
        return {"prompts": queries, "raw_predictions": outputs, "reasonings": [raw.split("Answer:")[0].strip() for raw in outputs], "raw_scores": [SCORE_MAPPING[o] for o in options], "parsed_options": options}

    def weighted_average_hypotheses(self, hypotheses, top_p=0.9):
        top_p = self.method_config["summary_top_p"]
        chosen = summary_indices(hypotheses.weights, top_p)
        self.emit("adaptive_summary_selection", summary_for_step=self.summary_cursor, summary_top_p=top_p, survivor_ids=list(hypotheses.ids), survivor_count=len(hypotheses.ids), summary_ids=[hypotheses.ids[i] for i in chosen], summary_count=len(chosen))
        self.summary_cursor += 1
        return super().weighted_average_hypotheses(hypotheses, top_p)

    def trace(self, input_text, target_agent=None):
        # No cross-example/story-only tracer cache. The model journal has full identity keys.
        if target_agent is None:
            target_agent = self.identify_target(input_text)
        parsed = self.preprocess_input(input_text, target_agent)
        if parsed is None:
            raise FormatError("Target identification/preprocessing failed")
        self.set_tracer_variables(parsed)
        trajectory, perceptions = parsed["trajectory"], parsed["perceptions"]
        if not trajectory or len(trajectory) != len(perceptions):
            raise FormatError("Empty/misaligned preprocessed trajectory")
        history, dead = [], set()
        self.stage_sets = history
        for index, (state_action, perception) in enumerate(zip(trajectory, perceptions)):
            if index == 0:
                current = self.initialize(state_action, perception)
                previous_by_id = {}
            else:
                previous = history[-1]
                previous_by_id = dict(zip(previous.ids, previous.texts))
                current = super().propagate(previous, state_action, perception)
                current.ids = list(previous.ids)
            if not current.ids or len(current.ids) != len(current.texts) or dead.intersection(current.ids):
                raise RuntimeError("Hypothesis ID/content invariant failed")
            if state_action["action"]:
                evaluation_type = "action_likelihood"
                evaluation = super().weigh(current, state_action["action"], "prompting")
                evaluation["parsed_options"] = [parse_option(raw) for raw in evaluation["raw_predictions"]]
                for option, score in zip(evaluation["parsed_options"], evaluation["raw_scores"]):
                    if SCORE_MAPPING[option] != float(score):
                        raise RuntimeError("Action evaluator mapping differs from fixed-4")
            else:
                evaluation_type = "state_only_plausibility"
                evaluation = self.state_only_evaluation(current, previous_by_id)
            scores = evaluation["raw_scores"]
            weights = softmax_scores(scores, len(current.ids))
            if len(evaluation["raw_predictions"]) != len(current.ids):
                raise ValueError("Evaluation output length mismatch")
            if "weights" in evaluation and not np.allclose(weights, evaluation["weights"], rtol=0, atol=1e-12):
                raise RuntimeError("Softmax differs from fixed-4")
            # Use the exact copied NumPy implementation for boundary decisions.
            weights = [float(value) for value in evaluation["weights"]] if "weights" in evaluation else fixed_softmax(np.asarray(scores, dtype=float)).tolist()
            pruning = prune_once(current.ids, weights, self.method_config["prune_threshold"])
            keep = pruning["keep_indices"]
            survivor = HypothesesSetV3(self.target_agent, deepcopy(current.contexts), deepcopy(current.perceptions), [current.texts[i] for i in keep], np.array(pruning["weights_after"], dtype=float), parent_hypotheses=[current.hypotheses[i] for i in keep])
            survivor.ids = list(pruning["kept_ids"])
            if history and not set(survivor.ids).issubset(history[-1].ids):
                raise RuntimeError("Hypothesis count increased or deleted hypothesis restored")
            snapshot = {"event": "adaptive_step", "step": index, "target_agent": self.target_agent, "evaluator_type": evaluation_type, "initial_count": self.initial_count, "count_before_evaluation": len(current.ids), "count_after_pruning": len(survivor.ids), "ids_before": list(current.ids), "previous_texts_by_id": previous_by_id, "texts_before": list(current.texts), "previous_weights": current.weights.tolist(), "raw_evaluations": list(evaluation["raw_predictions"]), "parsed_options": list(evaluation["parsed_options"]), "raw_scores": [float(s) for s in scores], "weights_before_pruning": weights, "ids_after": list(survivor.ids), "texts_after": list(survivor.texts), "weights_after_pruning": survivor.weights.tolist(), "deleted_ids": pruning["deleted_ids"], "delete_reason": "normalized_weight_strictly_less_than_threshold", "prune_threshold": self.method_config["prune_threshold"], "prune_comparison": "strictly_less_than", "minimum_one_guard": pruning["minimum_one_guard"], "prune_passes": 1, "resampling": False, "rejuvenation": False, "hypothesis_evaluations": len(current.ids), "batch_interface_calls": 1, "logical_interact_slots_including_replays": len(current.ids), "actual_requests_counted_from_calls_journal": True, "external_api_calls": 0, "state_action": deepcopy(state_action), "perception": deepcopy(perception)}
            self.history_snapshots.append(deepcopy(snapshot))
            event = {key: value for key, value in snapshot.items() if key not in ("event", "step")}
            self.emit("adaptive_step", **event)
            dead.update(pruning["deleted_ids"])
            history.append(survivor)
        trace = self.chain_weighted_average_trace(history)
        trace_text = f"{self.trace_header}\n\n{trace['text']}"
        self.dump(trace, history)
        return trace_text

    def rejuvenate_hypotheses(self, *args, **kwargs):
        raise RuntimeError("adaptive-A forbids rejuvenation")
