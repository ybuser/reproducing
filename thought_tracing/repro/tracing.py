"""B: observe official methods without changing their algorithm or prompts."""
from contextlib import contextmanager

import numpy as np
import hypothesis as hypothesis_module
import tracer as official
from utils import overall_jaccard_similarity

from .local_model import append_json


class RecordingTracer(official.Tracer):
    def __init__(self, args, directory):
        super().__init__(args)
        self.directory = directory
        self.events = directory / "trace_events.jsonl"
        self.output_file = str(directory / "official_trace_dump.jsonl")
        self.step = -1

    def emit(self, event, **values):
        append_json(self.events, {"event": event, "step": self.step, "case_attempt_id": self.tracer_model.case_attempt_id, **values})

    @contextmanager
    def stage(self, name):
        previous = self.tracer_model.stage
        self.tracer_model.stage = name
        try:
            yield
        finally:
            self.tracer_model.stage = previous

    def identify_target(self, input_text):
        with self.stage("tt.target"):
            target = super().identify_target(input_text)
        self.tracer_model.target_agent = target
        self.emit("target", target_agent=target)
        return target

    def label_action(self, agent, input_text):
        with self.stage("tt.action_label"):
            result = super().label_action(agent, input_text)
        self.emit("action_label", target_agent=agent, input=input_text, raw_annotation=result)
        return result

    def preprocess_input(self, text, target_agent=None):
        result = super().preprocess_input(text, target_agent)
        self.emit("preprocessing", parsed=result)
        return result

    def track_perception(self, trajectory, target_agent=None):
        with self.stage("tt.perception"):
            result = super().track_perception(trajectory, target_agent)
        self.emit("perceptions", trajectory=trajectory, inferred=result)
        return result

    def get_agent_state(self, target_agent, context):
        with self.stage("tt.initial_agent_state"):
            return super().get_agent_state(target_agent, context)

    def initialize(self, state_action, perceptions):
        self.step = 0
        with self.stage("tt.initialize"):
            result = super().initialize(state_action, perceptions)
        self.emit("initialize", state_action=state_action, perceptions=perceptions, hypotheses=result.dump())
        if len(result.texts) != self.args.n_hypotheses:
            raise RuntimeError(f"Initialization format failure: expected 4 hypotheses, got {len(result.texts)} after official retries")
        return result

    def propagate(self, existing_hypotheses, state_action, perceptions):
        self.step += 1
        with self.stage("tt.propagate"):
            result = super().propagate(existing_hypotheses, state_action, perceptions)
        self.emit("propagate", state_action=state_action, perceptions=perceptions, previous_texts=existing_hypotheses.texts, previous_weights=existing_hypotheses.weights, hypotheses=result.dump())
        return result

    def weigh(self, hypotheses, action, mode):
        with self.stage("tt.likelihood"):
            result = super().weigh(hypotheses, action, mode)
        answers = [text.split("Answer:")[-1].strip() for text in result["raw_predictions"]]
        def option(answer):
            for o in "abcdef":
                if answer.startswith(f"({o})") or answer.startswith(f"{o})") or answer.startswith(f"{o}.") or answer.startswith(f"{o} ") or f"({o})" in answer or f" {o})" in answer or f" {o}." in answer or o == answer:
                    return o
            return None
        parsed_options = [option(answer) for answer in answers]
        self.emit("weigh", action=action, texts=hypotheses.texts, previous_weights=hypotheses.weights, parsed_options=parsed_options, format_error=any(o is None for o in parsed_options), results=result, ess=float(1 / np.sum(result["weights"] ** 2)), text_diversity=float(1 - overall_jaccard_similarity(hypotheses.texts)), ess_threshold=self.args.n_hypotheses / 2, rejuvenation_diversity_threshold=0.25)
        return result

    def rejuvenate_hypotheses(self, existing_hypotheses):
        before = list(existing_hypotheses.texts)
        with self.stage("tt.rejuvenate"):
            result = super().rejuvenate_hypotheses(existing_hypotheses)
        self.emit("rejuvenate", before=before, after=result.texts, weights=result.weights, diversity_after=float(1 - overall_jaccard_similarity(result.texts)))
        return result

    def weighted_average_hypotheses(self, hypotheses, top_p=0.9):
        with self.stage("tt.summary"):
            result = super().weighted_average_hypotheses(hypotheses, top_p)
        self.emit("weighted_summary", source=hypotheses.dump(), top_p=top_p, result=result)
        return result

    def dump(self, traced_thought, hypotheses_list):
        self.emit("trace_complete", traced_thought=traced_thought, final_sets=[h.dump() for h in hypotheses_list])
        super().dump(traced_thought, hypotheses_list)


@contextmanager
def record_resampling(tracer):
    original_resample = official.resample_hypotheses_with_other_info
    def recorded(hypotheses, ess):
        choices = hypothesis_module.random.choices
        indices = []
        def observe_choices(*args, **kwargs):
            result = choices(*args, **kwargs)
            indices.extend(result)
            return result
        hypothesis_module.random.choices = observe_choices
        try:
            result = original_resample(hypotheses, ess)
        finally:
            hypothesis_module.random.choices = choices
        tracer.emit("resample", ess=ess, indices=indices, weights_before=hypotheses.weights, texts_before=hypotheses.texts, weights_after=result.weights, texts_after=result.texts, with_replacement=True)
        return result
    official.resample_hypotheses_with_other_info = recorded
    try:
        yield
    finally:
        official.resample_hypotheses_with_other_info = original_resample
