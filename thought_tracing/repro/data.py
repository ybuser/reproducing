import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "official/data/paraphrased_tomi"
SEED = 20261020


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_rows():
    metadata = (DATA / "test.trace").read_text(encoding="utf-8").splitlines()
    rows, story, raw_story = [], [], []
    for line_no, line in enumerate((DATA / "test.txt").read_text(encoding="utf-8").splitlines(), 1):
        if "\t" not in line:
            text = line.split(" ", 1)[1]
            raw_story.append(line)
            story.append(text if text.endswith(".") else text + ".")
            continue
        q, answer, i = line.strip().split("\t")
        row_id = len(rows)
        trace = metadata[row_id].split(",")
        qtype = trace[-2]
        order = 1 if qtype.startswith("first_order") else 2 if qtype.startswith("second_order") else None
        belief = ("true" if "_no_tom" in qtype else "false") if order else None
        text = " ".join(story).replace("_", " ")
        question = q.split(" ", 1)[1].replace("_", " ").replace("-", " - ")
        rows.append({"row_id": row_id, "question_line_1based": line_no, "question_index": int(i), "story": text, "question": question, "answer": answer.replace("_", " "), "qtype": qtype, "order": order, "belief": belief, "stratum": f"order{order}_{belief}" if order else qtype, "story_sha256": sha(text), "raw_story_sha256": sha("\n".join(raw_story)), "trace_events": trace[:-2], "template_sha256": sha(",".join(trace[:-2])), "scenario_type": trace[-1]})
        story, raw_story = [], []
    assert len(rows) == len(metadata)
    return rows


def model_prompt(row):
    return f"{row['story'].strip('.')}.\n\nQuestion: {row['question']}\nAnswer:"


def prepare_manifests():
    destination = ROOT / "manifests"
    destination.mkdir(exist_ok=True)
    if (destination / "pilot20.json").exists() or (destination / "smoke3.json").exists():
        raise RuntimeError("Refuse to replace a fixed manifest")
    rows = load_rows()
    rng = random.Random(SEED)
    candidates = rows.copy()
    rng.shuffle(candidates)
    used_story, used_template = set(), set()

    def choose(stratum, n):
        selected = []
        for row in candidates:
            if row["stratum"] != stratum or row["story_sha256"] in used_story or row["template_sha256"] in used_template:
                continue
            selected.append(row)
            used_story.add(row["story_sha256"])
            used_template.add(row["template_sha256"])
            if len(selected) == n:
                return selected
        raise RuntimeError(f"Cannot fill {stratum} with distinct stories/event templates")

    smoke = choose("order1_true", 1) + choose("order1_false", 1) + choose("order2_false", 1)
    pilot = sum((choose(s, 5) for s in ["order1_true", "order1_false", "order2_true", "order2_false"]), [])
    gold_vocabulary = sorted({r["answer"] for r in rows}, key=lambda s: (-len(s), s))
    import re
    for selection in [smoke, pilot]:
        for row in selection:
            row["candidate_answer_locations"] = [c for c in gold_vocabulary if re.search(r"(?<!\w)" + re.escape(c) + r"(?!\w)", row["story"], re.I)]
    common = {"seed": SEED, "selection_algorithm": "stdlib random.Random(seed).shuffle; smoke strata first, then pilot strata; first eligible rows; unique normalized story and exact event-sequence template across all 23 rows", "dataset": "paraphrased_tomi/test", "source_files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [DATA / "test.txt", DATA / "test.trace"]}, "question_type_definition": "official revised_tomi/test_tomi.py:257-260; *_tom=false belief, *_no_tom=true belief; 0/1 preserved as raw subtypes without inferring a target or equating them to trace event role IDs; final trace field describes scenario and is not substituted for qtype", "limitations": "Only belief questions, excluding reality/memory. Lexically different stories and distinct exact event templates can share broader generator families; samples are not assumed independent."}
    for name, selection in [("smoke3", smoke), ("pilot20", pilot)]:
        payload = {**common, "split": name, "count": len(selection), "rows": selection}
        text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        (destination / f"{name}.json").write_text(text, encoding="utf-8")
        file_digest = hashlib.sha256((destination / f"{name}.json").read_bytes()).hexdigest()
        (destination / f"{name}.sha256").write_text(file_digest + "\n", encoding="ascii")
    report = {"rows": len(rows), "qtypes": dict(Counter(r["qtype"] for r in rows)), "answer_vocabulary": gold_vocabulary, "unique_stories": len({r["story_sha256"] for r in rows}), "unique_event_templates": len({r["template_sha256"] for r in rows}), "smoke_ids": [r["row_id"] for r in smoke], "pilot_ids": [r["row_id"] for r in pilot]}
    (ROOT / "records/data_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    prepare_manifests()
