"""Dataset loader for Trainaya PyTorch Engine.

Supports:
1. Active Learning / Failure Mining JSONL (`state`, `questions`, `gold_label`, `subset`)
   + Automatic Anchor Replay from `~/.cache/trainaya/benchmarks/public` to prevent catastrophic forgetting!
2. Public Benchmark JSONL (`input.state`, `input.questions`, `reference.label`)
3. Trainaya Decision JSONL (`state`, `question`, `options`, `target_index`)
4. Standard Classification / NLI JSONL or CSV (`text`, `label`)
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple


class NormalizedDecisionRecord:
    """Canonical decision item ready for Laya, Kev, or LLM-Logits encoding."""

    def __init__(
        self,
        record_id: str,
        state: Any,
        qtype: str,
        instructions: str,
        criteria: Any,
        option_keys: List[str],
        gold_index: int,
        sample_weight: float = 1.0,
        subset: str = "custom",
    ):
        self.id = record_id
        self.state = state
        self.qtype = qtype  # "choice", "score", or "noul"
        self.instructions = instructions
        self.criteria = criteria
        self.option_keys = option_keys
        self.gold_index = gold_index
        self.sample_weight = sample_weight
        self.subset = subset

    def to_laya_q(self) -> Dict[str, Any]:
        return {
            "t": self.qtype,
            "ins": self.instructions,
            "crit": self.criteria,
        }


def _resolve_gold_index(qtype: str, criteria: Any, gold_label: Any) -> Tuple[List[str], int]:
    """Map gold_label to option index matching Ollaya's exact option ordering."""
    gold_str = str(gold_label).strip()
    if isinstance(gold_label, bool):
        gold_str = "true" if gold_label else "false"

    if qtype == "noul":
        option_keys = ["false", "true"]
        idx = 1 if gold_str.lower() in ("true", "yes", "1") else 0
        return option_keys, idx

    if qtype == "score":
        if isinstance(criteria, list):
            option_keys = [str(i) for i in range(len(criteria))]
        elif isinstance(criteria, dict):
            option_keys = list(criteria.keys())
        else:
            option_keys = ["0", "1", "2", "3", "4"]
        try:
            idx = int(float(gold_str))
            idx = max(0, min(idx, len(option_keys) - 1))
        except ValueError:
            idx = 0
            for i, k in enumerate(option_keys):
                if k.lower() == gold_str.lower():
                    idx = i
                    break
        return option_keys, idx

    # Default: "choice"
    if isinstance(criteria, dict):
        option_keys = list(criteria.keys())
    elif isinstance(criteria, list):
        option_keys = [str(x) for x in criteria]
    else:
        option_keys = ["false", "true"]

    for i, k in enumerate(option_keys):
        if k.strip().lower() == gold_str.lower():
            return option_keys, i
    return option_keys, 0


def _parse_questions_dict(
    rec_id: str,
    state: Any,
    questions: Dict[str, Any],
    gold_label: Any,
    sample_weight: float = 1.0,
    subset: str = "custom",
) -> List[NormalizedDecisionRecord]:
    out: List[NormalizedDecisionRecord] = []
    for qid, qdef in questions.items():
        if not isinstance(qdef, dict):
            continue
        qtype = str(qdef.get("type", "choice")).lower()
        if qtype not in ("choice", "score", "noul"):
            qtype = "choice"
        instructions = str(qdef.get("instructions") or qdef.get("statement") or qid)
        criteria = qdef.get("criteria")
        if criteria is None and qtype == "noul":
            criteria = {"false": "No, false.", "true": "Yes, true."}
        option_keys, gold_idx = _resolve_gold_index(qtype, criteria, gold_label)
        out.append(
            NormalizedDecisionRecord(
                record_id=f"{rec_id}:{qid}",
                state=state,
                qtype=qtype,
                instructions=instructions,
                criteria=criteria,
                option_keys=option_keys,
                gold_index=gold_idx,
                sample_weight=sample_weight,
                subset=subset,
            )
        )
    return out


def load_decision_records(
    data_path: str,
    include_anchor_replay: bool = True,
) -> List[NormalizedDecisionRecord]:
    """Load any Trainaya dataset file into canonical NormalizedDecisionRecord items."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    ext = os.path.splitext(data_path)[1].lower()
    if ext in (".csv", ".tsv"):
        return _load_csv_records(data_path)

    raw_objs: List[Dict[str, Any]] = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_objs.append(json.loads(line))

    if not raw_objs:
        raise ValueError(f"Dataset file is empty: {data_path}")

    first = raw_objs[0]

    # Case 1: Failure export JSONL (`questions` + `gold_label`)
    if "questions" in first and "gold_label" in first:
        records: List[NormalizedDecisionRecord] = []
        seen_ids = set()
        subsets_seen = set()
        for idx, obj in enumerate(raw_objs):
            rec_id = str(obj.get("id", f"row-{idx}"))
            seen_ids.add(rec_id)
            subset = str(obj.get("subset", "custom"))
            subsets_seen.add(subset)
            state = obj.get("state", obj.get("text", ""))
            # Give misclassified failures higher weight so the optimizer focuses on flipping them
            ftype = str(obj.get("failure_type", ""))
            is_failure = ("misclassified" in ftype) or (obj.get("is_anchor") is False)
            w = 1.6 if is_failure else 1.0
            records.extend(
                _parse_questions_dict(
                    rec_id,
                    state,
                    obj["questions"],
                    obj["gold_label"],
                    sample_weight=w,
                    subset=subset,
                )
            )

        # Anchor replay: mix in passed records from the same subsets in ~/.cache/trainaya/benchmarks/public
        if include_anchor_replay and subsets_seen:
            cache_dir = os.path.expanduser("~/.cache/trainaya/benchmarks/public")
            if os.path.isdir(cache_dir):
                for subset in sorted(subsets_seen):
                    sub_file = os.path.join(cache_dir, subset, "all.jsonl")
                    if not os.path.exists(sub_file):
                        continue
                    with open(sub_file, "r", encoding="utf-8") as sf:
                        for line_idx, sline in enumerate(sf):
                            if line_idx >= 50:
                                break
                            sline = sline.strip()
                            if not sline:
                                continue
                            sobj = json.loads(sline)
                            sid = str(sobj.get("id", f"{subset}-{line_idx}"))
                            if sid in seen_ids:
                                continue
                            inp = sobj.get("input", {})
                            ref = sobj.get("reference", {})
                            if "questions" in inp and "label" in ref:
                                records.extend(
                                    _parse_questions_dict(
                                        sid,
                                        inp.get("state", ""),
                                        inp["questions"],
                                        ref["label"],
                                        sample_weight=1.0,
                                        subset=subset,
                                    )
                                )
        return records

    # Case 2: Public benchmark JSONL (`input.questions` + `reference.label`)
    if "input" in first and "reference" in first:
        records = []
        for idx, obj in enumerate(raw_objs):
            rec_id = str(obj.get("id", f"bench-{idx}"))
            inp = obj["input"]
            ref = obj["reference"]
            records.extend(
                _parse_questions_dict(
                    rec_id,
                    inp.get("state", ""),
                    inp.get("questions", {}),
                    ref.get("label"),
                )
            )
        return records

    # Case 3: DecisionSample JSONL (`options` + `target_index`)
    if "options" in first and "target_index" in first:
        records = []
        for idx, obj in enumerate(raw_objs):
            rec_id = str(obj.get("id", f"sample-{idx}"))
            state = obj.get("state", obj.get("text", ""))
            question = str(obj.get("question", "Which option best applies?"))
            options = [str(o) for o in obj.get("options", ["false", "true"])]
            target_idx = int(obj.get("target_index", 0))
            is_noul = len(options) == 2 and {o.lower() for o in options} == {"false", "true"}
            qtype = "noul" if is_noul else "choice"
            criteria = (
                {"false": "No, false.", "true": "Yes, true."}
                if is_noul
                else {opt: "" for opt in options}
            )
            records.append(
                NormalizedDecisionRecord(
                    record_id=rec_id,
                    state=state,
                    qtype=qtype,
                    instructions=question,
                    criteria=criteria,
                    option_keys=options,
                    gold_index=target_idx,
                )
            )
        return records

    # Case 4: Generic classification / NLI JSONL (`text`/`premise` + `label`)
    return _synthesize_classification_records(raw_objs)


def _synthesize_classification_records(raw_objs: List[Dict[str, Any]]) -> List[NormalizedDecisionRecord]:
    parsed: List[Tuple[Any, str]] = []
    labels_set = set()
    for obj in raw_objs:
        if "premise" in obj and "hypothesis" in obj:
            state: Any = {"premise": obj["premise"], "hypothesis": obj["hypothesis"]}
        else:
            state = (
                obj.get("text")
                or obj.get("sentence")
                or obj.get("state")
                or obj.get("input")
                or obj.get("prompt")
                or ""
            )
        label = str(obj.get("label") or obj.get("target") or obj.get("category") or "").strip()
        if state and label:
            parsed.append((state, label))
            labels_set.add(label)

    option_keys = sorted(labels_set)
    if not option_keys:
        option_keys = ["false", "true"]
    label_to_idx = {k: i for i, k in enumerate(option_keys)}
    criteria = {k: "" for k in option_keys}

    records: List[NormalizedDecisionRecord] = []
    for idx, (state, label) in enumerate(parsed):
        records.append(
            NormalizedDecisionRecord(
                record_id=f"cls-{idx}",
                state=state,
                qtype="choice",
                instructions="Which category best applies to this input?",
                criteria=criteria,
                option_keys=option_keys,
                gold_index=label_to_idx.get(label, 0),
            )
        )
    return records


def _load_csv_records(path: str) -> List[NormalizedDecisionRecord]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return _synthesize_classification_records(rows)
