"""
quality_checks.py - Orchestrates automated quality checks, integrates schema, validation, and complexity scoring.

Usage (module API):
  from eval_pipeline.qa.quality_checks import run_quality_checks
  run_quality_checks(input="data/questions.json",
                     schema="eval_pipeline/qa/question_schema.json",
                     out="reports/qa_report.json")

CLI:
  uv run python -m eval_pipeline.qa.quality_checks \
    --input data/questions.json \
    --schema eval_pipeline/qa/question_schema.json \
    --out reports/qa_report.json
"""

import argparse, json, pathlib, datetime
from typing import Dict, Any, List
from eval_pipeline.qa.validators import (
    _load_json_or_jsonl,
    validate_schema,
    validate_distribution,
    validate_duplicates,
    validate_easy_answer_in_context,
    _parse_dist
)
from eval_pipeline.qa.complexity import score_question

DEFAULT_DIST = "easy:0.4,medium:0.3,hard:0.3"

def attach_complexity(items: List[Dict[str, Any]]) -> None:
    """Attach complexity scores to each item if missing."""
    for obj in items:
        try:
            if "complexity_score" not in obj:
                obj["complexity_score"] = score_question(
                    obj.get("question", ""),
                    obj.get("contexts", [])
                )
        except Exception:
            # Fail-safe: never break validation pipeline due to scoring errors
            pass

def run_quality_checks(
    input: str,
    schema: str,
    out: str = None,
    difficulty: str = DEFAULT_DIST,
    tolerance: float = 0.05
) -> Dict[str, Any]:
    """Run schema + distribution + duplication + easy-answer checks with complexity scoring."""
    items = _load_json_or_jsonl(input)
    attach_complexity(items)

    # Core validations
    ok_schema, schema_errs, schema_warns = validate_schema(items, schema)
    dist = _parse_dist(difficulty)
    dist_ok, dist_msg = validate_distribution(items, dist, tolerance)
    dup_ok, dup_detail = validate_duplicates(items)
    easy_ok, easy_detail = validate_easy_answer_in_context(items)

    report = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "files": {"input": input, "schema": schema},
        "results": {
            "schema": {
                "pass": ok_schema,
                "errors": schema_errs,
                "warnings": schema_warns
            },
            "distribution": {"pass": dist_ok, "detail": dist_msg},
            "duplicates": {"pass": dup_ok, "detail": dup_detail},
            "easy_answer_in_context": {"pass": easy_ok, "detail": easy_detail},
        }
    }

    # Write optional report + enriched dataset
    if out:
        p = pathlib.Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2), encoding="utf-8")
        enriched_path = p.with_suffix(".enriched.json")
        enriched_path.write_text(json.dumps(items, indent=2), encoding="utf-8")

    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--schema", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--difficulty", default=DEFAULT_DIST)
    ap.add_argument("--tolerance", type=float, default=0.05)
    args = ap.parse_args()

    rep = run_quality_checks(args.input, args.schema, args.out, args.difficulty, args.tolerance)
    print(json.dumps(rep, indent=2))

if __name__ == "__main__":
    main()
