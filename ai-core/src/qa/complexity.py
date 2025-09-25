"""
complexity.py - Heuristic query complexity scoring
Computes a 0..1 score based on linguistic and structural features.

Usage:
  from eval_pipeline.qa.complexity import score_question
"""

import re
from typing import Dict, Any

def _has_math_ops(text: str) -> int:
    return 1 if re.search(
        r"[\+\-\*/%]|percent|rate|ratio|average|growth|yield|cap\s*rate|dscr|noi|per\s*sf",
        text,
        re.I
    ) else 0

def _has_temporal_compare(text: str) -> int:
    return 1 if re.search(
        r"\b(vs|versus|compared|difference|change|delta|increase|decrease)\b",
        text,
        re.I
    ) else 0

def _has_multi_entity(text: str) -> int:
    # crude proxy: multiple proper nouns or tenant mentions
    return 1 if len(re.findall(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?\b", text)) >= 2 else 0

def _count_numbers(text: str) -> int:
    return len(re.findall(r"\b\d+(?:[\.,]\d+)?\b", text))

def _len_bucket(n: int) -> float:
    # normalize text length (words) to 0..1
    return min(1.0, max(0.0, (n - 6) / 30.0))

def score_question(question: str, contexts: list) -> Dict[str, Any]:
    """Score question complexity on a 0–1 scale with feature breakdown."""
    q = question.strip()
    ctx = " ".join(contexts or [])
    words = len(re.findall(r"\w+", q))

    features = {
        "math_ops": float(_has_math_ops(q)),
        "temporal_compare": float(_has_temporal_compare(q)),
        "multi_entity": float(_has_multi_entity(q)),
        "num_mentions": min(1.0, _count_numbers(q)/4.0),
        "length_norm": _len_bucket(words),
        "context_span": min(1.0, len(ctx.split())/400.0)
    }

    # weights: tuned to roughly map easy~0.2-0.4, medium~0.4-0.7, hard~0.7-1.0
    weights = {
        "math_ops": 0.25,
        "temporal_compare": 0.20,
        "multi_entity": 0.15,
        "num_mentions": 0.10,
        "length_norm": 0.15,
        "context_span": 0.15
    }

    score = sum(features[k] * weights[k] for k in weights)
    return {"score": round(min(1.0, score), 4), "components": features}
