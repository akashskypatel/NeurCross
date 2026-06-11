from __future__ import annotations


GATE_PROFILES = {
    "none": {
        "grade_a_score": float("inf"),
        "grade_b_score": float("inf"),
        "grade_b_flip": float("inf"),
        "strict_accept": False,
    },
    "default": {
        "grade_a_score": 1.0,
        "grade_b_score": 5.0,
        "grade_b_flip": 0.05,
        "strict_accept": False,
    },
    "strict": {
        "grade_a_score": 0.5,
        "grade_b_score": 2.5,
        "grade_b_flip": 0.02,
        "strict_accept": True,
    },
    "loose": {
        "grade_a_score": 2.0,
        "grade_b_score": 8.0,
        "grade_b_flip": 0.1,
        "strict_accept": False,
    },
}


def evaluate_quality_gate(best_metrics: dict, *, gate_name: str = "default") -> dict[str, object]:
    if gate_name not in GATE_PROFILES:
        raise ValueError(f"unsupported quality gate: {gate_name}")

    field_validity = best_metrics.get("field_validity") or {}
    score = float(best_metrics.get("score", best_metrics.get("field_score", 0.0)))
    nan_count = int(field_validity.get("nan_count", 0))
    flipped_frame_ratio = float(field_validity.get("flipped_frame_ratio", 0.0))
    profile = GATE_PROFILES[gate_name]

    failed_checks: list[str] = []
    warning_checks: list[str] = []
    failure_reason = None

    if nan_count > 0:
        failed_checks.append("field_contains_nonfinite_vectors")
        failure_reason = "field_contains_nonfinite_vectors"
        grade = "D"
    else:
        if score <= profile["grade_a_score"] and flipped_frame_ratio <= min(0.01, profile["grade_b_flip"]):
            grade = "A"
        elif score <= profile["grade_b_score"] and flipped_frame_ratio <= profile["grade_b_flip"]:
            grade = "B"
        else:
            grade = "C"
            if score > profile["grade_b_score"]:
                warning_checks.append("field_score_above_grade_b_threshold")
            if flipped_frame_ratio > profile["grade_b_flip"]:
                warning_checks.append("flipped_frame_ratio_above_grade_b_threshold")

    accepted = nan_count == 0
    if profile["strict_accept"]:
        accepted = accepted and grade in {"A", "B"}

    if failure_reason is not None:
        recommended_destination = "failed"
    elif accepted:
        recommended_destination = "accepted"
    else:
        recommended_destination = "quarantine"

    return {
        "accepted": accepted,
        "quality_grade": grade,
        "quality_gate": gate_name,
        "field_score": score,
        "failure_reason": failure_reason,
        "failed_checks": failed_checks,
        "warning_checks": warning_checks,
        "recommended_destination": recommended_destination,
    }
