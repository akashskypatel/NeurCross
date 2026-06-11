import pytest


def test_quality_gate_grades_a_for_clean_low_score():
    from quad_mesh.quality_gates import evaluate_quality_gate

    result = evaluate_quality_gate(
        {
            "score": 0.25,
            "field_validity": {
                "nan_count": 0,
                "flipped_frame_ratio": 0.0,
            },
        },
        gate_name="default",
    )

    assert result["quality_grade"] == "A"
    assert result["accepted"] is True
    assert result["recommended_destination"] == "accepted"


def test_quality_gate_grades_b_for_moderate_score():
    from quad_mesh.quality_gates import evaluate_quality_gate

    result = evaluate_quality_gate(
        {
            "score": 2.0,
            "field_validity": {
                "nan_count": 0,
                "flipped_frame_ratio": 0.01,
            },
        },
        gate_name="default",
    )

    assert result["quality_grade"] == "B"
    assert result["accepted"] is True


def test_quality_gate_grades_c_and_quarantines_for_warning_thresholds():
    from quad_mesh.quality_gates import evaluate_quality_gate

    result = evaluate_quality_gate(
        {
            "score": 9.0,
            "field_validity": {
                "nan_count": 0,
                "flipped_frame_ratio": 0.2,
            },
        },
        gate_name="default",
    )

    assert result["quality_grade"] == "C"
    assert result["accepted"] is True
    assert result["recommended_destination"] == "accepted"
    assert "field_score_above_grade_b_threshold" in result["warning_checks"]
    assert "flipped_frame_ratio_above_grade_b_threshold" in result["warning_checks"]


def test_quality_gate_grades_d_and_fails_for_nonfinite_vectors():
    from quad_mesh.quality_gates import evaluate_quality_gate

    result = evaluate_quality_gate(
        {
            "score": 0.1,
            "field_validity": {
                "nan_count": 4,
                "flipped_frame_ratio": 0.0,
            },
        },
        gate_name="default",
    )

    assert result["quality_grade"] == "D"
    assert result["accepted"] is False
    assert result["recommended_destination"] == "failed"
    assert result["failure_reason"] == "field_contains_nonfinite_vectors"


def test_strict_gate_rejects_grade_c_output():
    from quad_mesh.quality_gates import evaluate_quality_gate

    result = evaluate_quality_gate(
        {
            "score": 3.0,
            "field_validity": {
                "nan_count": 0,
                "flipped_frame_ratio": 0.0,
            },
        },
        gate_name="strict",
    )

    assert result["quality_grade"] == "C"
    assert result["accepted"] is False
    assert result["recommended_destination"] == "quarantine"


def test_missing_optional_metrics_do_not_break_field_only_acceptance():
    from quad_mesh.quality_gates import evaluate_quality_gate

    result = evaluate_quality_gate({}, gate_name="loose")

    assert result["quality_grade"] in {"A", "B"}
    assert result["accepted"] is True
    assert result["failure_reason"] is None
