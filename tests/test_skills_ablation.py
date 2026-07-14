from __future__ import annotations

from pathlib import Path

import pytest

from agent.skills import discover_skills
from evals.run_skills_ablation import (
    AblationRecord,
    EvidenceBlock,
    ModelResponse,
    STRATEGIES,
    aggregate_records,
    build_case_inputs,
    evaluate_input,
    evaluate_target,
)
from evals.schema import EvaluationCase


def make_case(
    *,
    task_type: str = "regulation_qa",
    should_refuse: bool = False,
) -> EvaluationCase:
    citations = () if should_refuse else ("law@2026#1",)
    return EvaluationCase(
        id=f"case-{task_type}",
        question="What does the requirement say?",
        task_type=task_type,
        gold_points=("a grounded point",),
        gold_citations=citations,
        should_refuse=should_refuse,
        source_versions=() if should_refuse else ("law@2026",),
        tags=("refusal",) if should_refuse else ("single_regulation",),
    )


def write_skill(root: Path, name: str, description: str, body: str) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        "---\n\n"
        f"{body}\n",
        encoding="utf-8",
    )


@pytest.fixture
def catalog(tmp_path: Path):
    write_skill(tmp_path, "regulation-qa", "Answer regulations", "QA SOP")
    write_skill(tmp_path, "clause-comparison", "Compare clauses", "COMPARE SOP")
    write_skill(tmp_path, "gap-analysis", "Map control gaps", "GAP SOP")
    return discover_skills(tmp_path)


def test_three_inputs_share_case_and_evidence_and_only_skill_message_differs(
    catalog,
) -> None:
    case = make_case()
    evidence = (
        EvidenceBlock(citation="law@2026#1", text="Requirement text"),
    )

    inputs = build_case_inputs(case, evidence=evidence, catalog=catalog)

    assert tuple(item.strategy for item in inputs) == STRATEGIES
    assert len({item.shared_input_sha256 for item in inputs}) == 1
    assert all(item.case == case for item in inputs)
    assert all(item.evidence == evidence for item in inputs)
    non_skill_messages = [
        tuple(
            message
            for message in item.messages
            if message.get("name") != "skill_instructions"
        )
        for item in inputs
    ]
    assert non_skill_messages[0] == non_skill_messages[1] == non_skill_messages[2]


def test_skill_injection_matches_each_strategy(catalog) -> None:
    inputs = build_case_inputs(make_case(), evidence=(), catalog=catalog)
    by_strategy = {item.strategy: item for item in inputs}

    assert by_strategy["none"].active_skills == ()
    assert by_strategy["none"].skill_text == ""
    assert set(by_strategy["full"].active_skills) == {
        "regulation-qa",
        "clause-comparison",
        "gap-analysis",
    }
    assert "QA SOP" in by_strategy["full"].skill_text
    assert "COMPARE SOP" in by_strategy["full"].skill_text
    assert "GAP SOP" in by_strategy["full"].skill_text
    assert by_strategy["progressive"].active_skills == ("regulation-qa",)
    assert "QA SOP" in by_strategy["progressive"].skill_text
    assert "COMPARE SOP" not in by_strategy["progressive"].skill_text


def test_progressive_unsupported_request_loads_no_skill(catalog) -> None:
    inputs = build_case_inputs(
        make_case(task_type="unsupported", should_refuse=True),
        evidence=(),
        catalog=catalog,
    )

    progressive = next(item for item in inputs if item.strategy == "progressive")

    assert progressive.active_skills == ()
    assert progressive.skill_text == ""


def test_evaluate_input_uses_reported_usage_and_checks_contract(catalog) -> None:
    experiment_input = build_case_inputs(
        make_case(),
        evidence=(EvidenceBlock("law@2026#1", "Requirement text"),),
        catalog=catalog,
    )[2]
    captured = []

    def complete(messages, *, model, temperature, max_tokens):
        captured.append((messages, model, temperature, max_tokens))
        return ModelResponse(
            answer="The requirement applies [1].",
            input_tokens=120,
            total_tokens=150,
        )

    record = evaluate_input(
        experiment_input,
        complete=complete,
        model="model-a",
        temperature=0.0,
        max_tokens=500,
    )

    assert captured[0][0] == experiment_input.messages
    assert captured[0][1:] == ("model-a", 0.0, 500)
    assert record.input_tokens == 120
    assert record.total_tokens == 150
    assert record.citation_format_pass is True
    assert record.task_success is True
    assert record.error is None


def test_length_limited_response_is_never_counted_as_success(catalog) -> None:
    experiment_input = build_case_inputs(
        make_case(),
        evidence=(EvidenceBlock("law@2026#1", "Requirement text"),),
        catalog=catalog,
    )[2]

    def complete(messages, *, model, temperature, max_tokens):
        return ModelResponse(
            answer="The requirement applies [1].",
            input_tokens=120,
            total_tokens=150,
            finish_reason="length",
        )

    record = evaluate_input(
        experiment_input,
        complete=complete,
        model="model-a",
        temperature=0.0,
        max_tokens=500,
    )

    assert record.finish_reason == "length"
    assert record.task_success is False
    assert record.error == "ModelOutputTruncated: finish_reason=length"


def make_record(
    strategy: str,
    case_id: str,
    task_type: str,
    *,
    input_tokens: int,
    total_tokens: int,
    latency_ms: float,
    task_success: bool,
    citation_format_pass: bool,
    active_skills: tuple[str, ...],
) -> AblationRecord:
    return AblationRecord(
        strategy=strategy,
        case_id=case_id,
        task_type=task_type,
        active_skills=active_skills,
        input_tokens=input_tokens,
        total_tokens=total_tokens,
        latency_ms=latency_ms,
        answer="answer",
        cited_evidence=(),
        citation_format_pass=citation_format_pass,
        task_success=task_success,
        error=None,
    )


def test_aggregation_computes_quality_latency_and_trigger_rates() -> None:
    records = [
        make_record(
            "progressive",
            "qa",
            "regulation_qa",
            input_tokens=100,
            total_tokens=130,
            latency_ms=10,
            task_success=True,
            citation_format_pass=True,
            active_skills=("regulation-qa",),
        ),
        make_record(
            "progressive",
            "gap",
            "gap_analysis",
            input_tokens=140,
            total_tokens=180,
            latency_ms=30,
            task_success=False,
            citation_format_pass=False,
            active_skills=(),
        ),
        make_record(
            "progressive",
            "refuse",
            "unsupported",
            input_tokens=60,
            total_tokens=80,
            latency_ms=20,
            task_success=True,
            citation_format_pass=True,
            active_skills=("regulation-qa",),
        ),
    ]

    summary = aggregate_records(records)["progressive"]

    assert summary.average_input_tokens == pytest.approx(100)
    assert summary.average_total_tokens == pytest.approx(130)
    assert summary.p95_latency_ms == 30
    assert summary.task_success_rate == pytest.approx(2 / 3)
    assert summary.citation_format_pass_rate == pytest.approx(2 / 3)
    assert summary.false_trigger_rate == 1
    assert summary.miss_trigger_rate == pytest.approx(1 / 2)


def test_target_requires_token_reduction_and_success_retention() -> None:
    records = []
    for index in range(100):
        records.append(
            make_record(
                "full",
                f"full-{index}",
                "regulation_qa",
                input_tokens=1000,
                total_tokens=1100,
                latency_ms=1,
                task_success=index < 90,
                citation_format_pass=True,
                active_skills=(
                    "regulation-qa",
                    "clause-comparison",
                    "gap-analysis",
                ),
            )
        )
        records.append(
            make_record(
                "progressive",
                f"progressive-{index}",
                "regulation_qa",
                input_tokens=700,
                total_tokens=800,
                latency_ms=1,
                task_success=index < 88,
                citation_format_pass=True,
                active_skills=("regulation-qa",),
            )
        )

    summaries = aggregate_records(records)
    target = evaluate_target(summaries)

    assert target.input_token_reduction == pytest.approx(0.30)
    assert target.success_rate_drop_pp == pytest.approx(2.0)
    assert target.token_target_met is True
    assert target.success_target_met is True
    assert target.overall_met is True
