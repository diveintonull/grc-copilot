"""Routing and workflow nodes for the GRC agent graph."""

from __future__ import annotations

from agent.state import AgentState


def route_intent(state: AgentState, classify_intent) -> dict:
    """Classify one request and return its intent state update."""
    intent = classify_intent(state["query"], state["control_text"])

    old_trace = state["trace"]
    new_event = {"node": "route_intent", "intent": intent}
    new_trace = old_trace + [new_event]

    return {
        "intent": intent,
        "trace": new_trace,
    }


def select_workflow(state: AgentState) -> str:
    """Return the next workflow node name for the classified intent."""
    intent = state["intent"]

    if intent == "regulation_qa":
        return "execute_regulation_qa"
    elif intent == "clause_comparison":
        return "execute_clause_comparison"
    elif intent == "gap_analysis":
        return "execute_gap_analysis"
    elif intent == "unsupported":
        return "execute_unsupported"
    else:
        raise ValueError(f"unknown intent: {intent}")


def execute_regulation_qa(state: AgentState, tools, llm) -> dict:
    """Search regulation evidence and produce an evidence-aware answer."""
    query = state["query"]
    evidence = tools.search_regulation(query, None)

    tool_call = {
        "tool": "search_regulation",
        "query": query,
        "source_ids": None,
        "result_count": len(evidence),
    }
    new_trace = {
        "node": "execute_regulation_qa",
        "tool": "search_regulation",
        "result_count": len(evidence),
    }

    if evidence:
        answer = llm.answer_regulation(query, evidence)
    else:
        answer = "insufficient regulation evidence"

    return {
        "answer": answer,
        "tool_calls": state.get("tool_calls", []) + [tool_call],
        "evidence": evidence,
        "trace": state["trace"] + [new_trace],
    }


def execute_clause_comparison(state: AgentState, tools, llm) -> dict:
    """Compare two precisely identified clauses when both sides exist."""
    query = state["query"]
    plan = llm.plan_comparison(query)

    comparison = tools.compare_clauses(
        plan["left"],
        plan["right"],
        plan["dimensions"],
    )

    left_evidence = comparison["left"]
    right_evidence = comparison["right"]
    left_found = left_evidence is not None
    right_found = right_evidence is not None

    evidence = [
        item
        for item in [left_evidence, right_evidence]
        if item is not None
    ]
    tool_call = {
        "tool": "compare_clauses",
        "left": plan["left"],
        "right": plan["right"],
        "dimensions": plan["dimensions"],
        "left_found": left_found,
        "right_found": right_found,
    }
    new_trace = {
        "node": "execute_clause_comparison",
        "tool": "compare_clauses",
        "left_found": left_found,
        "right_found": right_found,
    }

    if left_found and right_found:
        answer = llm.answer_comparison(query, comparison)
    else:
        answer = "incomplete comparison evidence"

    return {
        "answer": answer,
        "tool_calls": state.get("tool_calls", []) + [tool_call],
        "evidence": evidence,
        "trace": state["trace"] + [new_trace],
    }


def execute_gap_analysis(state: AgentState) -> dict:
    """Return a deterministic placeholder for the gap-analysis route."""
    return {
        "answer": "fake gap_analysis result",
        "trace": state["trace"] + [{"node": "execute_gap_analysis"}],
    }


def execute_unsupported(state: AgentState, tools) -> dict:
    """Refuse an unsupported request without calling a retrieval tool."""
    return {
        "answer": "unsupported request",
        "trace": state["trace"] + [{"node": "execute_unsupported"}],
    }


def verify(state: AgentState) -> dict:
    """Return the deterministic verification update for a routed request."""
    intent = state["intent"]
    evidence = state.get("evidence", [])

    if intent == "regulation_qa":
        citations_valid = bool(evidence)
    elif intent == "clause_comparison":
        citations_valid = len(evidence) == 2
    elif intent == "gap_analysis":
        citations_valid = True
    else:
        citations_valid = False

    return {
        "citations_valid": citations_valid,
        "trace": state["trace"] + [
            {
                "node": "verify",
                "citations_valid": citations_valid,
            }
        ],
    }


def finish(state: AgentState) -> dict:
    """Return the final status update and terminal trace event."""
    citations_valid = state["citations_valid"]
    final_status = "completed" if citations_valid else "refused"

    return {
        "final_status": final_status,
        "trace": state["trace"] + [
            {
                "node": "finish",
                "final_status": final_status,
            }
        ],
    }
