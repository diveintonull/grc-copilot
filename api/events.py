"""Stable Server-Sent Event contracts and serialization."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal, TypedDict


EventType = Literal[
    "status",
    "text",
    "reference",
    "recommendation",
    "trace",
    "done",
    "error",
]
EVENT_TYPES = {
    "status",
    "text",
    "reference",
    "recommendation",
    "trace",
    "done",
    "error",
}
TERMINAL_EVENT_TYPES = {"done", "error"}


class SSEEvent(TypedDict):
    """One machine-readable event sent through the SSE stream."""

    type: EventType
    request_id: str
    data: dict[str, Any]


def make_event(
    event_type: EventType,
    request_id: str,
    data: Mapping[str, Any],
) -> SSEEvent:
    """Build one event after validating its stable identity fields."""
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unknown SSE event type: {event_type}")
    if not request_id.strip():
        raise ValueError("request_id must not be blank")
    return {
        "type": event_type,
        "request_id": request_id,
        "data": dict(data),
    }


def encode_sse(event: Mapping[str, Any]) -> str:
    """Serialize a stable event as one SSE frame."""
    event_type = event.get("type")
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unknown SSE event type: {event_type}")

    request_id = event.get("request_id")
    data = event.get("data")
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("request_id must not be blank")
    if not isinstance(data, Mapping):
        raise ValueError("SSE event data must be an object")

    payload = {
        "type": event_type,
        "request_id": request_id,
        "data": dict(data),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"event: {event_type}\ndata: {encoded}\n\n"
