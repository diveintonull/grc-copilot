"""Container composition root for the reproducible local demo."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from typing import Any

from qdrant_client import QdrantClient

from api.main import ChatRequest, ReadinessProbe, create_app


DEFAULT_QDRANT_URL = "http://localhost:6333"
SUPPORTED_RUN_MODES = {"demo"}

AUTHENTICATION_EVIDENCE = {
    "parent_id": "GBT-22239@2019#8.1.4.1",
    "source_id": "GB/T 22239",
    "version": "2019",
    "section_number": "8.1.4.1",
    "text": "应采用两种或两种以上组合的鉴别技术对用户进行身份鉴别。",
    "score": 0.94,
}
ISO_AUTHENTICATION_EVIDENCE = {
    "parent_id": "ISO-27001@2022#A.5.17",
    "source_id": "ISO/IEC 27001",
    "version": "2022",
    "section_number": "A.5.17",
    "text": (
        "Authentication information shall be allocated and managed "
        "through a management process."
    ),
    "score": 0.89,
}


def _trace(node: str, tool: str, duration_ms: int) -> dict[str, Any]:
    return {
        "node": node,
        "tool": tool,
        "duration_ms": duration_ms,
        "status": "completed",
    }


async def demo_runner(request: ChatRequest) -> Mapping[str, Any]:
    """Return deterministic evidence-linked fixtures for deployment checks."""
    if "慢速" in request.query:
        await asyncio.sleep(30)

    if request.mode == "clause_comparison":
        return {
            "answer": (
                "GB/T 22239 要求组合使用两种或以上鉴别技术[1]；"
                "ISO/IEC 27001 强调鉴别信息的分配与管理过程[2]。"
                "两者关注点不同，不能仅凭名称判断等价。"
            ),
            "evidence": [
                AUTHENTICATION_EVIDENCE,
                ISO_AUTHENTICATION_EVIDENCE,
            ],
            "recommendations": [
                "正式控制映射仍需法规与控制负责人共同复核。"
            ],
            "trace": [
                _trace("route_intent", "mode_selector", 1),
                _trace("execute_clause_comparison", "get_clause", 12),
                _trace("verify", "citation_validator", 2),
            ],
            "final_status": "completed",
        }

    if request.mode == "gap_analysis":
        if not request.control_text.strip():
            return {
                "answer": "缺少企业当前控制描述，无法进行差距判断。",
                "evidence": [],
                "recommendations": [
                    "请描述当前实际措施，不要预先填写合规结论。"
                ],
                "trace": [
                    _trace("execute_gap_analysis", "input_guard", 1)
                ],
                "final_status": "refused",
            }
        return {
            "answer": (
                f"当前控制事实为“{request.control_text}”。"
                "法规要求采用两种或以上鉴别技术[1]。"
                "若现状确实只有单一密码，则存在待复核差距；"
                "该判断不是审计结论，必须由控制负责人确认。"
            ),
            "evidence": [AUTHENTICATION_EVIDENCE],
            "recommendations": [
                "确认生产环境是否已启用第二种鉴别因素。",
                "保存配置截图、抽样登录记录和例外审批。",
            ],
            "trace": [
                _trace("route_intent", "mode_selector", 1),
                _trace("execute_gap_analysis", "search_regulation", 15),
                _trace("verify", "citation_validator", 2),
            ],
            "final_status": "completed",
        }

    return {
        "answer": (
            "管理员账户应采用两种或两种以上鉴别技术的组合[1]。"
            "实际落地仍需结合系统边界、账户类型和例外流程人工确认。"
        ),
        "evidence": [AUTHENTICATION_EVIDENCE],
        "recommendations": [
            "先核对条款版本，再把要求映射到具体系统控制。"
        ],
        "trace": [
            _trace("route_intent", "mode_selector", 1),
            _trace("execute_regulation_qa", "search_regulation", 10),
            _trace("verify", "citation_validator", 2),
        ],
        "final_status": "completed",
    }


async def qdrant_ready() -> bool:
    """Return whether the configured Qdrant REST service accepts requests."""
    url = os.environ.get("QDRANT_URL", DEFAULT_QDRANT_URL)

    def check() -> bool:
        client = QdrantClient(url=url, timeout=2)
        try:
            client.get_collections()
            return True
        finally:
            client.close()

    try:
        return await asyncio.to_thread(check)
    except Exception:
        return False


def create_deployment_app(
    *,
    readiness_probe: ReadinessProbe | None = None,
    run_mode: str | None = None,
):
    """Build the explicit container app and reject unknown run modes early."""
    selected_mode = run_mode or os.environ.get("APP_RUN_MODE", "demo")
    if selected_mode not in SUPPORTED_RUN_MODES:
        supported = ", ".join(sorted(SUPPORTED_RUN_MODES))
        raise RuntimeError(
            f"unsupported APP_RUN_MODE {selected_mode!r}; expected {supported}"
        )
    return create_app(
        agent_runner=demo_runner,
        readiness_probe=readiness_probe or qdrant_ready,
        text_chunk_size=12,
    )


app = create_deployment_app()
