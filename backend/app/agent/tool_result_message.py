"""Agent Runtime 共用的模型可见 Tool Result 序列化。"""

import json

from app.agent.model_response import LLMToolResult


def build_model_facing_tool_result_content(tool_result: LLMToolResult) -> str:
    """为模型序列化 Tool Result，并强化知识证据引用协议。

    该函数只改变发送给模型的 Tool Message 文本；内部 Tool Contract、
    Observer、SSE 与数据库持久化结构均不因此增加字段。
    """

    if tool_result.tool_name != "search_knowledge":
        return tool_result.content_json

    try:
        payload = json.loads(tool_result.content_json)
    except json.JSONDecodeError:
        return tool_result.content_json

    if not isinstance(payload, dict) or payload.get("ok") is not True:
        return tool_result.content_json

    result = payload.get("result")
    if not isinstance(result, dict):
        return tool_result.content_json

    items = result.get("items")
    if not isinstance(items, list):
        return tool_result.content_json

    source_refs: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        source_ref = item.get("source_ref")
        if (
            not isinstance(source_ref, str)
            or not source_ref.strip()
            or source_ref in seen
        ):
            continue
        seen.add(source_ref)
        source_refs.append(source_ref)

    enriched_payload = dict(payload)
    if source_refs:
        enriched_payload["_agent_citation_instruction"] = (
            "If the final answer uses any fact from this search result, "
            "cite at least one supporting source_ref exactly as "
            "[source:<source_ref>]. Use only source_ref values returned "
            "in this tool result. Do not replace the format with [1], "
            "a markdown link, or a bare doc:... reference."
        )
        enriched_payload["_available_source_refs"] = source_refs
    else:
        enriched_payload["_agent_citation_instruction"] = (
            "This search returned no evidence. Do not invent a source_ref "
            "or claim that the knowledge base supports an answer."
        )

    return json.dumps(
        enriched_payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )
