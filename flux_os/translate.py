"""Format translation.

flux-os works internally on the OpenAI Chat Completions shape. This module
converts to and from:

* Anthropic Messages (upstream calls to Claude, and the inbound ``/v1/messages``
  endpoint used by Claude Code and the Anthropic SDKs)
* OpenAI Responses (the inbound ``/v1/responses`` endpoint used by the OpenAI
  Agents SDK default mode, n8n, and newer OpenAI SDK code)
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from .classifier import content_text

ANTHROPIC_DEFAULT_MAX_TOKENS = 8192


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _parse_args(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    try:
        parsed = json.loads(arguments or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _data_uri_to_anthropic(url: str) -> dict[str, Any]:
    if url.startswith("data:") and ";base64," in url:
        header, data = url[5:].split(";base64,", 1)
        return {"type": "image", "source": {"type": "base64", "media_type": header, "data": data}}
    return {"type": "image", "source": {"type": "url", "url": url}}


# ═══════════════════════════ OpenAI chat → Anthropic ═══════════════════════════


def _oa_content_to_anthropic(content: Any) -> list[dict[str, Any]]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind in ("text", "input_text", "output_text") and part.get("text"):
            blocks.append({"type": "text", "text": part["text"]})
        elif kind == "image_url":
            img = part.get("image_url")
            url = img.get("url") if isinstance(img, dict) else img
            if url:
                blocks.append(_data_uri_to_anthropic(url))
    return blocks


def chat_to_anthropic(req: dict[str, Any], upstream_model: str, max_output: int) -> dict[str, Any]:
    """Convert an OpenAI chat request to an Anthropic Messages request."""
    system: list[str] = []
    messages: list[dict[str, Any]] = []

    def push(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(blocks)
        else:
            messages.append({"role": role, "content": list(blocks)})

    for m in req.get("messages") or []:
        role = m.get("role")
        if role in ("system", "developer"):
            text = content_text(m.get("content"))
            if text:
                system.append(text)
        elif role == "user":
            push("user", _oa_content_to_anthropic(m.get("content")))
        elif role == "assistant":
            blocks = _oa_content_to_anthropic(m.get("content"))
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id") or _uid("toolu"),
                        "name": fn.get("name", ""),
                        "input": _parse_args(fn.get("arguments")),
                    }
                )
            push("assistant", blocks)
        elif role == "tool":
            result = content_text(m.get("content"))
            push(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": m.get("tool_call_id", ""),
                        "content": result or "(empty)",
                    }
                ],
            )

    rf = req.get("response_format")
    if isinstance(rf, dict) and rf.get("type") in ("json_object", "json_schema"):
        instruction = "Respond with a single valid JSON object and nothing else."
        schema = (rf.get("json_schema") or {}).get("schema")
        if schema:
            instruction += " It must match this JSON schema: " + json.dumps(schema)
        system.append(instruction)

    limit = req.get("max_completion_tokens") or req.get("max_tokens")
    body: dict[str, Any] = {
        "model": upstream_model,
        "messages": messages,
        "max_tokens": int(min(limit or ANTHROPIC_DEFAULT_MAX_TOKENS, max_output)),
    }
    if system:
        body["system"] = "\n\n".join(system)
    if req.get("temperature") is not None:
        body["temperature"] = min(float(req["temperature"]), 1.0)
    elif req.get("top_p") is not None:
        body["top_p"] = req["top_p"]
    stop = req.get("stop")
    if stop:
        body["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    if req.get("stream"):
        body["stream"] = True
    if req.get("user"):
        body["metadata"] = {"user_id": str(req["user"])}

    tools = []
    for t in req.get("tools") or []:
        fn = t.get("function") if isinstance(t, dict) else None
        if not fn:
            continue
        tools.append(
            {
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    if tools:
        body["tools"] = tools
        choice = req.get("tool_choice")
        tc: dict[str, Any] | None = None
        if choice == "required":
            tc = {"type": "any"}
        elif choice == "none":
            tc = {"type": "none"}
        elif isinstance(choice, dict) and (choice.get("function") or {}).get("name"):
            tc = {"type": "tool", "name": choice["function"]["name"]}
        elif choice == "auto" or req.get("parallel_tool_calls") is False:
            tc = {"type": "auto"}
        if tc is not None:
            if req.get("parallel_tool_calls") is False and tc["type"] != "none":
                tc["disable_parallel_tool_use"] = True
            body["tool_choice"] = tc
    return body


_ANTHROPIC_STOP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
    "refusal": "content_filter",
}


def _anthropic_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    usage = usage or {}
    cached = int(usage.get("cache_read_input_tokens") or 0)
    prompt = int(usage.get("input_tokens") or 0) + cached + int(usage.get("cache_creation_input_tokens") or 0)
    completion = int(usage.get("output_tokens") or 0)
    out = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
    if cached:
        out["prompt_tokens_details"] = {"cached_tokens": cached}  # type: ignore[assignment]
    return out


def anthropic_to_chat(resp: dict[str, Any], model_id: str) -> dict[str, Any]:
    """Convert an Anthropic Messages response to an OpenAI chat completion."""
    texts, tool_calls = [], []
    for block in resp.get("content") or []:
        if block.get("type") == "text":
            texts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id") or _uid("call"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )
    message: dict[str, Any] = {"role": "assistant", "content": "".join(texts) if texts else None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    elif message["content"] is None:
        message["content"] = ""
    return {
        "id": resp.get("id") or _uid("chatcmpl"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": _ANTHROPIC_STOP.get(resp.get("stop_reason") or "", "stop"),
            }
        ],
        "usage": _anthropic_usage(resp.get("usage")),
    }


class AnthropicStreamToChat:
    """Stateful converter: Anthropic SSE event dicts → OpenAI chat chunks."""

    def __init__(self, model_id: str, include_usage: bool = False) -> None:
        self.model_id = model_id
        self.include_usage = include_usage
        self.id = _uid("chatcmpl")
        self.created = int(time.time())
        self.tool_index: dict[int, int] = {}
        self.usage: dict[str, Any] = {}

    def _chunk(self, delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    def feed(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        kind = event.get("type")
        if kind == "message_start":
            msg = event.get("message") or {}
            self.usage.update(msg.get("usage") or {})
            return [self._chunk({"role": "assistant", "content": ""})]
        if kind == "content_block_start":
            block = event.get("content_block") or {}
            if block.get("type") == "tool_use":
                idx = len(self.tool_index)
                self.tool_index[int(event.get("index", 0))] = idx
                return [
                    self._chunk(
                        {
                            "tool_calls": [
                                {
                                    "index": idx,
                                    "id": block.get("id"),
                                    "type": "function",
                                    "function": {"name": block.get("name", ""), "arguments": ""},
                                }
                            ]
                        }
                    )
                ]
            if block.get("type") == "text" and block.get("text"):
                return [self._chunk({"content": block["text"]})]
            return []
        if kind == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                return [self._chunk({"content": delta.get("text", "")})]
            if delta.get("type") == "input_json_delta":
                idx = self.tool_index.get(int(event.get("index", 0)), 0)
                return [
                    self._chunk(
                        {
                            "tool_calls": [
                                {"index": idx, "function": {"arguments": delta.get("partial_json", "")}}
                            ]
                        }
                    )
                ]
            return []
        if kind == "message_delta":
            self.usage.update(event.get("usage") or {})
            reason = (event.get("delta") or {}).get("stop_reason")
            return [self._chunk({}, _ANTHROPIC_STOP.get(reason or "", "stop"))]
        if kind == "message_stop" and self.include_usage:
            return [
                {
                    "id": self.id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.model_id,
                    "choices": [],
                    "usage": _anthropic_usage(self.usage),
                }
            ]
        return []


# ═══════════════════════ inbound Anthropic Messages API ═══════════════════════


def anthropic_request_to_chat(body: dict[str, Any]) -> dict[str, Any]:
    """Convert an inbound Anthropic Messages request to an OpenAI chat request."""
    messages: list[dict[str, Any]] = []
    system = body.get("system")
    if system:
        text = system if isinstance(system, str) else content_text(system)
        if text:
            messages.append({"role": "system", "content": text})

    for m in body.get("messages") or []:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        parts: list[dict[str, Any]] = []
        tool_calls: list[dict[str, Any]] = []
        for block in content or []:
            kind = block.get("type")
            if kind == "text":
                parts.append({"type": "text", "text": block.get("text", "")})
            elif kind == "image":
                src = block.get("source") or {}
                if src.get("type") == "base64":
                    url = f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"
                else:
                    url = src.get("url", "")
                if url:
                    parts.append({"type": "image_url", "image_url": {"url": url}})
            elif kind == "document":
                src = block.get("source") or {}
                if src.get("type") == "text" and src.get("data"):
                    parts.append({"type": "text", "text": src["data"]})
            elif kind == "tool_use":
                call: dict[str, Any] = {
                    "id": block.get("id") or _uid("toolu"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
                if block.get("extra_content"):
                    call["extra_content"] = block["extra_content"]
                tool_calls.append(call)
            elif kind == "tool_result":
                result = block.get("content")
                text = result if isinstance(result, str) else content_text(result)
                if block.get("is_error"):
                    text = f"ERROR: {text}"
                messages.append(
                    {"role": "tool", "tool_call_id": block.get("tool_use_id", ""), "content": text}
                )
        if role == "assistant":
            text = "".join(p.get("text", "") for p in parts if p["type"] == "text")
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            if msg["content"] is not None or tool_calls:
                messages.append(msg)
        elif parts:
            if all(p["type"] == "text" for p in parts):
                messages.append({"role": role, "content": "\n".join(p["text"] for p in parts)})
            else:
                messages.append({"role": role, "content": parts})

    req: dict[str, Any] = {"model": body.get("model"), "messages": messages}
    if body.get("max_tokens"):
        req["max_tokens"] = body["max_tokens"]
    for key in ("temperature", "top_p", "stream"):
        if body.get(key) is not None:
            req[key] = body[key]
    if body.get("stop_sequences"):
        req["stop"] = body["stop_sequences"]
    tools = []
    for t in body.get("tools") or []:
        if "input_schema" in t:  # server tools (web_search_*, bash_*, ...) have no schema
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
                    },
                }
            )
    if tools:
        req["tools"] = tools
        tc = body.get("tool_choice") or {}
        kind = tc.get("type")
        if kind == "any":
            req["tool_choice"] = "required"
        elif kind == "none":
            req["tool_choice"] = "none"
        elif kind == "tool" and tc.get("name"):
            req["tool_choice"] = {"type": "function", "function": {"name": tc["name"]}}
        if tc.get("disable_parallel_tool_use"):
            req["parallel_tool_calls"] = False
    return req


_TO_ANTHROPIC_STOP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "refusal",
}


def chat_to_anthropic_response(resp: dict[str, Any], model: str) -> dict[str, Any]:
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content: list[dict[str, Any]] = []
    text = content_text(message.get("content"))
    if text:
        content.append({"type": "text", "text": text})
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        block: dict[str, Any] = {
            "type": "tool_use",
            "id": tc.get("id") or _uid("toolu"),
            "name": fn.get("name", ""),
            "input": _parse_args(fn.get("arguments")),
        }
        if tc.get("extra_content"):
            block["extra_content"] = tc["extra_content"]  # opaque (Gemini thought_signature)
        content.append(block)
    usage = resp.get("usage") or {}
    return {
        "id": _uid("msg"),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _TO_ANTHROPIC_STOP.get(choice.get("finish_reason") or "stop", "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens") or 0),
            "output_tokens": int(usage.get("completion_tokens") or 0),
        },
    }


class ChatStreamToAnthropic:
    """OpenAI chat chunks → Anthropic SSE events (list of (event, data))."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.index = -1
        self.open: str | None = None  # "text" | "tool:<openai index>"
        self.finish: str | None = None
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    def start(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": _uid("msg"),
                        "type": "message",
                        "role": "assistant",
                        "model": self.model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
            )
        ]

    def _close(self) -> list[tuple[str, dict[str, Any]]]:
        if self.open is None:
            return []
        self.open = None
        return [("content_block_stop", {"type": "content_block_stop", "index": self.index})]

    def feed(self, chunk: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        out: list[tuple[str, dict[str, Any]]] = []
        usage = chunk.get("usage")
        if usage:
            self.usage["input_tokens"] = int(usage.get("prompt_tokens") or 0)
            self.usage["output_tokens"] = int(usage.get("completion_tokens") or 0)
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if text:
                if self.open != "text":
                    out += self._close()
                    self.index += 1
                    self.open = "text"
                    out.append(
                        (
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": self.index,
                                "content_block": {"type": "text", "text": ""},
                            },
                        )
                    )
                out.append(
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": self.index,
                            "delta": {"type": "text_delta", "text": text},
                        },
                    )
                )
            for tc in delta.get("tool_calls") or []:
                key = f"tool:{tc.get('index', 0)}"
                fn = tc.get("function") or {}
                if self.open != key:
                    out += self._close()
                    self.index += 1
                    self.open = key
                    block: dict[str, Any] = {
                        "type": "tool_use",
                        "id": tc.get("id") or _uid("toolu"),
                        "name": fn.get("name", ""),
                        "input": {},
                    }
                    if tc.get("extra_content"):
                        block["extra_content"] = tc["extra_content"]  # opaque (Gemini thought_signature)
                    out.append(
                        (
                            "content_block_start",
                            {"type": "content_block_start", "index": self.index, "content_block": block},
                        )
                    )
                if fn.get("arguments"):
                    out.append(
                        (
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": self.index,
                                "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]},
                            },
                        )
                    )
            if choice.get("finish_reason"):
                self.finish = choice["finish_reason"]
        return out

    def end(self) -> list[tuple[str, dict[str, Any]]]:
        out = self._close()
        out.append(
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": _TO_ANTHROPIC_STOP.get(self.finish or "stop", "end_turn"),
                        "stop_sequence": None,
                    },
                    "usage": {"output_tokens": self.usage["output_tokens"]},
                },
            )
        )
        out.append(("message_stop", {"type": "message_stop"}))
        return out


# ═════════════════════════ inbound OpenAI Responses API ═════════════════════════


def _responses_content(content: Any) -> Any:
    if isinstance(content, str) or content is None:
        return content or ""
    parts: list[dict[str, Any]] = []
    for part in content:
        kind = part.get("type")
        if kind in ("input_text", "output_text", "text"):
            parts.append({"type": "text", "text": part.get("text", "")})
        elif kind == "input_image":
            url = part.get("image_url")
            if isinstance(url, dict):
                url = url.get("url")
            if url:
                parts.append({"type": "image_url", "image_url": {"url": url}})
    if all(p["type"] == "text" for p in parts):
        return "\n".join(p["text"] for p in parts)
    return parts


def responses_request_to_chat(
    body: dict[str, Any], history: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Convert an inbound Responses API request to an OpenAI chat request."""
    messages: list[dict[str, Any]] = list(history or [])
    if body.get("instructions"):
        messages.insert(0, {"role": "system", "content": body["instructions"]})
    items = body.get("input")
    if isinstance(items, str):
        items = [{"role": "user", "content": items}]
    for item in items or []:
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role", "user")
            if role == "developer":
                role = "system"
            messages.append({"role": role, "content": _responses_content(item.get("content"))})
        elif kind == "function_call":
            call = {
                "id": item.get("call_id") or item.get("id") or _uid("call"),
                "type": "function",
                "function": {"name": item.get("name", ""), "arguments": item.get("arguments") or "{}"},
            }
            last = messages[-1] if messages else None
            if (
                last
                and last.get("role") == "assistant"
                and (last.get("tool_calls") or not last.get("content"))
            ):
                last.setdefault("tool_calls", []).append(call)
            else:
                messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
        elif kind == "function_call_output":
            output = item.get("output")
            if not isinstance(output, str):
                output = content_text(output) if isinstance(output, list) else json.dumps(output)
            messages.append({"role": "tool", "tool_call_id": item.get("call_id", ""), "content": output})

    req: dict[str, Any] = {"model": body.get("model"), "messages": messages}
    if body.get("max_output_tokens"):
        req["max_tokens"] = body["max_output_tokens"]
    for key in ("temperature", "top_p", "stream", "parallel_tool_calls", "user"):
        if body.get(key) is not None:
            req[key] = body[key]
    tools = [
        {
            "type": "function",
            "function": {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("parameters") or {"type": "object", "properties": {}},
            },
        }
        for t in body.get("tools") or []
        if t.get("type") == "function"
    ]
    if tools:
        req["tools"] = tools
        choice = body.get("tool_choice")
        if choice in ("auto", "none", "required"):
            req["tool_choice"] = choice
        elif isinstance(choice, dict) and choice.get("name"):
            req["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    fmt = (body.get("text") or {}).get("format") or {}
    if fmt.get("type") == "json_schema":
        req["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": fmt.get("name", "output"),
                "schema": fmt.get("schema") or {},
                "strict": bool(fmt.get("strict", False)),
            },
        }
    elif fmt.get("type") == "json_object":
        req["response_format"] = {"type": "json_object"}
    return req


def _responses_usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    usage = usage or {}
    i, o = int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
    return {
        "input_tokens": i,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": o,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": i + o,
    }


def _response_shell(body: dict[str, Any], resp_id: str, model: str, status: str) -> dict[str, Any]:
    return {
        "id": resp_id,
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "error": None,
        "incomplete_details": None,
        "instructions": body.get("instructions"),
        "max_output_tokens": body.get("max_output_tokens"),
        "model": model,
        "output": [],
        "parallel_tool_calls": body.get("parallel_tool_calls", True),
        "previous_response_id": body.get("previous_response_id"),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "text": body.get("text") or {"format": {"type": "text"}},
        "tool_choice": body.get("tool_choice", "auto"),
        "tools": body.get("tools") or [],
        "metadata": body.get("metadata") or {},
        "usage": None,
    }


def chat_to_responses(resp: dict[str, Any], body: dict[str, Any], model: str, resp_id: str) -> dict[str, Any]:
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    out = _response_shell(body, resp_id, model, "completed")
    text = content_text(message.get("content"))
    if text:
        out["output"].append(
            {
                "type": "message",
                "id": _uid("msg"),
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        )
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        out["output"].append(
            {
                "type": "function_call",
                "id": _uid("fc"),
                "call_id": tc.get("id") or _uid("call"),
                "name": fn.get("name", ""),
                "arguments": fn.get("arguments") or "{}",
                "status": "completed",
            }
        )
    if choice.get("finish_reason") == "length":
        out["status"] = "incomplete"
        out["incomplete_details"] = {"reason": "max_output_tokens"}
    out["output_text"] = text
    out["usage"] = _responses_usage(resp.get("usage"))
    return out


class ChatStreamToResponses:
    """OpenAI chat chunks → Responses API streaming events."""

    def __init__(self, body: dict[str, Any], model: str, resp_id: str) -> None:
        self.body = body
        self.model = model
        self.resp_id = resp_id
        self.seq = 0
        self.items: list[dict[str, Any]] = []
        self.text_item: dict[str, Any] | None = None
        self.tools: dict[int, dict[str, Any]] = {}
        self.finish: str | None = None
        self.usage: dict[str, Any] | None = None

    def _ev(self, kind: str, **data: Any) -> tuple[str, dict[str, Any]]:
        self.seq += 1
        return kind, {"type": kind, "sequence_number": self.seq, **data}

    def start(self) -> list[tuple[str, dict[str, Any]]]:
        shell = _response_shell(self.body, self.resp_id, self.model, "in_progress")
        return [
            self._ev("response.created", response=shell),
            self._ev("response.in_progress", response=shell),
        ]

    def feed(self, chunk: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        out: list[tuple[str, dict[str, Any]]] = []
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                if self.text_item is None:
                    self.text_item = {
                        "type": "message",
                        "id": _uid("msg"),
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                        "_text": "",
                        "_index": len(self.items),
                    }
                    self.items.append(self.text_item)
                    item = {k: v for k, v in self.text_item.items() if not k.startswith("_")}
                    out.append(
                        self._ev(
                            "response.output_item.added", output_index=self.text_item["_index"], item=item
                        )
                    )
                    out.append(
                        self._ev(
                            "response.content_part.added",
                            item_id=self.text_item["id"],
                            output_index=self.text_item["_index"],
                            content_index=0,
                            part={"type": "output_text", "text": "", "annotations": []},
                        )
                    )
                self.text_item["_text"] += delta["content"]
                out.append(
                    self._ev(
                        "response.output_text.delta",
                        item_id=self.text_item["id"],
                        output_index=self.text_item["_index"],
                        content_index=0,
                        delta=delta["content"],
                    )
                )
            for tc in delta.get("tool_calls") or []:
                idx = int(tc.get("index", 0))
                fn = tc.get("function") or {}
                item = self.tools.get(idx)
                if item is None:
                    item = {
                        "type": "function_call",
                        "id": _uid("fc"),
                        "call_id": tc.get("id") or _uid("call"),
                        "name": fn.get("name", ""),
                        "arguments": "",
                        "status": "in_progress",
                        "_index": len(self.items),
                    }
                    self.tools[idx] = item
                    self.items.append(item)
                    public = {k: v for k, v in item.items() if not k.startswith("_")}
                    out.append(
                        self._ev("response.output_item.added", output_index=item["_index"], item=public)
                    )
                if fn.get("arguments"):
                    item["arguments"] += fn["arguments"]
                    out.append(
                        self._ev(
                            "response.function_call_arguments.delta",
                            item_id=item["id"],
                            output_index=item["_index"],
                            delta=fn["arguments"],
                        )
                    )
            if choice.get("finish_reason"):
                self.finish = choice["finish_reason"]
        return out

    def end(self) -> list[tuple[str, dict[str, Any]]]:
        out: list[tuple[str, dict[str, Any]]] = []
        final_items = []
        for item in self.items:
            if item["type"] == "message":
                text = item["_text"]
                part = {"type": "output_text", "text": text, "annotations": []}
                out.append(
                    self._ev(
                        "response.output_text.done",
                        item_id=item["id"],
                        output_index=item["_index"],
                        content_index=0,
                        text=text,
                    )
                )
                out.append(
                    self._ev(
                        "response.content_part.done",
                        item_id=item["id"],
                        output_index=item["_index"],
                        content_index=0,
                        part=part,
                    )
                )
                done = {
                    "type": "message",
                    "id": item["id"],
                    "status": "completed",
                    "role": "assistant",
                    "content": [part],
                }
            else:
                out.append(
                    self._ev(
                        "response.function_call_arguments.done",
                        item_id=item["id"],
                        output_index=item["_index"],
                        arguments=item["arguments"] or "{}",
                    )
                )
                done = {k: v for k, v in item.items() if not k.startswith("_")}
                done["arguments"] = done["arguments"] or "{}"
                done["status"] = "completed"
            out.append(self._ev("response.output_item.done", output_index=item["_index"], item=done))
            final_items.append(done)
        final = _response_shell(self.body, self.resp_id, self.model, "completed")
        final["output"] = final_items
        final["output_text"] = self.text_item["_text"] if self.text_item else ""
        final["usage"] = _responses_usage(self.usage)
        if self.finish == "length":
            final["status"] = "incomplete"
            final["incomplete_details"] = {"reason": "max_output_tokens"}
        self.final = final
        out.append(self._ev("response.completed", response=final))
        return out
