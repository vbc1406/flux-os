"""Request validation: type-invalid input is a 400 with a clear message, never a 500.

Messages are generic and never echo request values.
"""

from __future__ import annotations

from typing import Any

from .router import RoutingError


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _fail(msg: str) -> RoutingError:
    return RoutingError(msg)


def _check_scalars(body: dict[str, Any], *, int_keys: tuple[str, ...], num_keys: tuple[str, ...]) -> None:
    model = body.get("model")
    if model is not None and not isinstance(model, str):
        raise _fail("'model' must be a string")
    for key in int_keys:
        v = body.get(key)
        if v is not None and (not _int(v) or v < 1):
            raise _fail(f"'{key}' must be a positive integer")
    for key in num_keys:
        v = body.get(key)
        if v is not None and not _num(v):
            raise _fail(f"'{key}' must be a number")
    stream = body.get("stream")
    if stream is not None and not isinstance(stream, bool):
        raise _fail("'stream' must be a boolean")
    ptc = body.get("parallel_tool_calls")
    if ptc is not None and not isinstance(ptc, bool):
        raise _fail("'parallel_tool_calls' must be a boolean")
    user = body.get("user")
    if user is not None and not isinstance(user, str):
        raise _fail("'user' must be a string")


def _check_list_of_dicts(value: Any, name: str) -> None:
    if value is not None and (not isinstance(value, list) or not all(isinstance(x, dict) for x in value)):
        raise _fail(f"'{name}' must be a list of objects")


def _check_content(content: Any, where: str) -> None:
    if content is None or isinstance(content, str):
        return
    if not isinstance(content, list) or not all(isinstance(p, (dict, str)) for p in content):
        raise _fail(f"{where} 'content' must be a string or a list of content parts")


def validate_chat_request(req: dict[str, Any]) -> None:
    """Validate an OpenAI Chat Completions body (also the internal shape for every endpoint)."""
    _check_scalars(
        req,
        int_keys=("max_tokens", "max_completion_tokens", "n"),
        num_keys=("temperature", "top_p", "presence_penalty", "frequency_penalty"),
    )
    seed = req.get("seed")
    if seed is not None and not _int(seed):
        raise _fail("'seed' must be an integer")
    stop = req.get("stop")
    if stop is not None and not (
        isinstance(stop, str) or (isinstance(stop, list) and all(isinstance(s, str) for s in stop))
    ):
        raise _fail("'stop' must be a string or a list of strings")
    _check_list_of_dicts(req.get("tools"), "tools")
    for t in req.get("tools") or []:
        fn = t.get("function")
        if fn is not None and not isinstance(fn, dict):
            raise _fail("'tools[].function' must be an object")
    choice = req.get("tool_choice")
    if choice is not None and not isinstance(choice, (str, dict)):
        raise _fail("'tool_choice' must be a string or an object")
    for key in ("response_format", "stream_options"):
        if req.get(key) is not None and not isinstance(req[key], dict):
            raise _fail(f"'{key}' must be an object")
    messages = req.get("messages")
    if not isinstance(messages, list):
        raise _fail("'messages' must be a list")
    for m in messages:
        if not isinstance(m, dict):
            raise _fail("'messages' must be a list of objects")
        if not isinstance(m.get("role"), str):
            raise _fail("every message needs a string 'role'")
        _check_content(m.get("content"), f"'{m['role']}' message")
        calls = m.get("tool_calls")
        if calls is not None:
            _check_list_of_dicts(calls, "messages[].tool_calls")
            for c in calls:
                fn = c.get("function")
                if fn is not None and not isinstance(fn, dict):
                    raise _fail("'tool_calls[].function' must be an object")


def validate_anthropic_body(body: dict[str, Any]) -> None:
    """Shape checks on a raw inbound Anthropic Messages body, before translation."""
    _check_scalars(body, int_keys=("max_tokens",), num_keys=("temperature", "top_p"))
    messages = body.get("messages")
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        raise _fail("'messages' must be a list of objects")
    for m in messages:
        if not isinstance(m.get("role"), str):
            raise _fail("every message needs a string 'role'")
        content = m.get("content")
        if content is not None and not isinstance(content, (str, list)):
            raise _fail("message 'content' must be a string or a list of content blocks")
        if isinstance(content, list) and not all(isinstance(b, dict) for b in content):
            raise _fail("message 'content' blocks must be objects")
    system = body.get("system")
    if system is not None and not isinstance(system, (str, list)):
        raise _fail("'system' must be a string or a list of blocks")
    _check_list_of_dicts(body.get("tools"), "tools")
    if body.get("tool_choice") is not None and not isinstance(body["tool_choice"], dict):
        raise _fail("'tool_choice' must be an object")
    stop = body.get("stop_sequences")
    if stop is not None and not (isinstance(stop, list) and all(isinstance(s, str) for s in stop)):
        raise _fail("'stop_sequences' must be a list of strings")


def validate_responses_body(body: dict[str, Any]) -> None:
    """Shape checks on a raw inbound OpenAI Responses body, before translation."""
    _check_scalars(body, int_keys=("max_output_tokens",), num_keys=("temperature", "top_p"))
    items = body.get("input")
    if items is not None and not isinstance(items, str):
        if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
            raise _fail("'input' must be a string or a list of objects")
        for i in items:
            _check_content(i.get("content"), "input item")
    if body.get("instructions") is not None and not isinstance(body["instructions"], str):
        raise _fail("'instructions' must be a string")
    _check_list_of_dicts(body.get("tools"), "tools")
    for key in ("text", "metadata"):
        if body.get(key) is not None and not isinstance(body[key], dict):
            raise _fail(f"'{key}' must be an object")
    fmt = (body.get("text") or {}).get("format")
    if fmt is not None and not isinstance(fmt, dict):
        raise _fail("'text.format' must be an object")
    prev = body.get("previous_response_id")
    if prev is not None and not isinstance(prev, str):
        raise _fail("'previous_response_id' must be a string")
