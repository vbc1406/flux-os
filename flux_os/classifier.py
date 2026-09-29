"""Heuristic request classifier: task type, complexity, and hard requirements.

Pure Python regexes — no LLM call, no network, well under a millisecond.
Works on OpenAI-shaped chat messages (the canonical format inside flux-os).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_CODE_REVIEW = re.compile(
    r"\b(review\s+(this\s+|my\s+|the\s+)?(code|function|class|pr|pull\s+request|diff)|debug\w*|"
    r"fix\s+(this|the|my)\b|what(?:'s|\s+is)\s+wrong|find\s+(the\s+)?bugs?|why\s+(does|is)\s+"
    r"(this|my)\s+(code|function|test)|stack\s*trace|traceback|refactor)\b",
    re.I,
)
_CODE_DEFECT = re.compile(
    r"\b(race\s+condition|deadlock|memory\s+leak|null\s+pointer|off.by.one|segfault|"
    r"use.after.free|buffer\s+overflow|data\s+race)\b",
    re.I,
)
_CODE_GEN = re.compile(
    r"\b(write\s+(a\s+|an\s+|the\s+)?(\w+\s+){0,2}(function|class|script|program|code|module|"
    r"query|regex|test|endpoint|component|cli)|implement\w*|"
    r"(build|create|make|scaffold)\s+(a\s+|an\s+)?(\w+\s+){0,2}(function|class|api|service|app|"
    r"application|component|server|client|script|cli|bot|website|endpoint))\b",
    re.I,
)
_LANG = re.compile(
    r"\b(python|javascript|typescript|java|golang|rust|ruby|swift|kotlin|php|scala|c\+\+|c#|"
    r"sql|bash|react|vue|django|flask|fastapi|node\.?js|next\.?js|rails|spring)\b",
    re.I,
)
_TRANSLATE = re.compile(
    r"\b(translat\w*|in\s+(spanish|french|german|italian|portuguese|japanese|chinese|korean|"
    r"arabic|russian|hindi|dutch)\b)",
    re.I,
)
_SUMMARIZE = re.compile(r"\b(summar\w*|tl;?dr|key\s+points|main\s+ideas|recap|condense)\b", re.I)
_EXTRACT = re.compile(
    r"\b(extract|parse\s+(all|the|this)|pull\s+out|find\s+all|list\s+(all|every)|"
    r"(return|output)\s+(\w+\s+){0,3}(json|fields|csv))\b",
    re.I,
)
_CLASSIFY = re.compile(
    r"\b(classif\w*|categori[sz]\w*|label\s+(this|each|the)|sentiment|is\s+this\s+(spam|positive|"
    r"negative|toxic))\b",
    re.I,
)
_CREATIVE = re.compile(
    r"\b(write\s+(a|an|me\s+a)\s+(\w+\s+){0,3}(story|poem|haiku|song|lyrics|novel|chapter|scene|"
    r"essay|blog|article|speech|tweet|joke|limerick|script)|brainstorm|come\s+up\s+with|"
    r"imagine\s+(a|if|that)|slogan|tagline)\b",
    re.I,
)
_REASONING = re.compile(
    r"\b(prove|proof|theorem|lemma|derive|derivation|solve|step[\s-]by[\s-]step|why\s+(does|is|do|"
    r"did|would)|explain\s+why|how\s+many|probability|puzzle|riddle|logic(al)?|optimi[sz]e|"
    r"algorithm|complexity\s+of|big.o)\b",
    re.I,
)
_ANALYSIS = re.compile(
    r"\b(analy[sz]\w*|compar\w*|evaluat\w*|assess\w*|critique|pros\s+and\s+cons|trade.?offs?|"
    r"implications|strategy|recommend\w*|explain\w*|design\s+(a|an|the))\b",
    re.I,
)
_SIMPLE_QA = re.compile(
    r"^\s*(what\s+(is|are|was)|what's|who\s+(is|was)|when\s+(is|was|did)|where\s+(is|was)|"
    r"define|how\s+(old|tall|far|big)|which\s+(is|was))\b",
    re.I,
)
_GREETING = re.compile(
    r"^\s*(hi|hello|hey|yo|thanks|thank\s+you|good\s+(morning|evening|night)|ok|okay|cool)\b", re.I
)
_HARD = re.compile(
    r"\b(production.grade|distributed|concurren\w+|scalab\w+|lock.free|consensus|byzantine|"
    r"cryptograph\w*|formal(ly)?\s+verif\w*|rigorous\w*|edge\s+cases|thread.safe|"
    r"lagrangian|eigen\w+|fourier|stochastic|bayesian|differential\s+equation|"
    r"architecture|security\s+audit|vulnerabilit\w+|end.to.end)\b",
    re.I,
)
_CONSTRAINT = re.compile(
    r"\b(without|instead\s+of|rather\s+than|must\s+not|must\s+be|cannot|can't|ensure|guarantee|"
    r"at\s+most|at\s+least|no\s+more\s+than|handle\s+(all|invalid|malformed|edge))\b",
    re.I,
)
_BRIEF = re.compile(
    r"\b(briefly|in\s+one\s+(word|sentence|line)|yes\s+or\s+no|short\s+answer|one.liner|tl;?dr)\b",
    re.I,
)
_MATH = re.compile(r"[∑∫≥≤∀∃≠√∞±∇∂∈⊆∩∪]|\\frac|\\sum|\\int|\d\s*[\^*/]\s*\d")
_PLAN = re.compile(
    r"\b(plan\s+(the|your|out)|make\s+a\s+plan|break\s+(this|it)\s+(down|into)|next\s+step|"
    r"decide\s+(which|what|the\s+next)|what\s+should\s+(i|we|you)\s+do)\b",
    re.I,
)
_FENCE = re.compile(r"```")

BASE_COMPLEXITY = {
    "conversation": 0.08,
    "simple_qa": 0.12,
    "classification": 0.2,
    "translation": 0.25,
    "extraction": 0.28,
    "summarization": 0.3,
    "general": 0.3,
    "creative_writing": 0.35,
    "code_generation": 0.45,
    "long_document": 0.45,
    "analysis": 0.48,
    "code_review": 0.5,
    "reasoning": 0.55,
}

# Typical answer length per task, used for cost estimates only.
OUTPUT_TOKENS = {
    "conversation": 120,
    "simple_qa": 150,
    "classification": 40,
    "translation": 400,
    "extraction": 400,
    "summarization": 350,
    "general": 500,
    "creative_writing": 900,
    "code_generation": 1200,
    "long_document": 800,
    "analysis": 900,
    "code_review": 900,
    "reasoning": 1000,
}


@dataclass
class Analysis:
    task: str
    complexity: float
    input_tokens: int
    output_tokens: int
    needs_tools: bool = False
    needs_vision: bool = False
    needs_json: bool = False
    agent_step: str | None = None  # "tool_call" | "tool_result" | "plan" | None
    signals: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "complexity": round(self.complexity, 3),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "needs_tools": self.needs_tools,
            "needs_vision": self.needs_vision,
            "needs_json": self.needs_json,
            "agent_step": self.agent_step,
            "signals": self.signals,
        }


def content_text(content: Any) -> str:
    """Text of an OpenAI message ``content`` (string or list of parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") in ("text", "input_text", "output_text"):
                    parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _has_image(content: Any) -> bool:
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") in ("image_url", "input_image", "image") for b in content
    )


def estimate_tokens(text: str) -> int:
    """~4 chars/token for prose, ~3 for code-heavy text. No tokenizer needed."""
    if not text:
        return 0
    symbols = sum(1 for c in text if c in "{}[]();=<>")
    per = 3.0 if symbols / max(len(text), 1) > 0.04 else 4.0
    return max(1, int(len(text) / per))


def _detect_task(text: str, total_tokens: int) -> tuple[str, str]:
    """Return (task, matched rule). Most specific patterns first."""
    has_fence = bool(_FENCE.search(text))
    if _CODE_DEFECT.search(text) or (_CODE_REVIEW.search(text) and (has_fence or _LANG.search(text))):
        return "code_review", "code_review"
    if _CODE_GEN.search(text) or (has_fence and _LANG.search(text)):
        return "code_generation", "code_generation"
    if _CODE_REVIEW.search(text):
        return "code_review", "code_review"
    if _TRANSLATE.search(text):
        return "translation", "translation"
    if total_tokens > 12_000 and (_SUMMARIZE.search(text) or _ANALYSIS.search(text)):
        return "long_document", "long_document"
    if _SUMMARIZE.search(text):
        return "summarization", "summarization"
    if _CLASSIFY.search(text):
        return "classification", "classification"
    if _EXTRACT.search(text):
        return "extraction", "extraction"
    if _CREATIVE.search(text):
        return "creative_writing", "creative"
    if _REASONING.search(text) or _MATH.search(text):
        return "reasoning", "reasoning"
    if _ANALYSIS.search(text):
        return "analysis", "analysis"
    words = len(text.split())
    if _GREETING.search(text) and words <= 8:
        return "conversation", "greeting"
    if _SIMPLE_QA.search(text) and words <= 25:
        return "simple_qa", "simple_question"
    if has_fence:
        return "code_generation", "code_block"
    if words <= 12:
        return "conversation", "short"
    return "general", "default"


def classify(
    messages: list[dict[str, Any]],
    tools: list[Any] | None = None,
    response_format: Any = None,
    max_tokens: int | None = None,
) -> Analysis:
    """Classify an OpenAI-style chat request."""
    messages = messages or []
    all_text = []
    needs_vision = False
    for m in messages:
        c = m.get("content")
        all_text.append(content_text(c))
        if _has_image(c):
            needs_vision = True
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            all_text.append(str(fn.get("arguments", "")))
    tool_text = json.dumps(tools) if tools else ""
    input_tokens = estimate_tokens("\n".join(all_text)) + estimate_tokens(tool_text) + 4 * len(messages)

    # Classify on the latest user turn — that's what the model must answer now.
    user_text = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            user_text = content_text(m.get("content"))
            break
    if not user_text and messages:
        user_text = content_text(messages[-1].get("content"))
    system_text = " ".join(
        content_text(m.get("content")) for m in messages if m.get("role") in ("system", "developer")
    )

    task, rule = _detect_task(user_text, input_tokens)
    if task in ("conversation", "general") and system_text:
        sys_task, _ = _detect_task(system_text, 0)
        if sys_task in ("code_generation", "code_review", "reasoning", "analysis"):
            task, rule = sys_task, f"system_prompt:{sys_task}"
    signals = [f"task:{rule}"]

    score = BASE_COMPLEXITY[task]
    words = len(user_text.split())

    def bump(amount: float, name: str) -> None:
        nonlocal score
        score += amount
        signals.append(f"{name}{amount:+.2f}")

    if words > 150:
        bump(0.08, "long_prompt")
    if words > 600:
        bump(0.07, "very_long_prompt")
    if input_tokens > 30_000:
        bump(0.08, "large_context")
    hard_hits = len({m.group(0).lower() for m in _HARD.finditer(user_text)})
    if hard_hits:
        bump(min(0.08 * hard_hits, 0.24), "hard_domain")
    constraints = len(_CONSTRAINT.findall(user_text))
    if constraints >= 2:
        bump(min(0.04 * constraints, 0.16), "constraints")
    if _MATH.search(user_text):
        bump(0.08, "math")
    if user_text.count("?") >= 3:
        bump(0.05, "multi_question")
    numbered = len(re.findall(r"^\s*(\d+[.)]|[-*•])\s", user_text, re.M))
    if numbered >= 4:
        bump(0.06, "multi_part")
    if _BRIEF.search(user_text):
        bump(-0.1, "brief_answer")
    if words <= 6 and task in ("conversation", "simple_qa", "general"):
        bump(-0.04, "tiny_prompt")

    needs_tools = bool(tools)
    agent_step = None
    last = messages[-1] if messages else {}
    if last.get("role") == "tool" or (
        isinstance(last.get("content"), list)
        and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in last["content"])
    ):
        agent_step = "tool_result"
    elif _PLAN.search(user_text) and needs_tools:
        agent_step = "plan"
        bump(0.1, "agent_plan")
    elif needs_tools:
        agent_step = "tool_call"
    if needs_tools:
        signals.append(f"agent:{agent_step}")

    needs_json = False
    if isinstance(response_format, dict) and response_format.get("type") in (
        "json_object",
        "json_schema",
    ):
        needs_json = True
        signals.append("json_output")

    out = OUTPUT_TOKENS[task]
    if max_tokens:
        out = min(out, int(max_tokens))

    return Analysis(
        task=task,
        complexity=max(0.0, min(1.0, score)),
        input_tokens=input_tokens,
        output_tokens=out,
        needs_tools=needs_tools,
        needs_vision=needs_vision,
        needs_json=needs_json,
        agent_step=agent_step,
        signals=signals,
    )
