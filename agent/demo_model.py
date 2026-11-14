"""A rule-based stand-in for the language model, used when no API key is set.

It drives the real graph, tools, grader fallback and answer check. It searches with the buyer's
own words, rewrites the query once if nothing relevant comes back, and answers by quoting the
best passage with its citation. The live model writes real answers; this keeps the demo free.
"""

from __future__ import annotations

import json
import re
from itertools import count
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .graph import latest_question, turn_messages
from .retrieval import tokenize

MY_RECORD = re.compile(r"\bmy (flat|apartment|booking|payments?|unit|price)\b", re.I)


class ScriptedHelpdeskModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "scripted-helpdesk-demo"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedHelpdeskModel:
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        question = latest_question(messages)
        results = _results(turn_messages(messages))
        ids = count(len(results) + 1)

        def call(name: str, **args) -> dict:
            return {"name": name, "args": args, "id": f"demo_{next(ids)}", "type": "tool_call"}

        if not question:  # the grader prompt arrives as plain text, not a conversation
            return AIMessage(content="[]")

        lowered = question.lower()
        if lowered.startswith("remember") and not any(n == "remember" for n, _ in results):
            fact = re.sub(r"^remember( that)?[:,]?\s*", "", question, flags=re.I)
            return AIMessage(content="Saving that.", tool_calls=[call("remember", fact=fact)])
        if lowered.startswith("remember"):
            return AIMessage(content="Noted. I'll remember that for next time.")

        searches = [p for n, p in results if n == "search_documents"]
        if not searches:
            calls = [call("search_documents", query=question, source="both")]
            if MY_RECORD.search(question):
                calls.append(call("get_my_booking"))
            return AIMessage(content="Searching the project documents and the RERA Act.", tool_calls=calls)

        passages = [p for s in searches for p in s.get("passages", [])]
        if not passages and len(searches) < 2:
            rewritten = " ".join(tokenize(question)) or question
            return AIMessage(
                content="Nothing relevant yet. Trying again with just the key words.",
                tool_calls=[call("search_documents", query=rewritten, source="both")],
            )
        if not passages:
            return AIMessage(content="I couldn't find this in the documents. Please ask the customer care desk.")

        top = passages[0]
        sentences = re.split(r"(?<=[.!?])\s+", top["text"])
        answer = f"{' '.join(sentences[:2])} [{top['id']}]"
        booking = next((p for n, p in results if n == "get_my_booking" and p.get("unit")), None)
        if booking:
            answer += (
                f"\n\nYour flat is {booking['unit']} with {booking['carpet_area_sq_ft']} sq ft carpet area. [booking]"
            )
        answer += "\n\n_Demo mode quotes the best passage. Add an API key for full answers._"
        return AIMessage(content=answer)


def _results(messages: list[BaseMessage]) -> list[tuple[str, dict]]:
    out = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            except (json.JSONDecodeError, TypeError):
                data = {"text": msg.content}
            out.append((msg.name, data if isinstance(data, dict) else {"text": data}))
    return out
