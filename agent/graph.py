"""The helpdesk agent: agentic RAG with a relevance grader, an answer check and per-buyer memory.

    START -> agent --(tool calls?)--> tools -> agent -> ... -> answer_check --(ok)--> END
                                                                   |
                                                       (problem) --+--> agent (with feedback)

* ``search_documents`` runs hybrid search, then a grader (an LLM call) keeps only the passages
  that actually help with the buyer's question. If none survive, the tool tells the agent to
  rewrite its query and search again.
* ``answer_check`` is plain code: every cited ID must come from this turn's tool results, an
  answer built on retrieved passages must cite them, and a factual answer without any search is
  sent back. At most MAX_CHECKS round trips.
* Memory: facts saved with ``remember`` are injected into the system prompt on every turn.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import date
from typing import Annotated, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode

from .documents import load_buyers, rera_section_text
from .memory import MemoryStore
from .retrieval import HybridRetriever, tokenize

MAX_STEPS = 40
MAX_CHECKS = 2
CHECK_PREFIX = "Answer check:"
NOT_FOUND = re.compile(
    r"couldn.t find|could not find|don.t have (that|this) information|not (in|covered by) the documents", re.I
)
CITATION = re.compile(r"\[([^\[\]]+)\]")
CITATION_ID = re.compile(r"^(booking|[a-z_]+-[\w-]+)$")
FACTUAL = re.compile(r"\d|₹|%|\bsection\b", re.I)

SYSTEM_PROMPT = """You are the buyer helpdesk assistant for Riverside Heights, a fictional residential project
by Sunrise Habitat Developers in New Town, Kolkata. You answer buyers' questions from two sources:
the project's own documents (FAQ, payment plan, possession, refunds, maintenance, escalation) and the
Real Estate (Regulation and Development) Act, 2016 ("RERA Act").

How to work:
1. For any question about the project, payments, possession, refunds, maintenance or the law, call
   search_documents first. Use source="project" for project policies, "rera" for the law, and "both"
   when unsure or when the buyer's rights under the law matter. Put several searches in one message
   when a question has several parts.
2. search_documents returns only passages a grader judged relevant. If it returns none, rewrite the
   query with different words (synonyms, or legal terms such as "allottee", "promoter", "refund with
   interest") and search again. At most 3 searches per question.
3. Use read_rera_section for the full text of a section, and get_my_booking for questions about the
   buyer's own flat, price or payments.
4. Answer only from the passages and the booking record. Cite every fact with its ID in square
   brackets, e.g. [payment_plan-3], [rera-18] or [booking]. Never cite an ID you did not receive.
5. If the documents don't answer the question, say "I couldn't find this in the documents" and suggest
   the customer care desk. Do not guess.
6. When the buyer tells you a lasting fact about themselves or their plans (e.g. "I'm taking a home
   loan", "I want to add my wife as co-buyer"), call remember with one short sentence. Never store ID,
   account or phone numbers.
7. Be brief and plain: 2-6 sentences or a short list. This is information, not legal advice.

Today's date: {today}.
Buyer: {buyer}
What you remember about this buyer:
{memory}"""


class HelpdeskState(MessagesState):
    buyer_id: str


# ------------------------------------------------------------------ helpers
def turn_messages(messages) -> list:
    """Messages after the buyer's latest question (answer-check feedback does not start a new turn)."""
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, HumanMessage) and not str(msg.content).startswith(CHECK_PREFIX):
            return messages[i:]
    return list(messages)


def latest_question(messages) -> str:
    turn = turn_messages(messages)
    return str(turn[0].content) if turn and isinstance(turn[0], HumanMessage) else ""


def message_text(msg) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(p for p in parts if p)


def _tool_payload(msg: ToolMessage) -> dict:
    try:
        data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
    except (json.JSONDecodeError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def retrieved_passages(messages) -> dict[str, dict]:
    """Passages (and the booking record) returned by tools during the latest turn, by ID."""
    found: dict[str, dict] = {}
    for msg in turn_messages(messages):
        if not isinstance(msg, ToolMessage):
            continue
        data = _tool_payload(msg)
        for p in data.get("passages", []):
            found[p["id"]] = p
        if msg.name == "read_rera_section" and data.get("id"):
            found[data["id"]] = data
        if msg.name == "get_my_booking" and data.get("unit"):
            found["booking"] = {"id": "booking", "title": "Your booking record", "text": json.dumps(data)}
    return found


def cited_ids(text: str) -> list[str]:
    ids: list[str] = []
    for group in CITATION.findall(text):
        for part in re.split(r"[,;]\s*", group.strip()):
            if CITATION_ID.match(part.strip()) and part.strip() not in ids:
                ids.append(part.strip())
    return ids


def searched_this_turn(messages) -> bool:
    return any(
        isinstance(m, ToolMessage) and m.name in ("search_documents", "read_rera_section", "get_my_booking")
        for m in turn_messages(messages)
    )


def answer_problems(messages) -> list[str]:
    """Deterministic checks on the final answer of the latest turn."""
    answer = message_text(messages[-1])
    retrieved = retrieved_passages(messages)
    cited = cited_ids(answer)
    problems = []
    unknown = [c for c in cited if c not in retrieved]
    if unknown:
        problems.append(
            f"you cited {', '.join(unknown)}, which no tool returned in this turn. "
            "Remove those citations or search for passages that support the claim."
        )
    if NOT_FOUND.search(answer):
        return problems
    if retrieved and not cited:
        problems.append("cite the passage IDs your answer is based on, in square brackets.")
    if not searched_this_turn(messages) and FACTUAL.search(answer):
        problems.append("search the documents before answering a factual question, then answer with citations.")
    return problems


def heuristic_grade(question: str, query: str, hits) -> list[str]:
    """Fallback grader: rank passages by how many of the question's words they share and keep
    those with at least half the best overlap."""
    words = set(tokenize(question)) | set(tokenize(query))
    overlap = {p.id: len(words & set(tokenize(f"{p.title} {p.text}"))) for p in hits}
    best = max(overlap.values(), default=0)
    if best == 0:
        return []
    keep = [p.id for p in hits if overlap[p.id] >= max(1, best / 2)]
    return sorted(keep, key=lambda i: -overlap[i])


GRADER_PROMPT = """You check search results for a real-estate buyer helpdesk.
Buyer's question: {question}
Search query used: {query}

Passages:
{passages}

Which passages contain information that helps answer the buyer's question? Reply with only a JSON list
of passage IDs, most useful first, for example ["faq-2", "rera-18"]. Reply [] if none help."""


def make_grader(models: Sequence[BaseChatModel]):
    def grade(question: str, query: str, hits) -> tuple[list[str], str]:
        if not hits:
            return [], "no hits"
        hit_ids = [p.id for p in hits]
        listing = "\n".join(f"[{p.id}] {p.title}: {p.text[:700]}" for p in hits)
        prompt = GRADER_PROMPT.format(question=question or query, query=query, passages=listing)
        for model in models:
            try:
                reply = message_text(model.invoke(prompt))
            except Exception:  # grader is best-effort: try the next provider, then the heuristic
                continue
            ids = [i for i in re.findall(r"[a-z_]+-[\w-]+", reply) if i in hit_ids]
            return list(dict.fromkeys(ids)), "llm"
        return heuristic_grade(question, query, hits), "keyword overlap"

    return grade


# ------------------------------------------------------------------ graph
def build_graph(
    llm: BaseChatModel,
    retriever: HybridRetriever,
    memory: MemoryStore,
    fallbacks: Sequence[BaseChatModel] = (),
    grader_models: Sequence[BaseChatModel] | None = None,
):
    buyers = load_buyers()
    grade = make_grader(list(grader_models) if grader_models is not None else [llm, *fallbacks])

    @tool
    def search_documents(
        query: str,
        state: Annotated[dict, InjectedState],
        source: Literal["both", "project", "rera"] = "both",
    ) -> dict:
        """Search the Riverside Heights project documents and/or the RERA Act 2016. Returns only the
        passages a grader judged relevant to the buyer's question, each with an ID to cite."""
        hits = retriever.search(query, source=source, k=6)
        keep, graded_by = grade(latest_question(state["messages"]), query, hits)
        kept = [p for p in hits if p.id in keep]
        kept.sort(key=lambda p: keep.index(p.id))
        result = {
            "query": query,
            "source": source,
            "graded_by": graded_by,
            "passages": [p.as_dict() for p in kept],
            "dropped_as_irrelevant": [p.id for p in hits if p.id not in keep],
        }
        if not kept:
            result["note"] = "No relevant passages. Rewrite the query with different words and search again."
        return result

    @tool
    def read_rera_section(section: str) -> dict:
        """Full text of one section of the RERA Act 2016, e.g. section="18"."""
        sec = rera_section_text(section)
        if sec is None:
            return {"error": f"Section {section} not found in the Act."}
        return {
            "id": f"rera-{sec['section']}",
            "title": f"RERA Act 2016 › Section {sec['section']}. {sec['title']}",
            "text": sec["text"],
        }

    @tool
    def get_my_booking(state: Annotated[dict, InjectedState]) -> dict:
        """The signed-in buyer's booking record: unit, carpet area, price, booking date, instalments paid.
        Cite it as [booking]."""
        record = buyers.get(state.get("buyer_id", ""))
        return dict(record) if record else {"error": "No booking record for this user."}

    @tool
    def remember(fact: str, state: Annotated[dict, InjectedState]) -> str:
        """Save one short, lasting fact the buyer told you about themselves or their plans."""
        return memory.add(state.get("buyer_id", "guest"), fact)

    tools = [search_documents, read_rera_section, get_my_booking, remember]
    models = [(m.__class__.__name__, m.bind_tools(tools)) for m in (llm, *fallbacks)]

    def system_prompt(buyer_id: str) -> str:
        record = buyers.get(buyer_id)
        buyer = f"{record['name']}, flat {record['unit']}" if record else f"{buyer_id or 'guest'} (no booking record)"
        facts = memory.list(buyer_id) if buyer_id else []
        lines = "\n".join(f"- {f['fact']}" for f in facts) or "(nothing yet)"
        return SYSTEM_PROMPT.format(today=date.today().isoformat(), buyer=buyer, memory=lines)

    def agent(state: HelpdeskState) -> dict:
        messages = [SystemMessage(system_prompt(state.get("buyer_id", ""))), *state["messages"]]
        errors = []
        for name, model in models:
            try:
                reply = model.invoke(messages)
            except Exception as exc:  # try the next provider, but keep every error
                errors.append(f"{name}: {type(exc).__name__}: {str(exc)[:300]}")
                continue
            if errors:
                reply.response_metadata = {**(reply.response_metadata or {}), "fallback_errors": errors}
            return {"messages": [reply]}
        raise RuntimeError("Every model provider failed:\n- " + "\n- ".join(errors))

    def answer_check(state: HelpdeskState) -> dict:
        problems = answer_problems(state["messages"])
        return {"messages": [HumanMessage(f"{CHECK_PREFIX} " + " Also ".join(problems) + " Then answer again.")]}

    def route(state: HelpdeskState) -> str:
        messages = state["messages"]
        if getattr(messages[-1], "tool_calls", None):
            return "tools"
        checks = sum(
            isinstance(m, HumanMessage) and str(m.content).startswith(CHECK_PREFIX) for m in turn_messages(messages)
        )
        if checks < MAX_CHECKS and answer_problems(messages):
            return "answer_check"
        return END

    graph = StateGraph(HelpdeskState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("answer_check", answer_check)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, ["tools", "answer_check", END])
    graph.add_edge("tools", "agent")
    graph.add_edge("answer_check", "agent")
    return graph.compile()


def run_config() -> dict:
    return {"recursion_limit": MAX_STEPS, "run_name": "buyer-helpdesk"}


def answered_by(messages) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            meta = msg.response_metadata or {}
            return meta.get("model_name") or meta.get("model") or ""
    return ""


def fallback_errors(messages) -> list[str]:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and (msg.response_metadata or {}).get("fallback_errors"):
            return msg.response_metadata["fallback_errors"]
    return []
