"""Streamlit chat UI for the Buyer Helpdesk Agent."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.documents import load_all, load_buyers
from agent.graph import (
    CHECK_PREFIX,
    answered_by,
    build_graph,
    cited_ids,
    fallback_errors,
    message_text,
    retrieved_passages,
    run_config,
)
from agent.llm import get_model
from agent.memory import MemoryStore
from agent.retrieval import HybridRetriever

st.set_page_config(page_title="Buyer Helpdesk Agent", page_icon="🏠", layout="wide")

ROOT = Path(__file__).parent
GITHUB = "https://github.com/Pallav6268/buyer-helpdesk-agent"
LINKEDIN = "https://www.linkedin.com/in/pallav-kumar-ai"
WEEK1 = "https://clause-deviation-agent.streamlit.app"
MODEL_SETTINGS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_MODEL",
    "DEEPSEEK_API_KEY",
    "DEEPSEEK_MODEL",
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "MODEL_ORDER",
)
EXAMPLES = [
    "How do I get my money back if I cancel?",
    "What happens if possession is delayed? What does RERA say?",
    "What are the maintenance charges for my flat?",
    "Remember that I'm planning to take a home loan.",
]


def load_secrets_into_env() -> None:
    """Copy Streamlit secrets (top level or inside any [section]) into the environment."""
    try:
        secrets = st.secrets.to_dict()
    except Exception:  # no secrets file (local run)
        return

    def put(mapping: dict) -> None:
        for key, value in mapping.items():
            if isinstance(value, dict):
                put(value)
            elif isinstance(value, str) and value.strip():
                os.environ[key.strip().upper()] = value.strip()

    put(secrets)


def fingerprint(paths: list[Path]) -> str:
    return hashlib.sha256(b"".join(p.read_bytes() for p in sorted(paths))).hexdigest()[:12]


def settings_fingerprint() -> tuple:
    """Changes when a model setting, the agent code or the documents change."""
    code = fingerprint([*(ROOT / "agent").glob("*.py"), *(ROOT / "data").rglob("*.*")])
    settings = (hashlib.sha256(os.getenv(name, "").encode()).hexdigest()[:12] for name in MODEL_SETTINGS)
    return (code, *settings)


@st.cache_resource(show_spinner="Indexing the documents…")
def load_retriever(data_version: str) -> HybridRetriever:
    return HybridRetriever(load_all())


@st.cache_resource
def load_memory() -> MemoryStore:
    return MemoryStore()


@st.cache_resource
def load_agent(settings: tuple):
    choice = get_model()
    retriever = load_retriever(settings[0])
    grader = [choice.llm, *choice.fallbacks] if choice.live else []
    return build_graph(choice.llm, retriever, load_memory(), choice.fallbacks, grader_models=grader), choice


def short(text: str, limit: int = 90) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def trace_line(node: str, msg) -> str | None:
    """One readable line per step for the 'How I answered' panel."""
    if node == "agent" and getattr(msg, "tool_calls", None):
        parts = []
        for c in msg.tool_calls:
            args = ", ".join(f"{k}={short(json.dumps(v, ensure_ascii=False), 70)}" for k, v in c["args"].items())
            parts.append(f"`{c['name']}({args})`")
        return "🧠 Decided to call " + " · ".join(parts)
    if node == "tools" and isinstance(msg, ToolMessage):
        try:
            data = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError):
            data = {}
        if msg.name == "search_documents" and isinstance(data, dict):
            kept = [p["id"] for p in data.get("passages", [])]
            dropped = data.get("dropped_as_irrelevant", [])
            line = f"🔎 Searched {data.get('source')} for “{short(data.get('query', ''), 60)}”: kept {len(kept)}"
            line += f" of {len(kept) + len(dropped)} passages (graded by {data.get('graded_by')})"
            return line + (f": {', '.join(kept)}" if kept else " → will rewrite the query")
        if msg.name == "read_rera_section":
            return f"📜 Read {data.get('title', 'a RERA section')}" if isinstance(data, dict) else None
        if msg.name == "get_my_booking":
            return "📄 Looked up the booking record"
        if msg.name == "remember":
            return f"💾 {msg.content}"
    if node == "answer_check":
        return "↩️ " + str(msg.content).removeprefix(CHECK_PREFIX).strip().replace(
            "you cited", "Answer check: you cited"
        )
    return None


load_secrets_into_env()
settings = settings_fingerprint()
graph, choice = load_agent(settings)
retriever = load_retriever(settings[0])
memory = load_memory()
buyers = load_buyers()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader("Who's chatting?")
    options = {f"{b['name']} · flat {b['unit']}": key for key, b in buyers.items()} | {"Guest (no booking)": "guest"}
    buyer_id = options[st.selectbox("Demo buyer", list(options), label_visibility="collapsed")]

    st.subheader("What I remember about you")
    facts = memory.list(buyer_id)
    if not facts:
        st.caption("Nothing yet. Tell me something lasting, e.g. *“I'm taking a home loan.”*")
    for f in facts:
        st.caption(f"• {f['fact']}")
        if st.button("Forget this", key=f"forget-{f['id']}", type="tertiary", icon=":material/delete:"):
            memory.delete(buyer_id, f["id"])
            st.rerun()
    if facts and st.button("Forget everything about me"):
        memory.clear(buyer_id)
        st.rerun()

    st.subheader("How it works")
    st.graphviz_chart(
        """digraph { rankdir=LR; node [shape=box, style=rounded, fontname=Helvetica, fontsize=10];
        START [shape=circle, label=""]; END [shape=doublecircle, label=""];
        START -> agent; agent -> tools [label="search / remember"]; tools -> agent [label="graded passages"];
        agent -> check [label="answer"]; check -> agent [label="fix citations", style=dashed];
        check -> END [label="ok"]; check [label="answer check"]; }"""
    )
    n_project = sum(p.source == "project" for p in retriever.passages)
    n_rera = sum(p.source == "rera" for p in retriever.passages)
    st.caption(f"Knowledge base: {n_project} project passages · {n_rera} RERA Act passages · search: {retriever.mode}.")
    st.caption(f"Model: {choice.label}")
    st.markdown(f"[Source on GitHub]({GITHUB}) · [Week 1 agent]({WEEK1}) · [Pallav Kumar]({LINKEDIN})")
    st.caption("Riverside Heights, its developer and buyers are fictional. Not legal advice.")

# ---------------------------------------------------------------- main
st.title("Buyer Helpdesk Agent")
st.write(
    "Ask about **Riverside Heights** (a fictional project) or your rights under the **RERA Act, 2016**. "
    "The agent searches, checks what it found, and cites every answer."
)
if not choice.live:
    st.info(
        "Running in **demo mode**: a rule-based stand-in drives the same search, grader fallback and answer "
        "check, and quotes the best passage. Add an API key for full answers.",
        icon="ℹ️",
    )

chats = st.session_state.setdefault("chats", {})
history = chats.setdefault(buyer_id, [])


def render_extras(entry: dict) -> None:
    if entry.get("sources"):
        with st.expander(f"Sources ({len(entry['sources'])})"):
            for s in entry["sources"]:
                st.markdown(f"**[{s['id']}] {s['title']}**")
                st.caption(s["text"][:1200] + ("…" if len(s["text"]) > 1200 else ""))
    if entry.get("trace"):
        label = "How I answered" + (f" · {entry['model']}" if entry.get("model") else "")
        with st.expander(label):
            for line in entry["trace"]:
                st.markdown(line)
            for err in entry.get("fallback_errors", []):
                st.caption(f"⚠️ Fell back because this provider failed: {err}")


for entry in history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry["role"] == "assistant":
            render_extras(entry)


def ask(example: str) -> None:
    st.session_state["pending"] = example


pending = st.session_state.pop("pending", None)
if not history and not pending:
    st.caption("Try one of these:")
    cols = st.columns(len(EXAMPLES))
    for i, (col, example) in enumerate(zip(cols, EXAMPLES, strict=True)):
        col.button(example, key=f"example-{i}", use_container_width=True, on_click=ask, args=(example,))
if history and st.button("Clear this chat"):
    history.clear()
    st.rerun()

question = st.chat_input("Ask about your flat, payments, possession, refunds or RERA") or pending
if question:
    with st.chat_message("user"):
        st.markdown(question)
    convo = [HumanMessage(e["content"]) if e["role"] == "user" else AIMessage(e["content"]) for e in history]
    messages = [*convo, HumanMessage(question)]
    trace: list[str] = []
    remembered = False
    with st.chat_message("assistant"):
        status = st.status("Thinking…", expanded=True)
        try:
            for update in graph.stream(
                {"messages": messages, "buyer_id": buyer_id}, config=run_config(), stream_mode="updates"
            ):
                for node, payload in update.items():
                    for msg in payload.get("messages", []):
                        messages.append(msg)
                        if line := trace_line(node, msg):
                            trace.append(line)
                            status.markdown(line)
                        remembered |= isinstance(msg, ToolMessage) and msg.name == "remember"
            status.update(label=f"Done · {len(trace)} steps", state="complete", expanded=False)
            answer = message_text(messages[-1]).strip() or "_(no answer)_"
            retrieved = retrieved_passages(messages)
            entry = {
                "role": "assistant",
                "content": answer,
                "sources": [retrieved[i] for i in cited_ids(answer) if i in retrieved],
                "trace": trace,
                "model": answered_by(messages),
                "fallback_errors": fallback_errors(messages),
            }
            st.markdown(answer)
            render_extras(entry)
            history.extend([{"role": "user", "content": question}, entry])
        except Exception as exc:  # surface model/API errors in the UI
            status.update(label="Failed", state="error")
            st.error(f"The model call failed.\n\n{type(exc).__name__}: {exc}")
    if remembered:
        st.rerun()  # refresh the memory panel in the sidebar
