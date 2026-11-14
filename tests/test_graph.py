"""The graph end to end with scripted models (no network)."""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.demo_model import ScriptedHelpdeskModel
from agent.graph import (
    CHECK_PREFIX,
    MAX_CHECKS,
    answer_problems,
    build_graph,
    cited_ids,
    fallback_errors,
    make_grader,
    message_text,
    run_config,
)
from agent.memory import MemoryStore


def call(name, args, id_):
    return {"name": name, "args": args, "id": id_, "type": "tool_call"}


class Replay(GenericFakeChatModel):
    """Replays fixed replies and accepts tools, standing in for a live model."""

    def bind_tools(self, tools, **kwargs):
        return self


class Failing(ScriptedHelpdeskModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("401 - invalid key")


def run(graph, question, buyer="guest"):
    return graph.invoke({"messages": [HumanMessage(question)], "buyer_id": buyer}, config=run_config())["messages"]


def test_cited_ids_parses_groups():
    assert cited_ids("Refund in 45 days [cancellation_refund-1, rera-18]. See [booking].") == [
        "cancellation_refund-1",
        "rera-18",
        "booking",
    ]


def test_demo_answers_with_a_citation(retriever, tmp_path):
    graph = build_graph(ScriptedHelpdeskModel(), retriever, MemoryStore(tmp_path / "m.db"), grader_models=[])
    messages = run(graph, "Can I keep a dog?")
    answer = message_text(messages[-1])
    assert "[maintenance_society-6]" in answer and "leash" in answer
    assert answer_problems(messages) == []


def test_booking_record_and_memory(retriever, tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    graph = build_graph(ScriptedHelpdeskModel(), retriever, store, grader_models=[])
    answer = message_text(run(graph, "What are the maintenance charges for my flat?", "ananya-sen")[-1])
    assert "A-101" in answer and "[booking]" in answer
    run(graph, "Remember that I am taking a home loan", "ananya-sen")
    assert [f["fact"] for f in store.list("ananya-sen")] == ["I am taking a home loan"]


def test_llm_grader_drops_irrelevant_passages(retriever):
    hits = retriever.search("maintenance charges", k=6)
    grader = Replay(messages=iter([AIMessage(content='["maintenance_society-1"]')]))
    keep, by = make_grader([grader])("How much is maintenance?", "maintenance charges", hits)
    assert keep == ["maintenance_society-1"] and by == "llm"


def test_grader_falls_back_to_keywords_when_models_fail(retriever):
    hits = retriever.search("maintenance charges", k=6)
    keep, by = make_grader([Failing()])("How much is maintenance?", "maintenance charges", hits)
    assert by == "keyword overlap" and "maintenance_society-1" in keep


def test_empty_grade_tells_the_agent_to_rewrite(retriever, tmp_path):
    replies = iter(
        [
            AIMessage(content="", tool_calls=[call("search_documents", {"query": "xyzzy plugh"}, "c1")]),
            AIMessage(content="I couldn't find this in the documents."),
        ]
    )
    graph = build_graph(Replay(messages=replies), retriever, MemoryStore(tmp_path / "m.db"), grader_models=[])
    messages = run(graph, "xyzzy plugh?")
    tool = next(m for m in messages if isinstance(m, ToolMessage))
    assert "Rewrite the query" in tool.content


def test_answer_check_rejects_invented_citations(retriever, tmp_path):
    replies = iter(
        [
            AIMessage(content="", tool_calls=[call("search_documents", {"query": "pets dog"}, "c1")]),
            AIMessage(content="Dogs are allowed on a leash [maintenance_society-6] [rera-99]."),
            AIMessage(content="Dogs are allowed on a leash [maintenance_society-6]."),
        ]
    )
    graph = build_graph(Replay(messages=replies), retriever, MemoryStore(tmp_path / "m.db"), grader_models=[])
    messages = run(graph, "Can I keep a dog?")
    checks = [m for m in messages if isinstance(m, HumanMessage) and m.content.startswith(CHECK_PREFIX)]
    assert len(checks) == 1 and "rera-99" in checks[0].content
    assert message_text(messages[-1]) == "Dogs are allowed on a leash [maintenance_society-6]."


def test_answer_check_requires_a_search_for_facts_and_gives_up_after_the_limit(retriever, tmp_path):
    replies = iter([AIMessage(content="Maintenance is ₹ 5 per sq ft.") for _ in range(MAX_CHECKS + 1)])
    graph = build_graph(Replay(messages=replies), retriever, MemoryStore(tmp_path / "m.db"), grader_models=[])
    messages = run(graph, "What is maintenance?")
    checks = [m for m in messages if isinstance(m, HumanMessage) and m.content.startswith(CHECK_PREFIX)]
    assert len(checks) == MAX_CHECKS and "search the documents" in checks[0].content


def test_small_talk_passes_the_check(retriever, tmp_path):
    graph = build_graph(
        Replay(messages=iter([AIMessage(content="Hello! How can I help?")])),
        retriever,
        MemoryStore(tmp_path / "m.db"),
        grader_models=[],
    )
    assert message_text(run(graph, "hi")[-1]) == "Hello! How can I help?"


def test_falls_back_and_records_why(retriever, tmp_path):
    graph = build_graph(
        Failing(), retriever, MemoryStore(tmp_path / "m.db"), [ScriptedHelpdeskModel()], grader_models=[]
    )
    messages = run(graph, "Can I keep a dog?")
    assert "[maintenance_society-6]" in message_text(messages[-1])
    assert any("401 - invalid key" in e for e in fallback_errors(messages))
