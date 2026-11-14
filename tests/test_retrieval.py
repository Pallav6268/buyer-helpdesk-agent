import json

from agent.documents import DATA_DIR, Passage, load_all, load_project_docs, load_rera, rera_section_text, split_section
from agent.retrieval import corpus_fingerprint, load_vectors, tokenize


def test_project_docs_become_citable_passages():
    passages = load_project_docs()
    ids = [p.id for p in passages]
    assert len(ids) == len(set(ids)) and len(ids) >= 35
    assert "payment_plan-3" in ids
    assert next(p for p in passages if p.id == "payment_plan-3").title == "Payment Plan › Interest on late payment"


def test_tokenize_stems_word_forms():
    assert tokenize("Cancels") == tokenize("cancellation") == ["cancel"]
    assert "the" not in tokenize("the refund")


def test_keyword_search_finds_the_right_policy(retriever):
    assert retriever.search("maintenance charges per sq ft", k=1)[0].id == "maintenance_society-1"
    assert retriever.search("transfer fee", source="project", k=1)[0].id == "cancellation_refund-5"


def test_source_filter(retriever):
    assert all(p.source == "project" for p in retriever.search("refund", source="project"))


def test_split_section_keeps_subsections_together():
    text = "(1) " + "a" * 900 + "\n(2) " + "b" * 900 + "\n(3) short"
    chunks = split_section(text, limit=1000)
    assert len(chunks) == 2 and chunks[0].startswith("(1)") and chunks[1].startswith("(2)")


def test_rera_act_has_all_92_sections():
    sections = json.loads((DATA_DIR / "rera" / "sections.json").read_text(encoding="utf-8"))
    assert [s["section"] for s in sections] == [str(n) for n in range(1, 93)]
    s13 = next(s for s in sections if s["section"] == "13")
    assert s13["title"].startswith("No deposit or advance to be taken by promoter")
    assert "ten per cent." in s13["text"]


def test_long_sections_are_split_into_readable_passages():
    passages = load_rera()
    assert max(len(p.text) for p in passages) < 2500
    assert sum(p.id.startswith("rera-2-part") for p in passages) > 5


def test_read_full_section():
    sec = rera_section_text("Section 18")
    assert sec["title"] == "Return of amount and compensation" and "withdraw from the project" in sec["text"]


def test_search_finds_the_right_rera_sections(retriever):
    assert (
        retriever.search("advance payment before agreement for sale ten per cent", source="rera", k=1)[0].id
        == "rera-13"
    )
    assert "rera-31" in [p.id for p in retriever.search("filing of complaints with the Authority", source="rera", k=3)]


def test_precomputed_embeddings_match_the_documents():
    # Fails if a document changed without re-running scripts/build_embeddings.py.
    passages = load_all()
    vectors = load_vectors(passages)
    assert vectors is not None, "data/embeddings.npz is stale: run python scripts/build_embeddings.py"
    assert vectors.shape == (len(passages), 384)


def test_fingerprint_changes_with_the_text():
    passages = load_project_docs()
    edited = [Passage(p.id, p.source, p.title, p.text + " edited") if i == 0 else p for i, p in enumerate(passages)]
    assert corpus_fingerprint(passages) != corpus_fingerprint(edited)
