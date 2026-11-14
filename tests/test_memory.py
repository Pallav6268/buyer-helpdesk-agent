from agent.memory import MAX_FACTS, MemoryStore


def test_add_list_delete_clear(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    assert store.add("b1", "Taking a home loan") == "Remembered: Taking a home loan"
    assert store.add("b1", "taking a HOME loan") == "Already remembered."
    store.add("b2", "Owns a dog")
    facts = store.list("b1")
    assert [f["fact"] for f in facts] == ["Taking a home loan"]
    store.delete("b1", facts[0]["id"])
    assert store.list("b1") == [] and len(store.list("b2")) == 1
    store.clear("b2")
    assert store.list("b2") == []


def test_never_stores_identifiers(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    assert store.add("b1", "My PAN is ABCDE1234F").startswith("Not saved")
    assert store.add("b1", "Account 123456789012 for refunds").startswith("Not saved")
    assert store.list("b1") == []


def test_keeps_only_the_latest_facts(tmp_path):
    store = MemoryStore(tmp_path / "m.db")
    for i in range(MAX_FACTS + 5):
        store.add("b1", f"fact number {i}")
    facts = store.list("b1")
    assert len(facts) == MAX_FACTS and facts[-1]["fact"] == f"fact number {MAX_FACTS + 4}"
