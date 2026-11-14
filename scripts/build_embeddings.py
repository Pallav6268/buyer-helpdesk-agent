"""Precompute passage embeddings into data/embeddings.npz so the app only embeds queries at runtime.

Run after changing any document:  python scripts/build_embeddings.py
The app checks the stored fingerprint and recomputes in memory if the file is stale.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.documents import load_all  # noqa: E402
from agent.retrieval import EMBED_BATCH, EMBED_MODEL, VECTORS_FILE, corpus_fingerprint, passage_text  # noqa: E402


def main() -> None:
    from fastembed import TextEmbedding

    passages = load_all()
    model = TextEmbedding(EMBED_MODEL)
    vectors = np.array(list(model.embed([passage_text(p) for p in passages], batch_size=EMBED_BATCH)), dtype=np.float32)
    np.savez_compressed(
        VECTORS_FILE,
        vectors=vectors,
        ids=np.array([p.id for p in passages]),
        model=np.array(EMBED_MODEL),
        fingerprint=np.array(corpus_fingerprint(passages)),
    )
    print(f"wrote {vectors.shape} to {VECTORS_FILE}")


if __name__ == "__main__":
    main()
