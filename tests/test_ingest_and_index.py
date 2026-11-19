import numpy as np

from verity.index.bm25 import BM25, tokenize
from verity.index.embeddings import HashingEmbedder
from verity.index.store import KnowledgeIndex
from verity.ingest.chunking import _fixed, _recursive, chunk_document
from verity.ingest.loaders import Document


def test_loader_ids_titles_and_types(docs):
    ids = {d.id for d in docs}
    assert "contracts/kaveri-foods" in ids and "policies/leave-policy" in ids
    kaveri = next(d for d in docs if d.id == "contracts/kaveri-foods")
    assert kaveri.title == "Contract: Kaveri Foods"
    assert kaveri.metadata["doc_type"] == "contracts"


def test_fixed_chunks_respect_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = _fixed(text, 200, 50)
    assert all(len(c) <= 200 for c in chunks)
    assert chunks[0].split()[-1] in chunks[1]  # consecutive windows overlap
    assert chunks[-1].split()[-1] == "word399"


def test_recursive_prefers_paragraph_boundaries():
    text = "Para one is short.\n\nPara two is also short.\n\n" + "x " * 300
    chunks = _recursive(text, 120, 0)
    assert chunks[0].startswith("Para one") and "Para two" in chunks[0]
    assert all(len(c) <= 120 for c in chunks)


def test_markdown_chunks_carry_contextual_headers():
    doc = Document("d/x", "Contract: Acme", "# Contract: Acme\n\n## Summary\n- Value: 10\n\n## Notes\nGood client.\n")
    chunks = chunk_document(doc, "markdown")
    assert [c.section for c in chunks] == ["Contract: Acme > Summary", "Contract: Acme > Notes"]
    assert chunks[1].text.startswith("Contract: Acme > Notes\n")  # header is embedded with the text


def test_tokenizer_keeps_numbers_and_folds_plurals():
    toks = tokenize("Annual value USD 78,000 for 99.9% uptime; contracts")
    assert "78000" in toks and "99.9" in toks and "contract" in toks and "for" not in toks


def test_bm25_ranks_exact_match_first():
    texts = ["the cat sat", "dog runs in the park", "a cat and a dog", "kaveri foods contract value 78,000"]
    bm = BM25().fit(texts)
    assert bm.search("78,000", 1)[0][0] == 3
    assert bm.search("cat dog", 1)[0][0] == 2


def test_index_roundtrip(tmp_path, docs):
    idx = KnowledgeIndex.build(docs[:5], strategy="recursive", embedder=HashingEmbedder())
    idx.save(tmp_path)
    again = KnowledgeIndex.load(tmp_path)
    assert len(again.chunks) == len(idx.chunks)
    assert np.allclose(again.vectors, idx.vectors)
    q = idx.chunks[0].text[:80]
    assert again.bm25_search(q, 1)[0][0] == idx.bm25_search(q, 1)[0][0]


def test_lsa_dense_search_finds_paraphrase(index):
    # no word overlap with "Delhivery" bio wording except the employer name
    top = index.dense_search("colleague who used to work at Delhivery", 5)
    assert any(index.chunks[i].doc_id == "employees/priya-nair" for i, _ in top)
