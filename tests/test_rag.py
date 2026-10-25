"""Offline tests: no Gemini API calls are made (embeddings and generation are mocked)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ask  # noqa: E402
import ingest  # noqa: E402
import rag  # noqa: E402


# ---------- chunking ----------

def test_chunks_overlap_and_cover_all_words():
    words = [f"w{i}" for i in range(200)]
    chunks = rag.chunk_text(" ".join(words), size=80, overlap=20)
    assert [len(c.split()) for c in chunks] == [80, 80, 80]
    # last 20 words of one chunk are the first 20 of the next
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]
    # every word appears somewhere, and the final word is included
    assert set(" ".join(chunks).split()) == set(words)


def test_short_text_is_one_chunk_and_empty_is_none():
    assert rag.chunk_text("just a few words") == ["just a few words"]
    assert rag.chunk_text("") == []


def test_bad_overlap_rejected():
    with pytest.raises(ValueError):
        rag.chunk_text("a b c", size=10, overlap=10)


# ---------- deterministic IDs ----------

def test_chunk_ids_are_deterministic():
    first = [r["id"] for r in rag.load_chunks()]
    second = [r["id"] for r in rag.load_chunks()]
    assert first == second
    assert len(first) == len(set(first)), "IDs must be unique"
    assert rag.chunk_id("a.md", 0, "text") == rag.chunk_id("a.md", 0, "text")
    assert rag.chunk_id("a.md", 0, "text") != rag.chunk_id("a.md", 0, "changed")


def test_sample_docs_have_titles_and_fictional_label():
    files = list(rag.DATA_DIR.glob("*.md"))
    assert len(files) >= 5
    for f in files:
        text = f.read_text()
        assert text.startswith("# ")
        assert "FICTIONAL DEMO POLICY" in text


# ---------- grounded prompt and citations ----------

CHUNKS = [
    {"text": "Permits cost $45.", "source": "parking_permits.md", "title": "Parking", "distance": 0.2},
    {"text": "Bills are bimonthly.", "source": "water_bills.md", "title": "Water", "distance": 0.4},
]


def test_prompt_contains_only_retrieved_excerpts_and_rules():
    prompt = rag.build_prompt("How much is a permit?", CHUNKS)
    assert "[parking_permits.md]" in prompt and "[water_bills.md]" in prompt
    assert "Permits cost $45." in prompt
    assert rag.NOT_FOUND in prompt
    assert "ONLY" in prompt
    assert "recycling.md" not in prompt


def test_check_citations_removes_unretrieved_sources():
    answer = "Permits cost $45 [parking_permits.md]. Glass is not accepted [recycling.md]."
    cleaned, valid, invalid = rag.check_citations(answer, {"parking_permits.md", "water_bills.md"})
    assert valid == ["parking_permits.md"]
    assert invalid == ["recycling.md"]
    assert "[recycling.md]" not in cleaned
    assert "[parking_permits.md]" in cleaned


def test_not_found_detection():
    assert rag.is_not_found(rag.NOT_FOUND)
    assert not rag.is_not_found("Permits cost $45 [parking_permits.md].")


# ---------- end-to-end with a real local Chroma and mocked Gemini ----------

def fake_embed(client, texts, task_type):
    """Tiny bag-of-keywords vectors so similar texts are near each other."""
    keywords = ["parking", "permit", "recycl", "water", "noise", "records", "building", "glass"]
    return [[float(t.lower().count(k)) + 0.01 for k in keywords] for t in texts]


@pytest.fixture
def offline(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "CHROMA_DIR", tmp_path / "chroma_db")
    monkeypatch.setattr(rag, "get_client", lambda: object())
    monkeypatch.setattr(rag, "embed", fake_embed)
    monkeypatch.setattr(rag, "gen_model", lambda: "fake-model")


def test_reingest_does_not_duplicate(offline):
    expected = len(rag.load_chunks())
    assert ingest.main() == 0
    assert rag.get_collection().count() == expected
    assert ingest.main() == 0
    assert rag.get_collection().count() == expected


def test_ask_before_ingest_gives_clear_error(offline, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ask.py", "anything"])
    assert ask.main() == 1
    assert "Run `python ingest.py` first" in capsys.readouterr().err


def test_ask_prints_answer_citations_and_strips_fake_ones(offline, monkeypatch, capsys):
    ingest.main()
    monkeypatch.setattr(rag, "generate",
                        lambda client, prompt: "You can apply online [parking_permits.md] [made_up.md].")
    monkeypatch.setattr(sys, "argv", ["ask.py", "Can I apply for a parking permit online?"])
    assert ask.main() == 0
    out = capsys.readouterr().out
    assert "Cited sources: parking_permits.md" in out
    assert "made_up.md" in out and "Removed citations" in out
    assert "Total latency" in out


def test_missing_api_key_message(monkeypatch, tmp_path):
    monkeypatch.setattr(rag, "ROOT", tmp_path)  # no .env here
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(rag.RAGError, match="GEMINI_API_KEY is not set"):
        rag.get_client()
