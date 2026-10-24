from __future__ import annotations

import io
import json
import struct
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from whatsapp_total_recall.chunking import chunk_into_episodes
from whatsapp_total_recall.cli import main as cli_main
from whatsapp_total_recall.embeddings import (
    DeterministicMockEmbeddingProvider,
    OllamaEmbeddingProvider,
    get_embedding_provider,
    l2_normalize,
)
from whatsapp_total_recall.ingestion import format_iso_timestamp
from whatsapp_total_recall.search import SearchIndex, cosine_distance
from whatsapp_total_recall.types import NormalizedMessage


def make_msg(
    pk: int,
    chat_jid: str,
    sender_name: str,
    text: str,
    timestamp: float,
    parent_pk: int | None = None,
    is_from_me: bool = False,
    chat_session_id: int = 1,
) -> NormalizedMessage:
    return NormalizedMessage(
        pk=pk,
        chat_session_id=chat_session_id,
        chat_jid=chat_jid,
        stanza_id=f"STANZA-{pk}",
        is_from_me=is_from_me,
        sender_name=sender_name,
        sender_jid="you" if is_from_me else f"{sender_name.lower()}@s.whatsapp.net",
        apple_date=timestamp - 978_307_200,
        unix_timestamp=timestamp,
        iso_date=format_iso_timestamp(timestamp),
        text=text,
        parent_pk=parent_pk,
    )


@pytest.fixture
def mock_embedder() -> DeterministicMockEmbeddingProvider:
    return DeterministicMockEmbeddingProvider(dim=64)


# ==============================================================================
# 1. Embedding Providers & Normalization Tests
# ==============================================================================


def test_l2_normalize():
    vec = [3.0, 4.0]
    norm = l2_normalize(vec)
    assert pytest.approx(norm[0]) == 0.6
    assert pytest.approx(norm[1]) == 0.8
    assert pytest.approx(sum(x * x for x in norm)) == 1.0

    # Zero vector
    zero = [0.0, 0.0]
    assert l2_normalize(zero) == zero


def test_deterministic_mock_embedder(mock_embedder: DeterministicMockEmbeddingProvider):
    assert mock_embedder.dimension == 64

    # Empty text
    empty_vec = mock_embedder.embed_text("")
    assert empty_vec == [0.0] * 64

    # Consistency
    v1 = mock_embedder.embed_text("sqlite-vec SIMD hybrid search")
    v2 = mock_embedder.embed_text("sqlite-vec SIMD hybrid search")
    assert v1 == v2
    assert pytest.approx(sum(x * x for x in v1)) == 1.0

    # Batch
    batch = mock_embedder.embed_batch(["text one", "text two"])
    assert len(batch) == 2
    assert len(batch[0]) == 64
    assert len(batch[1]) == 64


def test_ollama_provider_embed_api_success():
    provider = OllamaEmbeddingProvider(
        host="http://127.0.0.1:11434", model="nomic-embed-text", dim=4
    )

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(
        {"embeddings": [[0.5, 0.5, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0]]}
    ).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = provider.embed_batch(["hello", "world"])
        assert len(res) == 2
        assert len(res[0]) == 4
        assert pytest.approx(sum(x * x for x in res[0])) == 1.0


def test_ollama_provider_unreachable():
    provider = OllamaEmbeddingProvider(host="http://127.0.0.1:99999", timeout=0.1)
    with pytest.raises(RuntimeError):
        provider.embed_text("test")


def test_get_embedding_provider_factory():
    p_mock = get_embedding_provider("mock", dim=128)
    assert isinstance(p_mock, DeterministicMockEmbeddingProvider)
    assert p_mock.dimension == 128

    p_ollama = get_embedding_provider("ollama", dim=768)
    assert isinstance(p_ollama, OllamaEmbeddingProvider)
    assert p_ollama.dimension == 768

    with pytest.raises(ValueError, match="Unknown embedding provider"):
        get_embedding_provider("nonexistent_provider")


# ==============================================================================
# 2. Vector Storage & Cosine Distance Tests
# ==============================================================================


def test_cosine_distance_calculation():
    # Identical vectors -> distance 0.0
    v1 = [1.0, 0.0, 0.0]
    b1 = struct.pack("3f", *v1)
    assert pytest.approx(cosine_distance(b1, b1)) == 0.0

    # Orthogonal vectors -> distance 1.0
    v2 = [0.0, 1.0, 0.0]
    b2 = struct.pack("3f", *v2)
    assert pytest.approx(cosine_distance(b1, b2)) == 1.0

    # Empty or mismatched
    assert cosine_distance(b"", b"") == 1.0
    assert cosine_distance(b1, b"") == 1.0


def test_vector_indexing_and_similarity(
    mock_embedder: DeterministicMockEmbeddingProvider,
):
    t0 = 1792659600.0
    group_jid = "120363041234567890@g.us"
    m1 = make_msg(1, group_jid, "Alice", "Zero Docker local vector search.", t0)
    m2 = make_msg(
        2, group_jid, "Bob", "Using SIMD instructions on Apple Silicon.", t0 + 60
    )

    episodes = chunk_into_episodes([m1, m2], chat_names={group_jid: "Architecture WG"})

    with SearchIndex(":memory:", dimension=64, embedder=mock_embedder) as index:
        indexed = index.index_episodes(episodes, embedder=mock_embedder)
        assert indexed == 1

        # Check vector table has 1 record
        cur = index.con.cursor()
        cur.execute("SELECT count(*) FROM episode_vectors")
        assert cur.fetchone()[0] == 1

        # Vector search
        res = index.search_episodes_vector("Zero Docker local vector", limit=5)
        assert len(res) == 1
        assert res[0].episode.episode_id == episodes[0].episode_id
        assert res[0].vec_rank == 1
        assert res[0].vector_distance is not None

        # Compare with unrelated query
        res_unrelated = index.search_episodes_vector("baking chocolate cake", limit=5)
        assert res[0].vector_distance < res_unrelated[0].vector_distance


# ==============================================================================
# 3. Unified SQL Reciprocal Rank Fusion (RRF) Tests
# ==============================================================================


def test_rrf_hybrid_exact_keyword_and_semantic_paraphrase(
    mock_embedder: DeterministicMockEmbeddingProvider,
):
    """Verify that RRF combines lexical and semantic matches according to Cormack formula."""
    t0 = 1792659600.0
    chat_a = "120363041111111111@g.us"
    chat_b = "120363042222222222@g.us"
    chat_c = "120363043333333333@g.us"

    # Ep 1: Technical jargon (lexical hit for '0xDEAD10CC')
    m1 = [
        make_msg(
            1, chat_a, "Alice", "Encountered crash code 0xDEAD10CC in coredata.", t0
        ),
        make_msg(
            2,
            chat_a,
            "Bob",
            "Snapshot isolation mode=ro&immutable=1 solves 0xDEAD10CC.",
            t0 + 10,
        ),
    ]

    # Ep 2: Semantic paraphrase (semantic hit for 'offline embedded AI retrieval')
    m2 = [
        make_msg(
            3,
            chat_b,
            "Charlie",
            "We need edge-native conversational intelligence.",
            t0 + 3600,
        ),
        make_msg(
            4,
            chat_b,
            "Dave",
            "Local single-file database without cloud dependencies.",
            t0 + 3610,
        ),
    ]

    # Ep 3: Irrelevant
    m3 = [
        make_msg(5, chat_c, "Eve", "Lunch plans at noon today.", t0 + 7200),
        make_msg(6, chat_c, "Frank", "Tacos sound good.", t0 + 7210),
    ]

    all_episodes = (
        chunk_into_episodes(m1, {chat_a: "Chat A"})
        + chunk_into_episodes(m2, {chat_b: "Chat B"})
        + chunk_into_episodes(m3, {chat_c: "Chat C"})
    )
    assert len(all_episodes) == 3

    with SearchIndex(":memory:", dimension=64, embedder=mock_embedder) as index:
        index.index_episodes(all_episodes, embedder=mock_embedder)

        # 1. Exact technical keyword query: Ep 1 should be ranked #1
        res_lex = index.search_hybrid("0xDEAD10CC", limit=3, embedder=mock_embedder)
        assert len(res_lex) >= 1
        assert res_lex[0].episode.chat_name == "Chat A"
        assert res_lex[0].fts_rank == 1

        # 2. Semantic query: Ep 2 should rank highly
        res_sem = index.search_hybrid(
            "edge-native conversational intelligence", limit=3, embedder=mock_embedder
        )
        assert len(res_sem) >= 1
        assert any(r.episode.chat_name == "Chat B" for r in res_sem)

        # 3. Score calibration check:
        # If an item has both FTS rank and Vec rank: RRF = 1/(60+r_fts) + 1/(60+r_vec)
        # If an item only has one: RRF = 1/(60+r)
        k_rrf = 60.0
        for item in res_lex:
            expected_fts = (
                1.0 / (k_rrf + item.fts_rank) if item.fts_rank is not None else 0.0
            )
            expected_vec = (
                1.0 / (k_rrf + item.vec_rank) if item.vec_rank is not None else 0.0
            )
            assert (
                pytest.approx(item.rrf_score, rel=1e-4) == expected_fts + expected_vec
            )


def test_rrf_dual_hit_beats_single_hit(
    mock_embedder: DeterministicMockEmbeddingProvider,
):
    """An episode matching both lexical AND semantic criteria should score higher than a single-mode match."""
    t0 = 1792659600.0
    group_jid = "120363041234567890@g.us"

    # Ep 1: Matches exact keyword AND semantic meaning
    m1 = [
        make_msg(
            1,
            group_jid,
            "Alice",
            "Reciprocal Rank Fusion merges BM25 and vector similarity.",
            t0,
        ),
        make_msg(
            2,
            group_jid,
            "Bob",
            "RRF formulation: 1/(60+r_fts) + 1/(60+r_vec).",
            t0 + 10,
        ),
    ]

    # Ep 2: Matches only keyword 'Fusion' with unrelated topic
    m2 = [
        make_msg(
            3, group_jid, "Charlie", "Nuclear fusion energy reactor updates.", t0 + 3600
        ),
        make_msg(
            4, group_jid, "Dave", "Hydrogen plasma confinement experiment.", t0 + 3610
        ),
    ]

    episodes = chunk_into_episodes(m1 + m2, {group_jid: "Physics & CS Group"})
    assert len(episodes) == 2

    with SearchIndex(":memory:", dimension=64, embedder=mock_embedder) as index:
        index.index_episodes(episodes, embedder=mock_embedder)

        # Query combines keyword 'Fusion' and semantic concept 'vector similarity'
        results = index.search_hybrid(
            "Fusion vector similarity", limit=2, embedder=mock_embedder
        )
        assert len(results) == 2
        # Ep 1 must rank first with higher RRF score
        assert results[0].episode.episode_id == episodes[0].episode_id
        assert results[0].rrf_score > results[1].rrf_score


# ==============================================================================
# 4. Cryptographic Purge Tests (PRAGMA secure_delete)
# ==============================================================================


def test_search_index_purge_message(mock_embedder: DeterministicMockEmbeddingProvider):
    """Verify that purge() removes the message, updates parent episodes, and re-indexes vectors."""
    t0 = 1792659600.0
    group_jid = "120363041234567890@g.us"
    m1 = make_msg(
        1, group_jid, "Alice", "Sensitive confidential financial statement.", t0
    )
    m2 = make_msg(2, group_jid, "Bob", "Acknowledged receipt of statement.", t0 + 60)

    episodes = chunk_into_episodes([m1, m2], {group_jid: "Finance"})

    with SearchIndex(":memory:", dimension=64, embedder=mock_embedder) as index:
        index.index_messages([m1, m2])
        index.index_episodes(
            episodes, index_constituent_messages=False, embedder=mock_embedder
        )

        assert index.get_message(1) is not None
        assert (
            "confidential financial"
            in index.get_episode(episodes[0].episode_id).content
        )

        # Purge message 1
        report = index.purge(1, embedder=mock_embedder)
        assert report.message_pk == 1
        assert report.raw_messages_purged == 1
        assert report.episodes_updated == 1
        assert report.episodes_deleted == 0
        assert report.secure_delete_executed is True

        # Assert message 1 is gone from messages table & FTS
        assert index.get_message(1) is None
        assert len(index.search("confidential financial")) == 0

        # Assert episode 0 was updated: no longer contains confidential text
        updated_ep = index.get_episode(episodes[0].episode_id)
        assert updated_ep is not None
        assert "confidential financial" not in updated_ep.content
        assert "Acknowledged receipt" in updated_ep.content


def test_search_index_purge_last_message_deletes_episode(
    mock_embedder: DeterministicMockEmbeddingProvider,
):
    """When the only message in an episode is purged, the episode is deleted entirely."""
    t0 = 1792659600.0
    group_jid = "120363041234567890@g.us"
    m1 = make_msg(1, group_jid, "Alice", "Sole secret message in this burst.", t0)

    episodes = chunk_into_episodes([m1], {group_jid: "Secret"})

    with SearchIndex(":memory:", dimension=64, embedder=mock_embedder) as index:
        index.index_messages([m1])
        index.index_episodes(
            episodes, index_constituent_messages=False, embedder=mock_embedder
        )

        ep_id = episodes[0].episode_id
        assert index.get_episode(ep_id) is not None

        report = index.purge(1, embedder=mock_embedder)
        assert report.episodes_deleted == 1
        assert index.get_episode(ep_id) is None

        # Check vectors table is also cleared
        cur = index.con.cursor()
        cur.execute(
            "SELECT count(*) FROM episode_vectors WHERE episode_id = ?", (ep_id,)
        )
        assert cur.fetchone()[0] == 0


# ==============================================================================
# 5. CLI Search Command End-to-End Tests
# ==============================================================================


def test_cli_search_hybrid_and_lexical(tmp_path: Path):
    """Verify CLI search command returns top-k ranked episodes via RRF and lexical search."""
    db_file = tmp_path / "test_cli_search.db"
    t0 = 1792659600.0
    chat_jid = "14155550199@s.whatsapp.net"

    m1 = make_msg(
        1, chat_jid, "Alice", "Meeting notes regarding Project Titan roadmap.", t0
    )
    m2 = make_msg(2, chat_jid, "Bob", "We will ship version 1.0 before Q4.", t0 + 30)

    episodes = chunk_into_episodes([m1, m2], {chat_jid: "Alice & Bob"})

    embedder = DeterministicMockEmbeddingProvider(dim=64)
    with SearchIndex(db_file, dimension=64, embedder=embedder) as index:
        index.index_messages([m1, m2])
        index.index_episodes(
            episodes, index_constituent_messages=False, embedder=embedder
        )

    # 1. Natural language hybrid search across episodes
    buf_hybrid = io.StringIO()
    with redirect_stdout(buf_hybrid):
        exit_code = cli_main(
            [
                "search",
                "Project Titan roadmap",
                "--db",
                str(db_file),
                "--embedder",
                "mock",
            ]
        )
    assert exit_code == 0
    hybrid_out = buf_hybrid.getvalue()
    assert "Found 1 matching episodes" in hybrid_out
    assert "Project Titan roadmap" in hybrid_out
    assert "RRF Score:" in hybrid_out

    # 2. Search raw messages via --target messages
    buf_msgs = io.StringIO()
    with redirect_stdout(buf_msgs):
        exit_code_msgs = cli_main(
            ["search", "Titan", "--target", "messages", "--db", str(db_file)]
        )
    assert exit_code_msgs == 0
    msgs_out = buf_msgs.getvalue()
    assert "Found 1 matching messages" in msgs_out
    assert "Alice:" in msgs_out
    assert "Meeting notes regarding Project Titan roadmap." in msgs_out

    # 3. Search episodes via 'episodes' command with hybrid mode
    buf_ep = io.StringIO()
    with redirect_stdout(buf_ep):
        exit_code_ep = cli_main(
            [
                "episodes",
                "Titan",
                "--mode",
                "hybrid",
                "--db",
                str(db_file),
                "--embedder",
                "mock",
            ]
        )
    assert exit_code_ep == 0
    ep_out = buf_ep.getvalue()
    assert "Found 1 matching episodes" in ep_out
    assert "Project Titan roadmap" in ep_out


def test_dimension_reconciliation_384_to_768_model_switch():
    """Verify switching embedder dimensions (e.g. 384 default to 768 Ollama) does not raise sqlite3.OperationalError."""
    chat_jid = "120363041234567890@g.us"
    m1 = NormalizedMessage(
        pk=1,
        chat_session_id=1,
        chat_jid=chat_jid,
        stanza_id="s1",
        is_from_me=False,
        sender_name="Alice",
        sender_jid="alice@s.whatsapp.net",
        apple_date=700000000.0,
        unix_timestamp=1672531200.0,
        iso_date="2023-01-01 00:00:00 UTC",
        text="Switching models should not crash",
    )
    episodes = chunk_into_episodes([m1], {chat_jid: "Group"})

    # 1. Initialize index with 384 dimensions and mock embedder
    embedder_384 = DeterministicMockEmbeddingProvider(dim=384)
    with SearchIndex(":memory:", dimension=384, embedder=embedder_384) as index:
        index.index_episodes(episodes, embedder=embedder_384)
        assert index.dimension == 384

        # 2. Now switch to 768-dim embedder (e.g. Ollama nomic-embed-text)
        embedder_768 = DeterministicMockEmbeddingProvider(dim=768)
        index.index_episodes(episodes, embedder=embedder_768)
        assert index.dimension == 768

        # 3. Vector search with 768-dim query succeeds
        scored = index.search_episodes_vector("models", limit=5, embedder=embedder_768)
        assert len(scored) == 1


def is_ollama_available() -> bool:
    import urllib.request

    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:11434/api/tags", timeout=0.5
        ) as resp:
            return resp.status == 200
    except Exception:
        return False


@pytest.mark.skipif(not is_ollama_available(), reason="Local Ollama not reachable")
def test_live_on_device_ollama_nomic_embed():
    """Live verification against on-device Ollama nomic-embed-text model."""
    provider = OllamaEmbeddingProvider(model="nomic-embed-text")
    vec = provider.embed_text("Privacy-first WhatsApp search engine")
    assert len(vec) == 768
    assert pytest.approx(sum(x * x for x in vec), rel=1e-3) == 1.0
