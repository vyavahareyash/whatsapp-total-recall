from __future__ import annotations

import io
import sqlite3
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from whatsapp_total_recall.chunking import (
    chunk_into_episodes,
)
from whatsapp_total_recall.cli import main as cli_main
from whatsapp_total_recall.ingestion import (
    APPLE_EPOCH_OFFSET,
    IngestionEngine,
    format_iso_timestamp,
)
from whatsapp_total_recall.search import SearchIndex
from whatsapp_total_recall.types import NormalizedMessage
from whatsapp_total_recall.vault import VaultConfig


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
    """Helper to quickly construct NormalizedMessage instances."""
    return NormalizedMessage(
        pk=pk,
        chat_session_id=chat_session_id,
        chat_jid=chat_jid,
        stanza_id=f"STANZA-{pk}",
        is_from_me=is_from_me,
        sender_name=sender_name,
        sender_jid="you" if is_from_me else f"{sender_name.lower()}@s.whatsapp.net",
        apple_date=timestamp - APPLE_EPOCH_OFFSET,
        unix_timestamp=timestamp,
        iso_date=format_iso_timestamp(timestamp),
        text=text,
        parent_pk=parent_pk,
    )


@pytest.fixture
def synthetic_vault_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Create synthetic WhatsApp storage with multiple 1:1 and group chats."""
    chat_storage_path = tmp_path / "ChatStorage.sqlite"
    contacts_path = tmp_path / "ContactsV2.sqlite"

    con_c = sqlite3.connect(contacts_path)
    cur_c = con_c.cursor()
    cur_c.executescript(
        """
        CREATE TABLE ZWAADDRESSBOOKCONTACT (
            Z_PK INTEGER PRIMARY KEY AUTOINCREMENT,
            ZWHATSAPPID TEXT,
            ZFULLNAME TEXT,
            ZPHONENUMBER TEXT
        );
        INSERT INTO ZWAADDRESSBOOKCONTACT (ZWHATSAPPID, ZFULLNAME, ZPHONENUMBER) VALUES
            ('14155550101@s.whatsapp.net', 'Alice Smith', '+1 415 555 0101'),
            ('14155550102@s.whatsapp.net', 'Bob Jones', '+1 415 555 0102'),
            ('14155550103@s.whatsapp.net', 'Charlie Work', '+1 415 555 0103');
        """
    )
    con_c.commit()
    con_c.close()

    con_chat = sqlite3.connect(chat_storage_path)
    cur_chat = con_chat.cursor()
    cur_chat.executescript(
        """
        CREATE TABLE ZWACHATSESSION (
            Z_PK INTEGER PRIMARY KEY AUTOINCREMENT,
            ZCONTACTJID TEXT UNIQUE,
            ZPARTNERNAME TEXT,
            ZMESSAGECOUNTER INTEGER,
            ZUNREADCOUNT INTEGER
        );
        CREATE TABLE ZWAPROFILEPUSHNAME (
            Z_PK INTEGER PRIMARY KEY AUTOINCREMENT,
            ZJID TEXT UNIQUE,
            ZPUSHNAME TEXT
        );
        CREATE TABLE ZWAGROUPMEMBER (
            Z_PK INTEGER PRIMARY KEY AUTOINCREMENT,
            ZCHATSESSION INTEGER,
            ZMEMBERJID TEXT
        );
        CREATE TABLE ZWAMESSAGE (
            Z_PK INTEGER PRIMARY KEY AUTOINCREMENT,
            ZCHATSESSION INTEGER,
            ZSTANZAID TEXT,
            ZISFROMME INTEGER,
            ZGROUPMEMBER INTEGER,
            ZPARENTMESSAGE INTEGER,
            ZMESSAGEDATE REAL,
            ZSENTDATE REAL,
            ZSORT INTEGER,
            ZTEXT TEXT,
            ZMESSAGETYPE INTEGER
        );

        -- Session 1: Engineering Group Chat
        INSERT INTO ZWACHATSESSION (Z_PK, ZCONTACTJID, ZPARTNERNAME, ZMESSAGECOUNTER, ZUNREADCOUNT)
        VALUES (1, '120363041234567890@g.us', 'Engineering Core', 10, 0);

        -- Session 2: Work 1:1 with Alice
        INSERT INTO ZWACHATSESSION (Z_PK, ZCONTACTJID, ZPARTNERNAME, ZMESSAGECOUNTER, ZUNREADCOUNT)
        VALUES (2, '14155550101@s.whatsapp.net', 'Alice Smith', 10, 0);

        -- Session 3: Personal 1:1 with Bob (Sensitive)
        INSERT INTO ZWACHATSESSION (Z_PK, ZCONTACTJID, ZPARTNERNAME, ZMESSAGECOUNTER, ZUNREADCOUNT)
        VALUES (3, '14155550102@s.whatsapp.net', 'Bob Jones', 10, 0);

        -- Group members for Session 1
        INSERT INTO ZWAGROUPMEMBER (Z_PK, ZCHATSESSION, ZMEMBERJID) VALUES
            (1, 1, '14155550101@s.whatsapp.net'),
            (2, 1, '14155550103@s.whatsapp.net');
        """
    )

    t0 = 1792659600.0 - APPLE_EPOCH_OFFSET  # 2026-10-22 09:00:00 UTC

    # Add messages to Session 1 (Group), Session 2 (Alice), Session 3 (Bob)
    msgs = [
        # Session 1: Engineering group messages
        (
            1,
            1,
            "S1-M1",
            0,
            1,
            None,
            t0,
            t0,
            1,
            "Alice: Group discussion started on RAG.",
            0,
        ),
        (
            2,
            1,
            "S1-M2",
            1,
            None,
            None,
            t0 + 60,
            t0 + 60,
            2,
            "You: sqlite-vec is lightweight.",
            0,
        ),
        # Session 2: Alice 1:1 messages
        (
            3,
            2,
            "S2-M1",
            0,
            None,
            None,
            t0,
            t0,
            3,
            "Alice: Hey, can you review the PR?",
            0,
        ),
        (
            4,
            2,
            "S2-M2",
            1,
            None,
            3,
            t0 + 300,
            t0 + 300,
            4,
            "You: Reviewing it right now.",
            0,
        ),
        # Session 3: Bob 1:1 (Sensitive, must stay filtered out by default vault)
        (
            5,
            3,
            "S3-M1",
            0,
            None,
            None,
            t0,
            t0,
            5,
            "Bob: Sensitive medical info here.",
            0,
        ),
        (
            6,
            3,
            "S3-M2",
            1,
            None,
            5,
            t0 + 120,
            t0 + 120,
            6,
            "You: Kept confidential.",
            0,
        ),
    ]

    cur_chat.executemany(
        """
        INSERT INTO ZWAMESSAGE (
            Z_PK, ZCHATSESSION, ZSTANZAID, ZISFROMME, ZGROUPMEMBER,
            ZPARENTMESSAGE, ZMESSAGEDATE, ZSENTDATE, ZSORT, ZTEXT, ZMESSAGETYPE
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        msgs,
    )
    con_chat.commit()
    con_chat.close()

    return chat_storage_path, contacts_path


# -----------------------------------------------------------------------------
# 1. Dynamic Inactivity Gap Chunking Tests
# -----------------------------------------------------------------------------


def test_group_dynamic_burst_gap_threshold():
    """Verify group chats split at > 15m (900s) inactivity gap, and stay unified <= 15m."""
    t0 = 1792659600.0
    group_jid = "120363041234567890@g.us"

    # Burst 1: 3 messages with 14-minute gap (<= 15m, should stay unified)
    m1 = make_msg(1, group_jid, "Alice", "First group message", t0)
    m2 = make_msg(2, group_jid, "Bob", "Second group message 14m later", t0 + 840)
    m3 = make_msg(3, group_jid, "Alice", "Third group message 2m later", t0 + 960)

    # Gap of 16 minutes (> 15m = 900s) triggers Episode boundary
    m4 = make_msg(
        4,
        group_jid,
        "Charlie",
        "Fourth group message after 16m silence",
        t0 + 960 + 960,
    )
    m5 = make_msg(
        5,
        group_jid,
        "You",
        "Fifth group message 1m later",
        t0 + 960 + 960 + 60,
        is_from_me=True,
    )

    episodes = chunk_into_episodes([m1, m2, m3, m4, m5])

    assert len(episodes) == 2
    # Episode 1 has messages 1, 2, 3
    assert [m.pk for m in episodes[0].messages] == [1, 2, 3]
    # Episode 2 has messages 4, 5
    assert [m.pk for m in episodes[1].messages] == [4, 5]


def test_direct_1to1_dynamic_burst_gap_threshold():
    """Verify 1:1 chats split at > 45m (2700s) inactivity gap, and stay unified <= 45m."""
    t0 = 1792659600.0
    direct_jid = "14155550101@s.whatsapp.net"

    # Burst 1: 3 messages with 40-minute gap (<= 45m, should stay unified)
    m1 = make_msg(1, direct_jid, "Alice", "Hello", t0)
    m2 = make_msg(
        2, direct_jid, "You", "Hey Alice (40m later)", t0 + 2400, is_from_me=True
    )
    m3 = make_msg(3, direct_jid, "Alice", "Quick question (2m later)", t0 + 2520)

    # Gap of 46 minutes (> 45m = 2700s) triggers Episode boundary
    m4 = make_msg(
        4, direct_jid, "Alice", "Following up after 46m gap", t0 + 2520 + 2760
    )
    m5 = make_msg(
        5,
        direct_jid,
        "You",
        "Here is the answer",
        t0 + 2520 + 2760 + 60,
        is_from_me=True,
    )

    episodes = chunk_into_episodes([m1, m2, m3, m4, m5])

    assert len(episodes) == 2
    assert [m.pk for m in episodes[0].messages] == [1, 2, 3]
    assert [m.pk for m in episodes[1].messages] == [4, 5]


# -----------------------------------------------------------------------------
# 2. Episode Sizing & Subdivision Tests (5-25 messages)
# -----------------------------------------------------------------------------


def test_oversized_burst_subdivision():
    """Verify a rapid burst exceeding 25 messages is subdivided into balanced chunks <= 25."""
    t0 = 1792659600.0
    direct_jid = "14155550101@s.whatsapp.net"

    # Generate 30 messages separated by only 5 seconds (single temporal burst)
    messages = [
        make_msg(
            i,
            direct_jid,
            "Alice" if i % 2 == 0 else "You",
            f"Message #{i}",
            t0 + (i * 5),
        )
        for i in range(1, 31)
    ]

    episodes = chunk_into_episodes(messages)

    # 30 messages should split into 2 balanced episodes of 15 each
    assert len(episodes) == 2
    assert len(episodes[0].messages) == 15
    assert len(episodes[1].messages) == 15
    assert all(len(ep.messages) <= 25 for ep in episodes)
    assert all(len(ep.messages) >= 5 for ep in episodes)
    assert [m.pk for m in episodes[0].messages] == list(range(1, 16))
    assert [m.pk for m in episodes[1].messages] == list(range(16, 31))


# -----------------------------------------------------------------------------
# 3. Cross-Burst Quote Context Preservation in Headers & Content
# -----------------------------------------------------------------------------


def test_cross_burst_quote_stitching_in_child_header():
    """Verify quotes spanning burst boundaries preserve parent quote context in child episode header."""
    t0 = 1792659600.0
    group_jid = "120363041234567890@g.us"

    # Episode 1: Morning discussion
    m1 = make_msg(
        101,
        group_jid,
        "Alice",
        "Should we use sqlite-vec or an external vector DB?",
        t0,
    )
    m2 = make_msg(
        102,
        group_jid,
        "Bob",
        "sqlite-vec is zero-cost and embedded, no Docker needed.",
        t0 + 60,
        parent_pk=101,
    )

    # Silence gap of 2 hours (> 15m)
    t_afternoon = t0 + 7200
    # Episode 2: Afternoon reply quoting message #102 from Episode 1
    m3 = make_msg(
        205,
        group_jid,
        "Charlie",
        "Going back to Bob's point, let's verify FTS5 tokenization.",
        t_afternoon,
        parent_pk=102,
    )
    m4 = make_msg(
        206, group_jid, "Bob", "Good call Charlie.", t_afternoon + 30, parent_pk=205
    )

    episodes = chunk_into_episodes(
        [m1, m2, m3, m4], chat_names={group_jid: "Architecture WG"}
    )

    assert len(episodes) == 2
    ep1, ep2 = episodes[0], episodes[1]

    # Episode 1 has intra-episode quote (102 replies to 101 within ep1)
    assert len(ep1.cross_burst_quotes) == 0
    assert "Cross-Burst Quotes" not in ep1.header
    assert "reply to #101" in ep1.content

    # Episode 2 contains cross-burst quote (205 replies to 102 from ep1)
    assert len(ep2.cross_burst_quotes) == 1
    cb = ep2.cross_burst_quotes[0]
    assert cb.child_pk == 205
    assert cb.parent_pk == 102
    assert cb.parent_sender_name == "Bob"
    assert "sqlite-vec is zero-cost" in cb.parent_text

    # Verify parent context is preserved in Episode 2's header
    assert "Cross-Burst Quotes:" in ep2.header
    assert "In reply to msg_id 102 from Bob: 'sqlite-vec is zero-cost" in ep2.header

    # Verify content turn formats cross-burst quote with parent sender & snippet
    assert (
        "in reply to earlier msg_id 102 from Bob: 'sqlite-vec is zero-cost"
        in ep2.content
    )
    assert "Going back to Bob's point" in ep2.content


# -----------------------------------------------------------------------------
# 4. Opt-In Chat Vault & Workspace Filtering Tests
# -----------------------------------------------------------------------------


def test_vault_config_defaults_to_zero_chats_indexed(
    synthetic_vault_fixture: tuple[Path, Path],
):
    """Verify default-zero policy: unconfigured VaultConfig indexes zero chats."""
    chat_storage, contacts = synthetic_vault_fixture
    engine = IngestionEngine(chat_storage, contacts)

    vault = VaultConfig()  # Empty default approved set
    messages, stats = engine.extract_messages(vault_config=vault)

    # 0 messages, 0 sessions ingested
    assert len(messages) == 0
    assert stats.messages_ingested == 0
    assert stats.direct_sessions_ingested == 0
    assert stats.group_sessions_ingested == 0
    assert stats.total_sessions_scanned == 3


def test_vault_config_strictly_filters_approved_chats(
    synthetic_vault_fixture: tuple[Path, Path],
):
    """Verify VaultConfig strictly allows only explicitly approved chats."""
    chat_storage, contacts = synthetic_vault_fixture
    engine = IngestionEngine(chat_storage, contacts)

    vault = VaultConfig()
    alice_jid = "14155550101@s.whatsapp.net"
    vault.approve(alice_jid)

    messages, stats = engine.extract_messages(
        include_direct=True, include_groups=True, vault_config=vault
    )

    # Only Alice's 2 messages should be ingested
    assert len(messages) == 2
    assert stats.messages_ingested == 2
    assert stats.direct_sessions_ingested == 1
    assert stats.group_sessions_ingested == 0
    assert all(m.chat_jid == alice_jid for m in messages)
    assert not any("Bob" in m.sender_name for m in messages)


def test_vault_workspaces_and_toggling(synthetic_vault_fixture: tuple[Path, Path]):
    """Verify workspace creation, chat association, and active toggling."""
    chat_storage, contacts = synthetic_vault_fixture
    engine = IngestionEngine(chat_storage, contacts)

    vault = VaultConfig()
    group_jid = "120363041234567890@g.us"
    alice_jid = "14155550101@s.whatsapp.net"

    # Create workspace "Work" containing Group and Alice
    vault.create_workspace("Work", [group_jid, alice_jid], active=True)

    assert vault.is_chat_approved(group_jid) is True
    assert vault.is_chat_approved(alice_jid) is True
    assert vault.is_chat_approved("14155550102@s.whatsapp.net") is False

    # Extract messages while "Work" is active
    messages, stats = engine.extract_messages(
        include_direct=True, include_groups=True, vault_config=vault
    )
    assert len(messages) == 4
    assert stats.direct_sessions_ingested == 1
    assert stats.group_sessions_ingested == 1

    # Toggle "Work" workspace to INACTIVE
    vault.toggle_workspace("Work", active=False)
    assert vault.is_chat_approved(group_jid) is False
    assert vault.is_chat_approved(alice_jid) is False

    messages_off, stats_off = engine.extract_messages(
        include_direct=True, include_groups=True, vault_config=vault
    )
    assert len(messages_off) == 0
    assert stats_off.messages_ingested == 0


def test_vault_config_serialization(tmp_path: Path):
    """Verify VaultConfig JSON serialization roundtrip."""
    config_file = tmp_path / "test_vault.json"
    vault = VaultConfig()
    vault.approve("14155550101@s.whatsapp.net")
    vault.create_workspace("Work", ["120363041234567890@g.us"], active=True)
    vault.create_workspace("Archive", ["old@s.whatsapp.net"], active=False)
    vault.save(config_file)

    loaded = VaultConfig.load(config_file)
    assert loaded.approved_chat_jids == {"14155550101@s.whatsapp.net"}
    assert loaded.workspaces["Work"] == {"120363041234567890@g.us"}
    assert "Work" in loaded.active_workspaces
    assert "Archive" not in loaded.active_workspaces
    assert loaded.is_chat_approved("14155550101@s.whatsapp.net") is True
    assert loaded.is_chat_approved("120363041234567890@g.us") is True
    assert loaded.is_chat_approved("old@s.whatsapp.net") is False


# -----------------------------------------------------------------------------
# 5. Ingestion Report & Discovery Tests
# -----------------------------------------------------------------------------


def test_ingestion_report_surfaces_discovered_chats(
    synthetic_vault_fixture: tuple[Path, Path],
):
    """Verify IngestionReport discovers all chats, message counts, and approval marks."""
    chat_storage, contacts = synthetic_vault_fixture
    engine = IngestionEngine(chat_storage, contacts)

    vault = VaultConfig()
    vault.approve("14155550101@s.whatsapp.net")
    vault.create_workspace("Devs", ["120363041234567890@g.us"], active=True)

    report = engine.discover_chats(vault_config=vault)
    assert report.total_chats_discovered == 3
    assert report.direct_chats_discovered == 2
    assert report.group_chats_discovered == 1
    assert report.total_messages_discovered == 6
    assert report.approved_chats_count == 2

    # Formatted report string inspection
    formatted = report.format_report()
    assert "TOTAL RECALL: INGESTION REPORT" in formatted
    assert "[x] 14155550101@s.whatsapp.net (Alice Smith)" in formatted
    assert "[x] 120363041234567890@g.us (Engineering Core)" in formatted
    assert "[ ] 14155550102@s.whatsapp.net (Bob Jones)" in formatted
    assert "[Workspaces: Devs]" in formatted


def test_engine_sync_method(synthetic_vault_fixture: tuple[Path, Path]):
    """Verify IngestionEngine.sync() interface returns episodes and populated report."""
    chat_storage, contacts = synthetic_vault_fixture
    engine = IngestionEngine(chat_storage, contacts)

    vault = VaultConfig()
    vault.approve("14155550101@s.whatsapp.net")

    episodes, report = engine.sync(vault_config=vault)
    assert len(episodes) == 1
    assert report.ingested_chats_count == 1
    assert report.ingested_messages_count == 2
    assert report.episodes_generated_count == 1
    assert episodes[0].chat_jid == "14155550101@s.whatsapp.net"
    assert "Alice Smith" in episodes[0].header


# -----------------------------------------------------------------------------
# 6. SearchIndex Episode Indexing & Retrieval Tests
# -----------------------------------------------------------------------------


def test_search_index_episodes_bm25():
    """Verify SearchIndex stores episodes and executes BM25 search over episode headers and content."""
    t0 = 1792659600.0
    group_jid = "120363041234567890@g.us"
    m1 = make_msg(
        1, group_jid, "Alice", "Investigating LanceDB vs sqlite-vec for SIMD KNN.", t0
    )
    m2 = make_msg(
        2,
        group_jid,
        "Bob",
        "sqlite-vec is completely self-contained without Docker.",
        t0 + 60,
        parent_pk=1,
    )

    episodes = chunk_into_episodes([m1, m2], chat_names={group_jid: "Architecture WG"})

    with SearchIndex(":memory:") as index:
        indexed_count = index.index_episodes(episodes)
        assert indexed_count == 1

        # Search for technical term in episode content
        results = index.search_episodes("LanceDB")
        assert len(results) == 1
        assert results[0].episode.episode_id == episodes[0].episode_id
        assert "Architecture WG" in results[0].episode.header
        assert results[0].bm25_score < 0.0

        # Search for participant/chat name in episode header
        header_results = index.search_episodes("Architecture WG")
        assert len(header_results) == 1

        # Verify get_episodes_for_chat
        chat_eps = index.get_episodes_for_chat(group_jid)
        assert len(chat_eps) == 1
        assert len(chat_eps[0].messages) == 2


# -----------------------------------------------------------------------------
# 7. CLI End-to-End Tests
# -----------------------------------------------------------------------------


def test_cli_vault_and_ingest_e2e(
    synthetic_vault_fixture: tuple[Path, Path], tmp_path: Path
):
    """Verify CLI vault management and filtered ingest end-to-end."""
    chat_storage, contacts = synthetic_vault_fixture
    vault_file = tmp_path / "vault.json"
    target_db = tmp_path / "vault_tested.db"
    alice_jid = "14155550101@s.whatsapp.net"

    # 1. Approve chat via CLI vault approve
    buf_appr = io.StringIO()
    with redirect_stdout(buf_appr):
        code_appr = cli_main(
            ["vault", "--vault", str(vault_file), "approve", alice_jid]
        )
    assert code_appr == 0
    assert f"+ {alice_jid}" in buf_appr.getvalue()

    # 2. Create workspace via CLI
    buf_ws = io.StringIO()
    with redirect_stdout(buf_ws):
        code_ws = cli_main(
            [
                "vault",
                "--vault",
                str(vault_file),
                "workspace",
                "create",
                "WorkTeam",
                alice_jid,
            ]
        )
    assert code_ws == 0
    assert "Workspace 'WorkTeam' created" in buf_ws.getvalue()

    # 3. Print report via CLI vault report
    buf_rep = io.StringIO()
    with redirect_stdout(buf_rep):
        code_rep = cli_main(
            [
                "vault",
                "--vault",
                str(vault_file),
                "--source",
                str(chat_storage),
                "--contacts",
                str(contacts),
                "report",
            ]
        )
    assert code_rep == 0
    rep_out = buf_rep.getvalue()
    assert "[x] 14155550101@s.whatsapp.net" in rep_out
    assert "[ ] 14155550102@s.whatsapp.net" in rep_out

    # 4. Ingest using the vault
    buf_ing = io.StringIO()
    with redirect_stdout(buf_ing):
        code_ing = cli_main(
            [
                "ingest",
                "--vault",
                str(vault_file),
                "--source",
                str(chat_storage),
                "--contacts",
                str(contacts),
                "--db",
                str(target_db),
            ]
        )
    assert code_ing == 0
    ing_out = buf_ing.getvalue()
    assert "Messages indexed: 2" in ing_out
    assert "Episodes generated: 1" in ing_out

    # 5. Search episodes via CLI
    buf_ep = io.StringIO()
    with redirect_stdout(buf_ep):
        code_ep = cli_main(["episodes", "review", "--db", str(target_db)])
    assert code_ep == 0
    ep_out = buf_ep.getvalue()
    assert "Found 1 matching episodes" in ep_out
    assert "Alice Smith" in ep_out


def test_cli_vault_approve_all_and_revoke_all(
    synthetic_vault_fixture: tuple[Path, Path], tmp_path: Path
):
    """Verify 'vault approve --all' approves all discovered chats and 'vault revoke --all' clears them."""
    chat_storage, contacts = synthetic_vault_fixture
    vault_file = tmp_path / "test_vault_all.json"

    # 1. Approve all
    buf_app = io.StringIO()
    with redirect_stdout(buf_app):
        code_app = cli_main(
            [
                "vault",
                "--vault",
                str(vault_file),
                "--source",
                str(chat_storage),
                "--contacts",
                str(contacts),
                "approve",
                "--all",
            ]
        )
    assert code_app == 0
    vault = VaultConfig.load(vault_file)
    assert vault.is_chat_approved("14155550101@s.whatsapp.net")
    assert vault.is_chat_approved("14155550102@s.whatsapp.net")

    # 2. Revoke all
    buf_rev = io.StringIO()
    with redirect_stdout(buf_rev):
        code_rev = cli_main(
            [
                "vault",
                "--vault",
                str(vault_file),
                "revoke",
                "--all",
            ]
        )
    assert code_rev == 0
    vault_empty = VaultConfig.load(vault_file)
    assert len(vault_empty.approved_chat_jids) == 0


def test_approved_group_auto_ingested_without_include_groups_flag(
    synthetic_vault_fixture: tuple[Path, Path],
):
    """Verify explicitly approved group chats in the vault are ingested even if include_groups=False."""
    chat_storage, contacts = synthetic_vault_fixture
    vault = VaultConfig()
    group_jid = "120363041234567890@g.us"
    vault.approve(group_jid)

    engine = IngestionEngine(chat_storage, contacts)
    msgs, stats = engine.extract_messages(
        include_direct=True,
        include_groups=False,
        vault_config=vault,
    )
    group_msgs = [m for m in msgs if m.chat_jid == group_jid]
    assert len(group_msgs) == 2
    assert stats.group_sessions_ingested == 1
