from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from whatsapp_total_recall.cli import main as cli_main
from whatsapp_total_recall.discovery import WhatsAppContainers
from whatsapp_total_recall.ingestion import APPLE_EPOCH_OFFSET, IngestionEngine
from whatsapp_total_recall.mcp import MCPServer
from whatsapp_total_recall.search import SearchIndex
from whatsapp_total_recall.vault import VaultConfig


@pytest.fixture
def synthetic_sync_db(tmp_path: Path) -> tuple[Path, Path]:
    """Create synthetic WhatsApp storage with 1:1 chat for incremental sync tests."""
    chat_storage = tmp_path / "ChatStorage.sqlite"
    contacts = tmp_path / "ContactsV2.sqlite"

    con_c = sqlite3.connect(contacts)
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
            ('14155550101@s.whatsapp.net', 'Alice Smith', '+1 415 555 0101');
        """
    )
    con_c.commit()
    con_c.close()

    con_chat = sqlite3.connect(chat_storage)
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

        INSERT INTO ZWACHATSESSION (Z_PK, ZCONTACTJID, ZPARTNERNAME, ZMESSAGECOUNTER, ZUNREADCOUNT)
        VALUES (1, '14155550101@s.whatsapp.net', 'Alice Smith', 2, 0);

        INSERT INTO ZWAMESSAGE (Z_PK, ZCHATSESSION, ZSTANZAID, ZISFROMME, ZMESSAGEDATE, ZSORT, ZTEXT)
        VALUES
            (1, 1, 'MSG-1', 0, 725000000.0, 1, 'First message from Alice'),
            (2, 1, 'MSG-2', 1, 725000060.0, 2, 'My reply to Alice');
        """
    )
    con_chat.commit()
    con_chat.close()
    return chat_storage, contacts


def test_search_index_sync_state_crud(tmp_path: Path):
    db_path = tmp_path / "test_index.db"
    with SearchIndex(db_path) as idx:
        assert idx.get_sync_state("alice@s.whatsapp.net") is None
        assert idx.get_sync_state() == {}

        idx.update_sync_state(
            chat_jid="alice@s.whatsapp.net",
            last_message_pk=10,
            last_apple_date=725000000.0,
            last_unix_timestamp=725000000.0 + APPLE_EPOCH_OFFSET,
            synced_count=5,
        )

        state = idx.get_sync_state("alice@s.whatsapp.net")
        assert state is not None
        assert state["chat_jid"] == "alice@s.whatsapp.net"
        assert state["last_message_pk"] == 10
        assert state["last_apple_date"] == 725000000.0
        assert state["total_messages_synced"] == 5

        # Update on conflict adds synced count
        idx.update_sync_state(
            chat_jid="alice@s.whatsapp.net",
            last_message_pk=15,
            last_apple_date=725000500.0,
            last_unix_timestamp=725000500.0 + APPLE_EPOCH_OFFSET,
            synced_count=2,
        )
        updated = idx.get_sync_state("alice@s.whatsapp.net")
        assert updated is not None
        assert updated["last_message_pk"] == 15
        assert updated["last_apple_date"] == 725000500.0
        assert updated["total_messages_synced"] == 7


def test_incremental_engine_sync(synthetic_sync_db: tuple[Path, Path], tmp_path: Path):
    chat_storage, contacts = synthetic_sync_db
    index_db = tmp_path / "total_recall.db"
    vault = VaultConfig()
    vault.approve("14155550101@s.whatsapp.net")

    engine = IngestionEngine(chat_storage, contacts)

    # 1. First sync - ingests initial 2 messages
    with SearchIndex(index_db) as index:
        episodes, report = engine.sync(
            vault_config=vault, index=index, incremental=True
        )
        assert report.ingested_messages_count == 2
        assert report.episodes_generated_count == 1
        state = index.get_sync_state("14155550101@s.whatsapp.net")
        assert state is not None
        assert state["last_message_pk"] == 2
        assert state["last_apple_date"] == 725000060.0
        assert state["total_messages_synced"] == 2

    # 2. Second sync with no changes - 0 messages ingested
    with SearchIndex(index_db) as index:
        episodes2, report2 = engine.sync(
            vault_config=vault, index=index, incremental=True
        )
        assert report2.ingested_messages_count == 0
        assert report2.episodes_generated_count == 0

    # 3. Add third message to WhatsApp storage
    con = sqlite3.connect(chat_storage)
    con.execute(
        """
        INSERT INTO ZWAMESSAGE (Z_PK, ZCHATSESSION, ZSTANZAID, ZISFROMME, ZMESSAGEDATE, ZSORT, ZTEXT)
        VALUES (3, 1, 'MSG-3', 0, 725000120.0, 3, 'Alice followup message');
        """
    )
    con.commit()
    con.close()

    # 4. Third sync - only ingests message 3
    with SearchIndex(index_db) as index:
        episodes3, report3 = engine.sync(
            vault_config=vault, index=index, incremental=True
        )
        assert report3.ingested_messages_count == 1
        assert report3.episodes_generated_count == 1
        state = index.get_sync_state("14155550101@s.whatsapp.net")
        assert state["last_message_pk"] == 3
        assert state["last_apple_date"] == 725000120.0
        assert state["total_messages_synced"] == 3


def test_mcp_tool_sync_chats(synthetic_sync_db: tuple[Path, Path], tmp_path: Path):
    chat_storage, contacts = synthetic_sync_db
    index_db = tmp_path / "mcp_index.db"
    server = MCPServer(db_path=index_db, token="token-123")

    mock_containers = WhatsAppContainers(
        group_dir=None,
        chat_storage=chat_storage,
        contacts=contacts,
        has_permission=True,
    )

    with patch(
        "whatsapp_total_recall.mcp.discover_whatsapp_containers",
        return_value=mock_containers,
    ):
        # First sync call via MCP
        res = server.execute_tool(
            "sync_chats", {"chat_id": "14155550101@s.whatsapp.net"}
        )
        assert not res.get("isError")
        payload = json.loads(res["content"][0]["text"])
        assert payload["status"] == "success"
        assert payload["messages_ingested"] == 2
        assert payload["sync_state"]["last_message_pk"] == 2

        # Second sync call via MCP - 0 new messages
        res2 = server.execute_tool(
            "sync_chats", {"chat_id": "14155550101@s.whatsapp.net"}
        )
        payload2 = json.loads(res2["content"][0]["text"])
        assert payload2["status"] == "success"
        assert payload2["messages_ingested"] == 0


def test_cli_sync_subcommand(synthetic_sync_db: tuple[Path, Path], tmp_path: Path):
    chat_storage, contacts = synthetic_sync_db
    index_db = tmp_path / "cli_test.db"

    mock_containers = WhatsAppContainers(
        group_dir=None,
        chat_storage=chat_storage,
        contacts=contacts,
        has_permission=True,
    )

    with patch(
        "whatsapp_total_recall.cli.discover_whatsapp_containers",
        return_value=mock_containers,
    ):
        code = cli_main(
            [
                "sync",
                "--source",
                str(chat_storage),
                "--contacts",
                str(contacts),
                "--db",
                str(index_db),
                "--chat",
                "14155550101@s.whatsapp.net",
                "--embedder",
                "mock",
            ]
        )
        assert code == 0

        with SearchIndex(index_db) as index:
            state = index.get_sync_state("14155550101@s.whatsapp.net")
            assert state is not None
            assert state["last_message_pk"] == 2


def test_cli_ingest_populates_sync_state(
    synthetic_sync_db: tuple[Path, Path], tmp_path: Path
):
    chat_storage, contacts = synthetic_sync_db
    index_db = tmp_path / "ingest_sync_test.db"

    mock_containers = WhatsAppContainers(
        group_dir=None,
        chat_storage=chat_storage,
        contacts=contacts,
        has_permission=True,
    )

    with patch(
        "whatsapp_total_recall.cli.discover_whatsapp_containers",
        return_value=mock_containers,
    ):
        code = cli_main(
            [
                "ingest",
                "--source",
                str(chat_storage),
                "--contacts",
                str(contacts),
                "--db",
                str(index_db),
                "--approve",
                "14155550101@s.whatsapp.net",
                "--embedder",
                "mock",
            ]
        )
        assert code == 0

        with SearchIndex(index_db) as index:
            state = index.get_sync_state("14155550101@s.whatsapp.net")
            assert state is not None
            assert state["last_message_pk"] == 2
            assert state["last_apple_date"] == 725000060.0
            assert state["total_messages_synced"] == 2


def test_sync_progress_callback_stages(
    synthetic_sync_db: tuple[Path, Path], tmp_path: Path
):
    chat_storage, contacts = synthetic_sync_db
    index_db = tmp_path / "cb_test.db"
    vault = VaultConfig()
    vault.approve("14155550101@s.whatsapp.net")

    engine = IngestionEngine(chat_storage, contacts)
    stages_seen: list[str] = []

    def on_progress(stage: str, cur: int, tot: int) -> None:
        if stage not in stages_seen:
            stages_seen.append(stage)

    with SearchIndex(index_db) as index:
        engine.sync(
            vault_config=vault,
            index=index,
            incremental=True,
            progress_callback=on_progress,
        )

    assert "Indexing messages" in stages_seen
    assert "Embedding & indexing episodes" in stages_seen
