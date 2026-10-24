from __future__ import annotations

import io
import json
import os
import stat
import threading
from pathlib import Path

import pytest

from whatsapp_total_recall.cli import main as cli_main
from whatsapp_total_recall.drafter import (
    AutonomousSendingProhibitedError,
    DraftReviewBox,
)
from whatsapp_total_recall.ingestion import APPLE_EPOCH_OFFSET, format_iso_timestamp
from whatsapp_total_recall.mcp import (
    MCPHTTPRequestHandler,
    MCPServer,
    MCPSSEServer,
    MCPStdioTransport,
    load_or_create_bridge_token,
    validate_loopback_host,
)
from whatsapp_total_recall.search import SearchIndex
from whatsapp_total_recall.summarizer import DeterministicMockLLMAdapter
from whatsapp_total_recall.types import NormalizedMessage


def make_test_msg(
    pk: int,
    chat_jid: str,
    sender_name: str,
    text: str,
    timestamp: float,
    parent_pk: int | None = None,
    is_from_me: bool = False,
) -> NormalizedMessage:
    return NormalizedMessage(
        pk=pk,
        chat_session_id=1,
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


def seed_test_database(db_path: Path) -> None:
    """Populate an SQLite index with structured test messages including threads and outgoing history."""
    t0 = 1700000000.0
    messages = [
        make_test_msg(
            pk=1,
            chat_jid="12345@g.us",
            sender_name="Alice",
            text="Has anyone reviewed the quarterly architectural specification?",
            timestamp=t0,
        ),
        make_test_msg(
            pk=2,
            chat_jid="12345@g.us",
            sender_name="Bob",
            text="Yes, I left review comments on section 2.3.",
            timestamp=t0 + 60,
            parent_pk=1,
        ),
        make_test_msg(
            pk=3,
            chat_jid="12345@g.us",
            sender_name="You",
            text="I will finalize the deployment plan tomorrow morning.",
            timestamp=t0 + 120,
            parent_pk=2,
            is_from_me=True,
        ),
        make_test_msg(
            pk=4,
            chat_jid="12345@g.us",
            sender_name="Charlie",
            text="Where is the database migration script stored?",
            timestamp=t0 + 180,
        ),
        make_test_msg(
            pk=5,
            chat_jid="direct@s.whatsapp.net",
            sender_name="You",
            text="Hello! Sounds great, thanks.",
            timestamp=t0 + 240,
            is_from_me=True,
        ),
    ]

    with SearchIndex(db_path) as index:
        index.index_messages(messages)


# ==============================================================================
# 1. Token Management & Permissions
# ==============================================================================


def test_token_generation_and_permissions(tmp_path: Path):
    token_file = tmp_path / "test.token"
    token = load_or_create_bridge_token(token_file)

    assert len(token) == 64
    assert token_file.exists()

    # Permissions on POSIX must be 0o600 (owner read/write only)
    if os.name == "posix":
        file_mode = stat.S_IMODE(token_file.stat().st_mode)
        assert file_mode == 0o600

    # Idempotent loading returns existing token
    reloaded = load_or_create_bridge_token(token_file)
    assert reloaded == token


def test_validate_loopback_host():
    # Valid loopback interfaces
    validate_loopback_host("127.0.0.1")
    validate_loopback_host("localhost")
    validate_loopback_host("::1")
    validate_loopback_host("127.0.0.2")

    # Invalid non-loopback interfaces must raise ValueError
    with pytest.raises(
        ValueError, match="MCP server must bind exclusively to 127.0.0.1"
    ):
        validate_loopback_host("0.0.0.0")

    with pytest.raises(
        ValueError, match="MCP server must bind exclusively to 127.0.0.1"
    ):
        validate_loopback_host("192.168.1.50")

    with pytest.raises(
        ValueError, match="MCP server must bind exclusively to 127.0.0.1"
    ):
        validate_loopback_host("example.com")


# ==============================================================================
# 2. Prohibited Tools Stripped / Omitted
# ==============================================================================


def test_prohibited_tools_omitted_and_rejected(tmp_path: Path):
    db_path = tmp_path / "test.db"
    server = MCPServer(db_path=db_path, token="secret-token")

    # tools/list must contain the sanctioned safe tools
    tools = server.get_tool_definitions()
    tool_names = [t["name"] for t in tools]
    assert set(tool_names) == {
        "list_chats",
        "search_messages",
        "get_chat_history",
        "get_thread_context",
        "stage_reply_draft",
    }

    # Autonomous send and presence tracking methods must be stripped and rejected
    prohibited_tools = [
        "send_message",
        "send_reply",
        "stage_and_send",
        "set_presence",
        "track_presence",
        "get_presence",
        "get_online_status",
    ]
    for p_name in prohibited_tools:
        result = server.execute_tool(p_name, {})
        assert result["isError"] is True
        error_text = result["content"][0]["text"]
        assert "Safety Violation" in error_text
        assert "strictly prohibited" in error_text


# ==============================================================================
# 3. Stdio Transport Protocol & Tool Execution
# ==============================================================================


def test_stdio_handshake_and_ping(tmp_path: Path):
    db_path = tmp_path / "test.db"
    server = MCPServer(db_path=db_path, token="secret-token")
    transport = MCPStdioTransport(server)

    input_lines = [
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05"},
            }
        ),
        json.dumps(
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }
        ),
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "ping",
            }
        ),
    ]

    in_stream = io.StringIO("\n".join(input_lines) + "\n")
    out_stream = io.StringIO()
    err_stream = io.StringIO()

    transport.run(reader=in_stream, writer=out_stream, error_writer=err_stream)

    outputs = [json.loads(line) for line in out_stream.getvalue().splitlines() if line]
    assert (
        len(outputs) == 2
    )  # initialize response + ping response (notification emits no response)

    # Initialize check
    assert outputs[0]["id"] == 1
    assert outputs[0]["result"]["protocolVersion"] == "2024-11-05"
    assert outputs[0]["result"]["serverInfo"]["name"] == "whatsapp-total-recall-mcp"

    # Ping check
    assert outputs[1]["id"] == 2
    assert outputs[1]["result"] == {}

    # Modern server/discover probe check
    discover_resp = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 99,
            "method": "server/discover",
            "params": {
                "_meta": {"io.modelcontextprotocol/protocolVersion": "2026-07-28"}
            },
        }
    )
    assert discover_resp["id"] == 99
    assert discover_resp["result"]["protocolVersion"] == "2026-07-28"
    assert "tools" in discover_resp["result"]["capabilities"]


def test_stdio_tool_invocations(tmp_path: Path):
    db_path = tmp_path / "test.db"
    seed_test_database(db_path)

    server = MCPServer(
        db_path=db_path,
        token="secret-token",
        llm_adapter=DeterministicMockLLMAdapter(),
    )
    transport = MCPStdioTransport(server)

    requests = [
        # 1. tools/list
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/list",
        },
        # 2. search_messages
        {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "search_messages",
                "arguments": {"query": "architectural", "limit": 5},
            },
        },
        # 3. get_thread_context
        {
            "jsonrpc": "2.0",
            "id": 12,
            "method": "tools/call",
            "params": {
                "name": "get_thread_context",
                "arguments": {"message_id": 2},
            },
        },
        # 4. stage_reply_draft
        {
            "jsonrpc": "2.0",
            "id": 13,
            "method": "tools/call",
            "params": {
                "name": "stage_reply_draft",
                "arguments": {
                    "chat_id": "12345@g.us",
                    "intent_points": ["Confirming review on section 2.3"],
                },
            },
        },
    ]

    in_stream = io.StringIO("\n".join(json.dumps(r) for r in requests) + "\n")
    out_stream = io.StringIO()
    err_stream = io.StringIO()

    transport.run(reader=in_stream, writer=out_stream, error_writer=err_stream)

    outputs = [json.loads(line) for line in out_stream.getvalue().splitlines() if line]
    assert len(outputs) == 4

    # 1. tools/list check
    tools_list = outputs[0]["result"]["tools"]
    assert len(tools_list) == 5

    # 2. search_messages check
    search_res = json.loads(outputs[1]["result"]["content"][0]["text"])
    assert len(search_res) >= 1
    assert search_res[0]["sender_name"] == "Alice"
    assert "architectural" in search_res[0]["text"]

    # 3. get_thread_context check
    thread_ctx = json.loads(outputs[2]["result"]["content"][0]["text"])
    assert thread_ctx["root_id"] == 1
    assert thread_ctx["thread_length"] == 3
    assert "Alice" in thread_ctx["rendered_tree"]
    assert "Bob" in thread_ctx["rendered_tree"]

    # 4. stage_reply_draft check
    draft_res = json.loads(outputs[3]["result"]["content"][0]["text"])
    assert draft_res["status"].lower() == "staged_for_review"
    assert draft_res["autonomous_sending_prohibited"] is True
    assert "Confirming review on section 2.3" in draft_res["intent_bullets"]
    draft_id = draft_res["draft_id"]

    # Verify draft is in DraftReviewBox and send() is prohibited
    review_box = DraftReviewBox(db_path)
    staged = review_box.get(draft_id)
    assert staged is not None
    assert staged.chat_jid == "12345@g.us"
    with pytest.raises(AutonomousSendingProhibitedError):
        review_box.send(draft_id)


# ==============================================================================
# 4. SSE Transport & Bearer Token Authentication
# ==============================================================================


def test_sse_server_auth_and_messaging(tmp_path: Path):
    import socket

    db_path = tmp_path / "test.db"
    seed_test_database(db_path)

    token = "test-secret-bridge-token"
    server = MCPServer(
        db_path=db_path,
        token=token,
        llm_adapter=DeterministicMockLLMAdapter(),
    )

    sse_server = MCPSSEServer(server, host="127.0.0.1", port=0)

    try:
        # A. Unauthenticated / Invalid Token Requests must be rejected (401)
        # 1. Request without token
        c_sock, s_sock = socket.socketpair()
        c_sock.sendall(b"GET /sse HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        MCPHTTPRequestHandler(s_sock, ("127.0.0.1", 12345), sse_server)
        res_no_auth = c_sock.recv(4096).decode("utf-8")
        assert "401 Unauthorized" in res_no_auth
        assert "WWW-Authenticate: Bearer" in res_no_auth
        c_sock.close()
        s_sock.close()

        # 2. Request with wrong token
        c_sock, s_sock = socket.socketpair()
        c_sock.sendall(
            b"GET /sse HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer wrong-token\r\n\r\n"
        )
        MCPHTTPRequestHandler(s_sock, ("127.0.0.1", 12345), sse_server)
        res_bad_auth = c_sock.recv(4096).decode("utf-8")
        assert "401 Unauthorized" in res_bad_auth
        c_sock.close()
        s_sock.close()

        # B. Health check with valid token
        c_sock, s_sock = socket.socketpair()
        c_sock.sendall(
            f"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {token}\r\n\r\n".encode()
        )
        MCPHTTPRequestHandler(s_sock, ("127.0.0.1", 12345), sse_server)
        res_health = c_sock.recv(4096).decode("utf-8")
        assert "200 OK" in res_health
        assert '"status":"ok"' in res_health
        c_sock.close()
        s_sock.close()

        # C. Query-parameter token authorization (?token=...)
        c_sock, s_sock = socket.socketpair()
        c_sock.sendall(
            f"GET /health?token={token} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n".encode()
        )
        MCPHTTPRequestHandler(s_sock, ("127.0.0.1", 12345), sse_server)
        res_query = c_sock.recv(4096).decode("utf-8")
        assert "200 OK" in res_query
        c_sock.close()
        s_sock.close()

        # D. Connect to /sse and receive endpoint event
        sse_c, sse_s = socket.socketpair()
        sse_c.sendall(
            f"GET /sse HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {token}\r\n\r\n".encode()
        )
        t = threading.Thread(
            target=lambda: MCPHTTPRequestHandler(
                sse_s, ("127.0.0.1", 12345), sse_server
            ),
            daemon=True,
        )
        t.start()

        initial_data = bytearray()
        while b"\n\n" not in initial_data:
            chunk = sse_c.recv(4096)
            if not chunk:
                break
            initial_data.extend(chunk)

        init_str = initial_data.decode("utf-8")
        assert "200 OK" in init_str
        assert "text/event-stream" in init_str
        assert "event: endpoint" in init_str

        endpoint_line = [
            line for line in init_str.splitlines() if line.startswith("data: ")
        ][0]
        endpoint = endpoint_line.split("data: ")[1].strip()
        assert endpoint.startswith("/messages?session_id=")
        session_id = endpoint.split("session_id=")[1].strip()

        # E. POST JSON-RPC request to the endpoint
        post_c, post_s = socket.socketpair()
        rpc_payload = {
            "jsonrpc": "2.0",
            "id": 100,
            "method": "tools/call",
            "params": {
                "name": "search_messages",
                "arguments": {"query": "migration", "limit": 5},
            },
        }
        body = json.dumps(rpc_payload).encode("utf-8")
        post_headers = (
            f"POST /messages?session_id={session_id} HTTP/1.1\r\n"
            f"Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {token}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n\r\n"
        ).encode()
        post_c.sendall(post_headers + body)
        MCPHTTPRequestHandler(post_s, ("127.0.0.1", 12345), sse_server)

        post_res = post_c.recv(4096).decode("utf-8")
        assert "200 OK" in post_res
        post_data = json.loads(post_res.split("\r\n\r\n", 1)[1])
        assert post_data["id"] == 100
        parsed_content = json.loads(post_data["result"]["content"][0]["text"])
        assert len(parsed_content) == 1
        assert parsed_content[0]["sender_name"] == "Charlie"
        post_c.close()
        post_s.close()

        # F. Verify event arrived on the open SSE connection
        sse_event = bytearray()
        while b"\n\n" not in sse_event:
            chunk = sse_c.recv(4096)
            if not chunk:
                break
            sse_event.extend(chunk)

        sse_str = sse_event.decode("utf-8")
        assert "event: message" in sse_str
        sse_data = json.loads(sse_str.split("data: ")[1].strip())
        assert sse_data["id"] == 100
        assert "result" in sse_data

        sse_c.close()
        sse_s.close()

    finally:
        sse_server.shutdown_server()


# ==============================================================================
# 5. CLI Integration Tests
# ==============================================================================


def test_cli_mcp_host_validation_rejection(tmp_path: Path, capsys):
    token_file = tmp_path / ".bridge-token"
    token_file.write_text("test-token\n")

    exit_code = cli_main(
        [
            "mcp",
            "--transport",
            "sse",
            "--host",
            "0.0.0.0",
            "--token-file",
            str(token_file),
        ]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "MCP server must bind exclusively to 127.0.0.1" in captured.err


def test_cli_mcp_stdio_ping(tmp_path: Path, monkeypatch, capsys):
    db_path = tmp_path / "test.db"
    seed_test_database(db_path)
    token_file = tmp_path / ".bridge-token"
    token_file.write_text("test-token\n")

    ping_req = json.dumps({"jsonrpc": "2.0", "id": 42, "method": "ping"}) + "\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(ping_req))

    exit_code = cli_main(
        [
            "mcp",
            "--transport",
            "stdio",
            "--db",
            str(db_path),
            "--token-file",
            str(token_file),
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    res = json.loads(captured.out.strip())
    assert res["id"] == 42
    assert res["result"] == {}


def test_mcp_server_vault_filter_enforcement(tmp_path: Path):
    """Verify MCP tools enforce Opt-In Chat Vault filtering at query time."""
    from whatsapp_total_recall.vault import VaultConfig

    db_path = tmp_path / "test.db"
    seed_test_database(db_path)
    vault_path = tmp_path / "vault_config.json"

    # Default-zero: no chats approved
    vault = VaultConfig()
    vault.save(vault_path)

    server = MCPServer(
        db_path=db_path,
        token="secret-token",
        vault_path=vault_path,
    )

    # 1. Search messages: returns 0 matches because 12345@g.us is not approved
    search_resp = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "search_messages",
                "arguments": {"query": "architectural"},
            },
        }
    )
    assert search_resp["result"]["isError"] is False
    search_data = json.loads(search_resp["result"]["content"][0]["text"])
    assert len(search_data) == 0

    # 2. Get thread context: blocked with error
    thread_resp = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "get_thread_context", "arguments": {"message_id": 1}},
        }
    )
    assert thread_resp["result"]["isError"] is True
    assert (
        "not approved in Opt-In Chat Vault"
        in thread_resp["result"]["content"][0]["text"]
    )

    # 3. Approve chat in vault: query immediately retrieves results
    vault.approve("12345@g.us")
    vault.save(vault_path)

    search_resp2 = server.handle_message(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "search_messages",
                "arguments": {"query": "architectural"},
            },
        }
    )
    assert search_resp2["result"]["isError"] is False
    search_data2 = json.loads(search_resp2["result"]["content"][0]["text"])
    assert len(search_data2) >= 1


def test_list_chats_and_chat_history_tools(tmp_path: Path):
    db_path = tmp_path / "test.db"
    seed_test_database(db_path)

    server = MCPServer(db_path=db_path, token="secret-token")

    # 1. list_chats
    res = server.execute_tool("list_chats", {"limit": 10})
    assert res["isError"] is False
    chats = json.loads(res["content"][0]["text"])
    assert len(chats) == 2
    jids = {c["chat_jid"] for c in chats}
    assert "12345@g.us" in jids
    assert "direct@s.whatsapp.net" in jids

    # Test filtering by query
    res_q = server.execute_tool("list_chats", {"query": "12345"})
    assert res_q["isError"] is False
    chats_q = json.loads(res_q["content"][0]["text"])
    assert len(chats_q) == 1
    assert chats_q[0]["chat_jid"] == "12345@g.us"
    assert chats_q[0]["is_group"] is True

    # 2. get_chat_history without anchor (latest messages)
    hist_latest = server.execute_tool(
        "get_chat_history", {"chat_id": "12345@g.us", "limit": 3}
    )
    assert hist_latest["isError"] is False
    data_latest = json.loads(hist_latest["content"][0]["text"])
    assert data_latest["chat_jid"] == "12345@g.us"
    assert data_latest["message_count"] == 3
    # Chronological order
    pks = [m["pk"] for m in data_latest["messages"]]
    assert pks == [2, 3, 4]

    # 3. get_chat_history with around_message_id
    hist_around = server.execute_tool(
        "get_chat_history",
        {"chat_id": "12345@g.us", "around_message_id": 2, "limit": 4},
    )
    assert hist_around["isError"] is False
    data_around = json.loads(hist_around["content"][0]["text"])
    assert data_around["target_message_id"] == 2
    around_pks = [m["pk"] for m in data_around["messages"]]
    assert 2 in around_pks
    # target flag set
    target_msg = next(m for m in data_around["messages"] if m["pk"] == 2)
    assert target_msg["is_target"] is True

    # 4. search_messages scoped by chat_id
    search_scoped = server.execute_tool(
        "search_messages",
        {"query": "You", "chat_id": "direct@s.whatsapp.net"},
    )
    assert search_scoped["isError"] is False
    data_scoped = json.loads(search_scoped["content"][0]["text"])
    assert all(r["chat_jid"] == "direct@s.whatsapp.net" for r in data_scoped)
