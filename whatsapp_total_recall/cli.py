from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

from whatsapp_total_recall.chunking import (
    DIRECT_BURST_GAP_SECONDS,
    GROUP_BURST_GAP_SECONDS,
    chunk_into_episodes,
)
from whatsapp_total_recall.config import (
    DEFAULT_CONFIG_PATH,
    AppConfig,
    get_canonical_storage_dir,
    get_default_db_path,
    get_default_vault_path,
)
from whatsapp_total_recall.dag import ReplyDAG
from whatsapp_total_recall.discovery import discover_whatsapp_containers
from whatsapp_total_recall.drafter import (
    DraftReviewBox,
    ReplyDrafter,
    detect_unanswered_questions,
)
from whatsapp_total_recall.embeddings import get_embedding_provider
from whatsapp_total_recall.ingestion import IngestionEngine
from whatsapp_total_recall.mcp import (
    DEFAULT_BRIDGE_TOKEN_PATH,
    MCPServer,
    MCPSSEServer,
    MCPStdioTransport,
    load_or_create_bridge_token,
    validate_loopback_host,
)
from whatsapp_total_recall.search import SearchIndex
from whatsapp_total_recall.summarizer import (
    DeterministicMockLLMAdapter,
    OllamaLLMAdapter,
    Summarizer,
)
from whatsapp_total_recall.types import NormalizedMessage
from whatsapp_total_recall.ui import build_native_menu_bar_app, run_ui_server
from whatsapp_total_recall.vault import VaultConfig

DEFAULT_DB_PATH = Path("whatsapp_total_recall.db")
DEFAULT_VAULT_PATH = Path("vault_config.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="whatsapp-total-recall",
        description="WhatsApp Total Recall: Privacy-First WhatsApp Conversational Intelligence",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Ingest subcommand
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest messages from WhatsApp Catalyst storage into the local index",
    )
    ingest_parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Path to ChatStorage.sqlite (defaults to discovered macOS path)",
    )
    ingest_parser.add_argument(
        "--contacts",
        type=Path,
        default=None,
        help="Path to ContactsV2.sqlite (defaults to discovered macOS path)",
    )
    ingest_parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Target SQLite database for indexed messages (default: total_recall.db)",
    )
    ingest_parser.add_argument(
        "--include-groups",
        "--groups",
        action="store_true",
        default=False,
        help="Include group chats in ingestion (defaults to 1:1 only)",
    )
    ingest_parser.add_argument(
        "--vault",
        type=Path,
        default=None,
        help="Path to Opt-In Chat Vault config JSON (default: vault_config.json if present)",
    )
    ingest_parser.add_argument(
        "--approve",
        action="append",
        default=[],
        help="Approve specific chat JID(s) on the fly for this ingestion run",
    )
    ingest_parser.add_argument(
        "--workspace",
        action="append",
        default=[],
        help="Include active workspace(s) for this ingestion run",
    )
    ingest_parser.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Bypass Opt-In Vault perimeter and ingest all matching chats",
    )
    ingest_parser.add_argument(
        "--embedder",
        choices=["auto", "ollama", "mock"],
        default="auto",
        help="Local embedding provider to use (default: auto)",
    )
    ingest_parser.add_argument(
        "--model",
        type=str,
        default="nomic-embed-text",
        help="Embedding model name (default: nomic-embed-text)",
    )
    ingest_parser.add_argument(
        "--host",
        "--ollama-host",
        type=str,
        default=None,
        help="Ollama daemon HTTP host (default: http://127.0.0.1:11434)",
    )
    ingest_parser.add_argument(
        "--group-gap",
        type=float,
        default=None,
        help="Inactivity gap threshold for group chats in minutes (default: 15.0)",
    )
    ingest_parser.add_argument(
        "--direct-gap",
        type=float,
        default=None,
        help="Inactivity gap threshold for 1:1 direct chats in minutes (default: 45.0)",
    )
    ingest_parser.add_argument(
        "--incremental",
        action="store_true",
        default=False,
        help="Perform incremental ingestion using sync_state high-water marks",
    )

    # 1b. Sync subcommand
    sync_parser = subparsers.add_parser(
        "sync",
        help="Incrementally synchronize WhatsApp messages into the local index",
    )
    sync_parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Path to ChatStorage.sqlite (defaults to discovered macOS path)",
    )
    sync_parser.add_argument(
        "--contacts",
        type=Path,
        default=None,
        help="Path to ContactsV2.sqlite (defaults to discovered macOS path)",
    )
    sync_parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Target SQLite database for indexed messages (default: total_recall.db)",
    )
    sync_parser.add_argument(
        "--chat",
        type=str,
        default=None,
        help="Sync specific chat JID only",
    )
    sync_parser.add_argument(
        "--vault",
        type=Path,
        default=None,
        help="Path to Opt-In Chat Vault config JSON",
    )
    sync_parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Bypass incremental sync and force full re-sync",
    )
    sync_parser.add_argument(
        "--embedder",
        choices=["auto", "ollama", "mock"],
        default="auto",
        help="Local embedding provider to use (default: auto)",
    )
    sync_parser.add_argument(
        "--model",
        type=str,
        default="nomic-embed-text",
        help="Embedding model name (default: nomic-embed-text)",
    )
    sync_parser.add_argument(
        "--host",
        "--ollama-host",
        type=str,
        default=None,
        help="Ollama daemon HTTP host (default: http://127.0.0.1:11434)",
    )
    sync_parser.add_argument(
        "--group-gap",
        type=float,
        default=None,
        help="Inactivity gap threshold for group chats in minutes (default: 15.0)",
    )
    sync_parser.add_argument(
        "--direct-gap",
        type=float,
        default=None,
        help="Inactivity gap threshold for 1:1 direct chats in minutes (default: 45.0)",
    )

    # 2. Search subcommand
    search_parser = subparsers.add_parser(
        "search",
        help="Search indexed chats using local hybrid RRF or lexical retrieval",
    )
    search_parser.add_argument(
        "query",
        type=str,
        help="Search query terms",
    )
    search_parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Target SQLite database to query (default: total_recall.db)",
    )
    search_parser.add_argument(
        "--target",
        choices=["auto", "episodes", "messages"],
        default="auto",
        help="Entity to search: episodes or messages (default: auto)",
    )
    search_parser.add_argument(
        "--mode",
        choices=["hybrid", "lexical", "vector"],
        default="hybrid",
        help="Episode retrieval mode (default: hybrid)",
    )
    search_parser.add_argument(
        "--embedder",
        choices=["auto", "ollama", "mock"],
        default="auto",
        help="Embedding provider for semantic query (default: auto)",
    )
    search_parser.add_argument(
        "--k-rrf",
        type=float,
        default=60.0,
        help="Reciprocal Rank Fusion smoothing parameter (default: 60.0)",
    )
    search_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum matching results to return (default: 10)",
    )
    search_parser.add_argument(
        "--host",
        "--ollama-host",
        type=str,
        default=None,
        help="Ollama daemon HTTP host (default: http://127.0.0.1:11434)",
    )
    search_parser.add_argument(
        "--vault",
        type=Path,
        default=None,
        help="Path to Opt-In Chat Vault config JSON to filter query results",
    )

    # 3. Thread subcommand
    thread_parser = subparsers.add_parser(
        "thread",
        help="Inspect full quoted reply thread from root question to final response",
    )
    thread_parser.add_argument(
        "message_id",
        type=int,
        nargs="?",
        default=None,
        help="Message PK to trace (resolves thread root and all replies)",
    )
    thread_parser.add_argument(
        "--chat",
        type=str,
        default=None,
        help="Target chat JID to inspect reply threads for",
    )
    thread_parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Path to ChatStorage.sqlite (defaults to discovered macOS path or local DB)",
    )
    thread_parser.add_argument(
        "--contacts",
        type=Path,
        default=None,
        help="Path to ContactsV2.sqlite",
    )
    thread_parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Local SQLite index database to query (default: total_recall.db)",
    )

    # 4. Vault subcommand
    vault_parser = subparsers.add_parser(
        "vault",
        help="Manage Opt-In Chat Vault perimeters, workspaces, and chat discovery",
    )
    vault_parser.add_argument(
        "--vault",
        type=Path,
        default=DEFAULT_VAULT_PATH,
        help="Path to Vault config file (default: vault_config.json)",
    )
    vault_parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Path to ChatStorage.sqlite",
    )
    vault_parser.add_argument(
        "--contacts",
        type=Path,
        default=None,
        help="Path to ContactsV2.sqlite",
    )
    vault_subparsers = vault_parser.add_subparsers(dest="vault_command", required=True)

    # vault list
    vault_subparsers.add_parser(
        "list",
        help="List all discovered chats and their vault approval status",
    )

    # vault report
    vault_subparsers.add_parser(
        "report",
        help="Surface comprehensive ingestion report of discovered chats and workspaces",
    )

    # vault approve <jids...>
    approve_sub = vault_subparsers.add_parser(
        "approve",
        help="Approve one or more chat JIDs in the vault",
    )
    approve_sub.add_argument("jids", nargs="*", default=[], help="Chat JIDs to approve")
    approve_sub.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Approve all discovered chats from WhatsApp storage",
    )

    # vault revoke <jids...>
    revoke_sub = vault_subparsers.add_parser(
        "revoke",
        help="Revoke approval for one or more chat JIDs in the vault",
    )
    revoke_sub.add_argument("jids", nargs="*", default=[], help="Chat JIDs to revoke")
    revoke_sub.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Revoke all approved chats from the vault",
    )

    # vault workspace [list|create|toggle]
    ws_sub = vault_subparsers.add_parser(
        "workspace",
        help="Manage workspace groupings",
    )
    ws_sub.add_argument(
        "action",
        choices=["list", "create", "toggle"],
        help="Workspace action",
    )
    ws_sub.add_argument("name", nargs="?", default=None, help="Workspace name")
    ws_sub.add_argument(
        "jids", nargs="*", default=[], help="Chat JIDs for create action"
    )

    # 5. Episodes subcommand
    episodes_parser = subparsers.add_parser(
        "episodes",
        help="Inspect or search indexed Conversational Flow Episodes",
    )
    episodes_parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=None,
        help="Optional search query across episodes",
    )
    episodes_parser.add_argument(
        "--chat",
        type=str,
        default=None,
        help="Filter episodes by chat JID",
    )
    episodes_parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Target SQLite database to query (default: total_recall.db)",
    )
    episodes_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum episodes to return (default: 5)",
    )
    episodes_parser.add_argument(
        "--mode",
        choices=["hybrid", "lexical", "vector"],
        default="hybrid",
        help="Episode retrieval mode when querying (default: hybrid)",
    )
    episodes_parser.add_argument(
        "--embedder",
        choices=["auto", "ollama", "mock"],
        default="auto",
        help="Embedding provider for semantic query (default: auto)",
    )
    episodes_parser.add_argument(
        "--host",
        "--ollama-host",
        type=str,
        default=None,
        help="Ollama daemon HTTP host (default: http://127.0.0.1:11434)",
    )

    # 6. Purge subcommand
    purge_parser = subparsers.add_parser(
        "purge",
        help="Hard cryptographic purge of revoked messages or expired disappearing TTLs",
    )
    purge_parser.add_argument(
        "message_id",
        type=str,
        nargs="?",
        default=None,
        help="Message PK or Stanza ID to purge",
    )
    purge_parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Target SQLite database (default: total_recall.db)",
    )
    purge_parser.add_argument(
        "--expired",
        action="store_true",
        default=False,
        help="Purge all expired disappearing messages based on TTL timer metadata",
    )
    purge_parser.add_argument(
        "--all-revoked",
        action="store_true",
        default=False,
        help="Detect and purge all revoked/deleted messages by comparing with source ChatStorage",
    )
    purge_parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Path to ChatStorage.sqlite (for --all-revoked)",
    )
    purge_parser.add_argument(
        "--contacts",
        type=Path,
        default=None,
        help="Path to ContactsV2.sqlite",
    )
    purge_parser.add_argument(
        "--embedder",
        choices=["auto", "ollama", "mock"],
        default="auto",
        help="Embedding provider for recomputing parent episode vectors (default: auto)",
    )
    purge_parser.add_argument(
        "--host",
        "--ollama-host",
        type=str,
        default=None,
        help="Ollama daemon HTTP host (default: http://127.0.0.1:11434)",
    )

    # 7. Summarize subcommand
    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Generate factual court-reporter summaries with prefix NLI verification",
    )
    summarize_parser.add_argument(
        "--chat",
        type=str,
        default=None,
        help="Target chat JID to summarize",
    )
    summarize_parser.add_argument(
        "--thread",
        type=int,
        default=None,
        help="Message PK to trace and summarize full reply DAG thread",
    )
    summarize_parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Target SQLite database (default: total_recall.db)",
    )
    summarize_parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Path to ChatStorage.sqlite",
    )
    summarize_parser.add_argument(
        "--contacts",
        type=Path,
        default=None,
        help="Path to ContactsV2.sqlite",
    )
    summarize_parser.add_argument(
        "--llm",
        choices=["auto", "ollama", "mock"],
        default="auto",
        help="Local LLM adapter to use (default: auto)",
    )
    summarize_parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:7b",
        help="Model identifier (default: qwen2.5:7b)",
    )
    summarize_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Inference temperature T in [0.0, 0.1] (default: 0.0)",
    )
    summarize_parser.add_argument(
        "--intent",
        type=str,
        default=None,
        help="Optional intent notes or focus instructions",
    )
    summarize_parser.add_argument(
        "--no-verify",
        action="store_true",
        default=False,
        help="Bypass prefix NLI verification and display raw completion",
    )
    summarize_parser.add_argument(
        "--host",
        "--ollama-host",
        type=str,
        default=None,
        help="Ollama daemon HTTP host (default: http://127.0.0.1:11434)",
    )

    # 8. Draft subcommand
    draft_parser = subparsers.add_parser(
        "draft",
        help="Synthesize intent-constrained reply draft with surface tone matching",
    )
    draft_parser.add_argument(
        "--chat",
        type=str,
        default=None,
        help="Target chat JID to reply to",
    )
    draft_parser.add_argument(
        "--thread",
        type=int,
        default=None,
        help="Message PK to trace and reply within full DAG thread",
    )
    draft_parser.add_argument(
        "--intent",
        "-i",
        action="append",
        default=[],
        help="Intent bullet point(s) strictly bounding the reply draft",
    )
    draft_parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to SQLite index database (default: total_recall.db)",
    )
    draft_parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Path to ChatStorage.sqlite",
    )
    draft_parser.add_argument(
        "--contacts",
        type=Path,
        default=None,
        help="Path to ContactsV2.sqlite",
    )
    draft_parser.add_argument(
        "--llm",
        choices=["auto", "ollama", "mock"],
        default="auto",
        help="Local LLM adapter (default: auto)",
    )
    draft_parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:7b",
        help="Model identifier (default: qwen2.5:7b)",
    )
    draft_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Drafting temperature T in [0.0, 0.2] (default: 0.0)",
    )
    draft_parser.add_argument(
        "--list-staged",
        action="store_true",
        default=False,
        help="List pending drafts in the staging review box",
    )
    draft_parser.add_argument(
        "--approve",
        type=str,
        default=None,
        help="Approve a staged draft by ID for manual copy",
    )
    draft_parser.add_argument(
        "--reject",
        type=str,
        default=None,
        help="Reject a staged draft by ID",
    )
    draft_parser.add_argument(
        "--delete",
        type=str,
        default=None,
        help="Permanently delete a staged draft by ID",
    )
    draft_parser.add_argument(
        "--host",
        "--ollama-host",
        type=str,
        default=None,
        help="Ollama daemon HTTP host (default: http://127.0.0.1:11434)",
    )

    # 9. Config subcommand
    config_parser = subparsers.add_parser(
        "config",
        help="View and manage persistent user configuration and preferences",
    )
    config_parser.add_argument(
        "action",
        choices=["show", "set", "init"],
        nargs="?",
        default="show",
        help="Configuration action: show, set, or init (default: show)",
    )
    config_parser.add_argument(
        "--host",
        "--ollama-host",
        type=str,
        default=None,
        help="Set default Ollama daemon host URL",
    )
    config_parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Set default embedding model name",
    )
    config_parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="Set default LLM model name for summarization and drafting",
    )
    config_parser.add_argument(
        "--group-gap",
        type=float,
        default=None,
        help="Set default group chat burst gap in minutes",
    )
    config_parser.add_argument(
        "--direct-gap",
        type=float,
        default=None,
        help="Set default direct 1:1 chat burst gap in minutes",
    )
    config_parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Set default target SQLite database path",
    )
    config_parser.add_argument(
        "--vault",
        type=Path,
        default=None,
        help="Set default vault config path",
    )
    config_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Target config file path (default: config.json)",
    )

    # 10. MCP subcommand
    mcp_parser = subparsers.add_parser(
        "mcp",
        help="Run local-first Model Context Protocol (MCP) server over stdio or token-authenticated SSE",
    )
    mcp_parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport protocol: stdio or sse (default: stdio)",
    )
    mcp_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface for SSE server; must bind strictly to 127.0.0.1 (default: 127.0.0.1)",
    )
    mcp_parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for SSE server (default: 8765)",
    )
    mcp_parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Bearer token for authentication (overrides token file)",
    )
    mcp_parser.add_argument(
        "--token-file",
        type=Path,
        default=DEFAULT_BRIDGE_TOKEN_PATH,
        help="Path to token file (default: .bridge-token)",
    )
    mcp_parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Target SQLite database to query (default: total_recall.db)",
    )
    mcp_parser.add_argument(
        "--llm",
        choices=["auto", "ollama", "mock"],
        default="auto",
        help="Local LLM adapter for reply drafting (default: auto)",
    )
    mcp_parser.add_argument(
        "--model",
        type=str,
        default="qwen2.5:7b",
        help="LLM model identifier (default: qwen2.5:7b)",
    )
    mcp_parser.add_argument(
        "--host-ollama",
        "--ollama-host",
        dest="host_ollama",
        type=str,
        default=None,
        help="Ollama daemon HTTP host",
    )
    mcp_parser.add_argument(
        "--vault",
        type=Path,
        default=DEFAULT_VAULT_PATH,
        help="Path to Opt-In Chat Vault config JSON (default: vault_config.json)",
    )

    # 11. UI subcommand
    ui_parser = subparsers.add_parser(
        "ui",
        help="Run standalone macOS menu bar application and local UI daemon",
    )
    ui_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind daemon; must bind strictly to 127.0.0.1 (default: 127.0.0.1)",
    )
    ui_parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for local UI daemon (default: 8765)",
    )
    ui_parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Target SQLite database (default: total_recall.db)",
    )
    ui_parser.add_argument(
        "--vault",
        type=Path,
        default=DEFAULT_VAULT_PATH,
        help="Path to vault config JSON (default: vault_config.json)",
    )
    ui_parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run daemon headlessly without spawning native desktop window",
    )
    ui_parser.add_argument(
        "--open-browser",
        action="store_true",
        default=False,
        help="Open default web browser",
    )
    ui_parser.add_argument(
        "--build-only",
        action="store_true",
        default=False,
        help="Compile native macOS Swift menu bar application and exit",
    )

    # 13. Install Skill subcommand
    skill_parser = subparsers.add_parser(
        "install-skill",
        help="Install WhatsApp Total Recall agent skill for Claude Code, Antigravity, Cursor, etc.",
    )
    skill_parser.add_argument(
        "--target",
        choices=["all", "claude", "antigravity", "cursor", "gemini", "custom"],
        default="all",
        help="Target AI assistant / agent environment (default: all)",
    )
    skill_parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Custom destination directory to install SKILL.md into",
    )
    skill_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        default=False,
        help="Overwrite existing SKILL.md files",
    )

    # 14. Reset / Opt-Out subcommand
    reset_parser = subparsers.add_parser(
        "reset",
        help="Complete opt-out: securely purge local database, configs, tokens, skills, and MCP registrations",
    )
    reset_parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        default=False,
        help="Bypass interactive confirmation prompt",
    )
    reset_parser.add_argument(
        "--keep-db",
        action="store_true",
        default=False,
        help="Retain the SQLite index database file and only remove configs, skills, and MCP registrations",
    )
    reset_parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Specific database path to purge (defaults to discovered database)",
    )
    reset_parser.add_argument(
        "--vault",
        type=Path,
        default=None,
        help="Specific vault config path to delete (defaults to discovered vault)",
    )

    return parser


def _resolve_source_and_contacts(
    source_path: Path | None, contacts_path: Path | None
) -> tuple[Path | None, Path | None]:
    if source_path and source_path.exists():
        return source_path, contacts_path

    containers = discover_whatsapp_containers()
    if (
        containers.has_permission
        and containers.chat_storage
        and containers.chat_storage.exists()
    ):
        return containers.chat_storage, contacts_path or containers.contacts
    return None, None


def _render_progress_bar(current: int, total: int, prefix: str = "Processing") -> None:
    if total <= 0:
        return
    pct = (current / total) * 100
    bar_len = 25
    filled = int(bar_len * current // total)
    bar = (
        "=" * filled
        + (">" if filled < bar_len else "")
        + " " * max(0, bar_len - filled - 1 if filled < bar_len else 0)
    )
    sys.stdout.write(f"\r{prefix}: [{bar}] {current}/{total} ({pct:.1f}%)")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def run_ingest(args: argparse.Namespace) -> int:
    source_path = args.source
    contacts_path = args.contacts

    if not source_path:
        containers = discover_whatsapp_containers()
        if not containers.has_permission:
            print(
                "Error: Permission denied accessing WhatsApp container. Full Disk Access is required.",
                file=sys.stderr,
            )
            return 1
        if not containers.chat_storage or not containers.chat_storage.exists():
            print(
                "Error: Could not locate ChatStorage.sqlite. Specify --source explicitly.",
                file=sys.stderr,
            )
            return 1
        source_path = containers.chat_storage
        contacts_path = containers.contacts

    if not source_path.exists():
        print(f"Error: ChatStorage file not found: {source_path}", file=sys.stderr)
        return 1

    # Resolve VaultConfig
    vault_config: VaultConfig | None = None
    if not args.all:
        if args.vault:
            vault_config = VaultConfig.load(args.vault)
        elif args.approve or args.workspace:
            vault_config = (
                VaultConfig.load(DEFAULT_VAULT_PATH)
                if DEFAULT_VAULT_PATH.exists()
                else VaultConfig()
            )
        elif DEFAULT_VAULT_PATH.exists():
            vault_config = VaultConfig.load(DEFAULT_VAULT_PATH)

    if vault_config is not None:
        for jid in args.approve:
            vault_config.approve(jid)
        for ws in args.workspace:
            vault_config.toggle_workspace(ws, active=True)

    print(f"Reading WhatsApp database: {source_path}")
    engine = IngestionEngine(source_path, contacts_path)
    try:
        report = engine.discover_chats(vault_config=vault_config)
        include_groups = bool(args.include_groups)
        if not include_groups and vault_config:
            has_approved_groups = any(
                jid.endswith("@g.us") for jid in vault_config.approved_chat_jids
            )
            if has_approved_groups:
                include_groups = True

        since_apple_dates: dict[str, float] | None = None
        if getattr(args, "incremental", False) and args.db.exists():
            with SearchIndex(args.db) as temp_idx:
                sync_states = temp_idx.get_sync_state()
                if isinstance(sync_states, dict):
                    since_apple_dates = {
                        jid: float(rec["last_apple_date"])
                        for jid, rec in sync_states.items()
                        if "last_apple_date" in rec
                    }

        print("Extracting and normalizing messages...")
        messages, stats = engine.extract_messages(
            include_direct=True,
            include_groups=include_groups,
            vault_config=vault_config,
            since_apple_dates=since_apple_dates,
        )
    except Exception as exc:
        print(f"Error extracting messages: {exc}", file=sys.stderr)
        return 1

    print(
        f"Extracted {len(messages)} message(s) from {stats.total_sessions_scanned} session(s)."
    )

    # Chunk into Conversational Flow Episodes
    chat_names = {c.chat_jid: c.chat_name for c in report.chats}
    grp_gap = (
        args.group_gap * 60.0
        if getattr(args, "group_gap", None) is not None
        else (
            vault_config.group_burst_minutes * 60.0
            if vault_config and hasattr(vault_config, "group_burst_minutes")
            else GROUP_BURST_GAP_SECONDS
        )
    )
    dir_gap = (
        args.direct_gap * 60.0
        if getattr(args, "direct_gap", None) is not None
        else (
            vault_config.direct_burst_minutes * 60.0
            if vault_config and hasattr(vault_config, "direct_burst_minutes")
            else DIRECT_BURST_GAP_SECONDS
        )
    )
    print("Segmenting into conversational flow episodes...")
    episodes = chunk_into_episodes(
        messages,
        chat_names=chat_names,
        group_gap_seconds=grp_gap,
        direct_gap_seconds=dir_gap,
    )
    print(f"Generated {len(episodes)} episode(s).")

    embedder_name = getattr(args, "embedder", "auto")
    print(f"Resolving embedding provider ({embedder_name})...")
    host = _resolve_host(args, vault_config)
    embedder = get_embedding_provider(
        name=embedder_name,
        model=getattr(args, "model", "nomic-embed-text"),
        host=host,
    )

    print(f"Opening index store '{args.db}'...")
    with SearchIndex(args.db, embedder=embedder) as index:
        indexed_count = len(index.get_all_message_pks())
        if indexed_count > 0:
            print(f"Checking revocations across {indexed_count} indexed message(s)...")
            rev_reports, exp_reports = engine.purge_revocations_and_expired(
                index=index,
                embedder=embedder,
                progress_callback=lambda cur, tot: _render_progress_bar(
                    cur, tot, prefix="Purging revocations"
                ),
            )
            if rev_reports:
                print(f"Purged {len(rev_reports)} revoked message(s) from index.")
            if exp_reports:
                print(
                    f"Purged {len(exp_reports)} expired disappearing message(s) from index."
                )
        else:
            print("Index store is clean (no previous messages to purge).")

        if messages:
            print(f"Indexing {len(messages)} message(s) into FTS5 store...")
            index.index_messages(
                messages,
                progress_callback=lambda cur, tot: _render_progress_bar(
                    cur, tot, prefix="Indexing messages"
                ),
            )
            chat_msg_map: dict[str, list[NormalizedMessage]] = {}
            for m in messages:
                chat_msg_map.setdefault(m.chat_jid, []).append(m)

            for jid, msgs in chat_msg_map.items():
                last_m = max(msgs, key=lambda m: (m.apple_date, m.pk))
                index.update_sync_state(
                    chat_jid=jid,
                    last_message_pk=last_m.pk,
                    last_apple_date=last_m.apple_date,
                    last_unix_timestamp=last_m.unix_timestamp,
                    synced_count=len(msgs),
                )
        if episodes:
            print(f"Computing embeddings & indexing {len(episodes)} episode(s)...")
            indexed_episodes = index.index_episodes(
                episodes,
                index_constituent_messages=False,
                embedder=embedder,
                progress_callback=lambda cur, tot: _render_progress_bar(
                    cur, tot, prefix="Embedding & indexing episodes"
                ),
            )
        else:
            indexed_episodes = 0

    print(f"Ingestion complete into '{args.db}':")
    print(f"  Sessions scanned: {stats.total_sessions_scanned}")
    print(f"  Direct 1:1 sessions: {stats.direct_sessions_ingested}")
    if stats.group_sessions_ingested > 0:
        print(f"  Group sessions: {stats.group_sessions_ingested}")
    print(f"  Messages indexed: {stats.messages_ingested}")
    if indexed_episodes > 0:
        print(f"  Episodes generated: {indexed_episodes}")

    if vault_config is not None:
        print("\n" + report.format_report())

    return 0


def run_sync(args: argparse.Namespace) -> int:
    source_path = args.source
    contacts_path = args.contacts

    if not source_path:
        containers = discover_whatsapp_containers()
        if not containers.has_permission:
            print(
                "Error: Permission denied accessing WhatsApp container. Full Disk Access is required.",
                file=sys.stderr,
            )
            return 1
        if not containers.chat_storage or not containers.chat_storage.exists():
            print(
                "Error: Could not locate ChatStorage.sqlite. Specify --source explicitly.",
                file=sys.stderr,
            )
            return 1
        source_path = containers.chat_storage
        contacts_path = containers.contacts

    if not source_path.exists():
        print(f"Error: ChatStorage file not found: {source_path}", file=sys.stderr)
        return 1

    vault_config: VaultConfig | None = None
    if getattr(args, "vault", None):
        vault_config = VaultConfig.load(args.vault)
    elif DEFAULT_VAULT_PATH.exists():
        vault_config = VaultConfig.load(DEFAULT_VAULT_PATH)
    else:
        vault_config = VaultConfig()

    target_chat = getattr(args, "chat", None)
    if target_chat:
        vault_config.approve(target_chat)

    embedder_name = getattr(args, "embedder", "auto")
    host = _resolve_host(args, vault_config)
    embedder = get_embedding_provider(
        name=embedder_name,
        model=getattr(args, "model", "nomic-embed-text"),
        host=host,
    )

    grp_gap = (
        args.group_gap * 60.0 if getattr(args, "group_gap", None) is not None else None
    )
    dir_gap = (
        args.direct_gap * 60.0
        if getattr(args, "direct_gap", None) is not None
        else None
    )

    mode_label = "Full backfill" if getattr(args, "force", False) else "Incremental"
    print(f"Reading WhatsApp database: {source_path}")
    print(f"Opening index store '{args.db}' ({mode_label.lower()} sync)...")

    engine = IngestionEngine(source_path, contacts_path)
    with SearchIndex(args.db, embedder=embedder) as index:
        episodes, report = engine.sync(
            vault_config=vault_config,
            index=index,
            embedder=embedder,
            group_gap_seconds=grp_gap,
            direct_gap_seconds=dir_gap,
            incremental=not getattr(args, "force", False),
            chat_jid=target_chat,
            progress_callback=lambda stage, cur, tot: _render_progress_bar(
                cur, tot, prefix=stage
            ),
        )

    print(f"{mode_label} sync complete into '{args.db}':")
    print(f"  Chats synced: {report.ingested_chats_count}")
    print(f"  Messages ingested: {report.ingested_messages_count}")
    print(f"  Episodes generated: {report.episodes_generated_count}")
    if report.revocations_purged_count > 0:
        print(f"  Revocations purged: {report.revocations_purged_count}")
    if report.expired_purged_count > 0:
        print(f"  Expired messages purged: {report.expired_purged_count}")

    if vault_config is not None and report.ingested_chats_count > 0:
        print("\n" + report.format_report())

    return 0


def run_search(args: argparse.Namespace) -> int:
    db_path: Path = args.db
    if not db_path.exists():
        print(
            f"Error: Index database '{db_path}' does not exist. Run 'whatsapp-total-recall ingest' first.",
            file=sys.stderr,
        )
        return 1

    target = getattr(args, "target", "auto")
    mode = getattr(args, "mode", "hybrid")
    embedder_name = getattr(args, "embedder", "auto")
    host = _resolve_host(args)
    embedder = get_embedding_provider(embedder_name, host=host)

    with SearchIndex(db_path, embedder=embedder) as index:
        # Determine whether to search messages or episodes
        search_episodes = target == "episodes" or (
            target == "auto" and len(index.get_all_episodes()) > 0
        )
        if target == "messages":
            search_episodes = False

        vault = (
            VaultConfig.load(args.vault)
            if getattr(args, "vault", None) and Path(args.vault).exists()
            else None
        )

        if not search_episodes:
            candidate_limit = args.limit if vault is None else max(args.limit * 3, 30)
            results = index.search(args.query, limit=candidate_limit)
            if vault is not None:
                results = [r for r in results if vault.is_chat_approved(r.chat_jid)]
            results = results[: args.limit]
            if not results:
                print(f"No matching messages found for: {args.query}")
                return 0

            print(f"Found {len(results)} matching messages for '{args.query}':\n")
            for r in results:
                print(f"[{r.iso_date}] {r.sender_name}:")
                print(f"{r.text}\n")
            return 0

        # Episode search
        k_rrf = getattr(args, "k_rrf", 60.0)
        candidate_limit = args.limit if vault is None else max(args.limit * 3, 30)
        if mode == "hybrid":
            scored = index.search_hybrid(
                args.query, limit=candidate_limit, k_rrf=k_rrf, embedder=embedder
            )
        elif mode == "vector":
            scored = index.search_episodes_vector(
                args.query, limit=candidate_limit, embedder=embedder
            )
        else:
            scored = index.search_episodes(args.query, limit=candidate_limit)

        if vault is not None:
            scored = [s for s in scored if vault.is_chat_approved(s.episode.chat_jid)]
        scored = scored[: args.limit]

        if not scored:
            print(f"No matching episodes found for: {args.query}")
            return 0

        print(f"Found {len(scored)} matching episodes for '{args.query}':\n")
        for s in scored:
            if mode == "hybrid":
                score_str = f"RRF Score: {s.rrf_score:.5f}"
                ranks = []
                if s.fts_rank is not None:
                    ranks.append(f"FTS rank: {s.fts_rank}")
                if s.vec_rank is not None:
                    ranks.append(f"Vec rank: {s.vec_rank}")
                if ranks:
                    score_str += f" ({', '.join(ranks)})"
            elif mode == "vector":
                dist_str = (
                    f"{s.vector_distance:.4f}"
                    if s.vector_distance is not None
                    else "N/A"
                )
                score_str = f"Vector Distance: {dist_str} (Vec rank: {s.vec_rank})"
            else:
                score_str = f"BM25 Score: {s.bm25_score:.3f}"

            print(f"{s.episode.header}\n{score_str}")
            print("-" * 50)
            print(s.episode.content)
            print("\n" + "=" * 60 + "\n")
        return 0


def run_thread(args: argparse.Namespace) -> int:
    messages = []

    if args.source and args.source.exists():
        engine = IngestionEngine(args.source, args.contacts)
        messages, _ = engine.extract_messages(include_direct=True, include_groups=True)
    elif args.db and args.db.exists():
        with SearchIndex(args.db) as index:
            if args.chat:
                messages = index.get_messages_for_chat(args.chat)
            else:
                messages = index.get_all_messages()
    else:
        containers = discover_whatsapp_containers()
        if containers.chat_storage and containers.chat_storage.exists():
            engine = IngestionEngine(containers.chat_storage, containers.contacts)
            messages, _ = engine.extract_messages(
                include_direct=True, include_groups=True
            )
        else:
            print(
                "Error: No index database or WhatsApp containers found. "
                "Run 'whatsapp-total-recall ingest' or specify --source.",
                file=sys.stderr,
            )
            return 1

    if args.chat:
        messages = [m for m in messages if m.chat_jid == args.chat]

    if not messages:
        print(
            f"No messages found{' for chat ' + args.chat if args.chat else ''}.",
            file=sys.stderr,
        )
        return 1

    dag = ReplyDAG(messages)

    if args.message_id is not None:
        if args.message_id not in dag.messages:
            print(
                f"Error: Message #{args.message_id} not found in available messages.",
                file=sys.stderr,
            )
            return 1

        root = dag.get_root_for(args.message_id)
        thread_messages = dag.get_thread_messages(root.pk)
        tree_view = dag.render_thread_tree(root.pk)

        print(
            f"Thread for message #{args.message_id} "
            f"(Root: #{root.pk}, {len(thread_messages)} messages in thread):\n"
        )
        print(tree_view)
        return 0

    if args.chat:
        roots = dag.get_roots()
        print(f"Reply threads for chat {args.chat} ({len(roots)} threads):\n")
        print(dag.render_ascii_tree())
        return 0

    print(
        "Error: Please specify a message_id or --chat <chat_jid> to inspect reply threads.",
        file=sys.stderr,
    )
    return 1


def run_vault(args: argparse.Namespace) -> int:
    vault_path: Path = args.vault
    config = VaultConfig.load(vault_path) if vault_path.exists() else VaultConfig()

    cmd = args.vault_command

    if cmd == "approve":
        jids_to_approve = list(args.jids)
        if getattr(args, "all", False):
            source, contacts = _resolve_source_and_contacts(
                getattr(args, "source", None), getattr(args, "contacts", None)
            )
            if not source or not source.exists():
                print(
                    "Error: ChatStorage.sqlite not found. Specify --source explicitly.",
                    file=sys.stderr,
                )
                return 1
            engine = IngestionEngine(source, contacts)
            report = engine.discover_chats()
            for chat in report.chats:
                if chat.chat_jid not in jids_to_approve:
                    jids_to_approve.append(chat.chat_jid)

        if not jids_to_approve:
            print(
                "Error: Specify chat JID(s) or pass --all to approve all discovered chats.",
                file=sys.stderr,
            )
            return 1

        for jid in jids_to_approve:
            config.approve(jid)
        config.save(vault_path)
        print(f"Approved {len(jids_to_approve)} chat(s) in vault '{vault_path}':")
        for jid in jids_to_approve:
            print(f"  + {jid}")
        return 0

    if cmd == "revoke":
        jids_to_revoke = list(args.jids)
        if getattr(args, "all", False):
            jids_to_revoke = list(config.approved_chat_jids)

        if not jids_to_revoke:
            print(
                "Error: Specify chat JID(s) or pass --all to revoke all approved chats.",
                file=sys.stderr,
            )
            return 1

        for jid in jids_to_revoke:
            config.revoke(jid)
        config.save(vault_path)
        print(f"Revoked {len(jids_to_revoke)} chat(s) from vault '{vault_path}':")
        for jid in jids_to_revoke:
            print(f"  - {jid}")
        return 0

    if cmd == "workspace":
        action = args.action
        if action == "list":
            if not config.workspaces:
                print(f"No workspaces configured in '{vault_path}'.")
                return 0
            print(f"Workspaces in '{vault_path}':")
            for name, jids in sorted(config.workspaces.items()):
                status = "ACTIVE" if name in config.active_workspaces else "INACTIVE"
                print(f"  • {name} [{status}] ({len(jids)} chats):")
                for jid in sorted(jids):
                    print(f"      - {jid}")
            return 0

        if action == "create":
            if not args.name:
                print(
                    "Error: Workspace name is required to create a workspace.",
                    file=sys.stderr,
                )
                return 1
            config.create_workspace(args.name, args.jids, active=True)
            config.save(vault_path)
            print(
                f"Workspace '{args.name}' created with {len(args.jids)} chats and set to ACTIVE."
            )
            return 0

        if action == "toggle":
            if not args.name:
                print("Error: Workspace name is required to toggle.", file=sys.stderr)
                return 1
            new_state = config.toggle_workspace(args.name)
            config.save(vault_path)
            state_str = "ACTIVE" if new_state else "INACTIVE"
            print(f"Workspace '{args.name}' toggled to {state_str}.")
            return 0

    if cmd in ("list", "report"):
        source, contacts = _resolve_source_and_contacts(args.source, args.contacts)
        if not source:
            print(
                "Error: ChatStorage.sqlite not found. Specify --source explicitly.",
                file=sys.stderr,
            )
            return 1

        engine = IngestionEngine(source, contacts)
        report = engine.discover_chats(vault_config=config)

        if cmd == "report":
            print(report.format_report())
        else:
            print(f"Discovered chats in '{source}' ({len(report.chats)} total):\n")
            for c in report.chats:
                mark = "[x]" if c.is_approved else "[ ]"
                ctype = "Group" if c.is_group else "1:1"
                ws = f" (Workspaces: {', '.join(c.workspaces)})" if c.workspaces else ""
                print(
                    f"  {mark} {c.chat_jid} | {c.chat_name} | {ctype} | {c.message_count} msgs{ws}"
                )

        return 0

    return 0


def run_episodes(args: argparse.Namespace) -> int:
    db_path: Path = args.db
    if not db_path.exists():
        print(
            f"Error: Database '{db_path}' does not exist. Run 'whatsapp-total-recall ingest' first.",
            file=sys.stderr,
        )
        return 1

    mode = getattr(args, "mode", "hybrid")
    embedder_name = getattr(args, "embedder", "auto")
    host = _resolve_host(args)
    embedder = get_embedding_provider(embedder_name, host=host)

    with SearchIndex(db_path, embedder=embedder) as index:
        if args.query:
            if mode == "lexical":
                scored = index.search_episodes(args.query, limit=args.limit)
            elif mode == "vector":
                scored = index.search_episodes_vector(
                    args.query, limit=args.limit, embedder=embedder
                )
            else:
                scored = index.search_hybrid(
                    args.query, limit=args.limit, embedder=embedder
                )

            if not scored:
                print(f"No matching episodes found for query: '{args.query}'")
                return 0
            print(f"Found {len(scored)} matching episodes for '{args.query}':\n")
            for s in scored:
                if mode == "hybrid":
                    score_str = f"Score: {s.rrf_score:.5f}"
                elif mode == "vector":
                    score_str = f"Score: {s.vector_distance:.4f}"
                else:
                    score_str = f"Score: {s.bm25_score:.3f}"
                print(f"{s.episode.header}\n{score_str}")
                print("-" * 50)
                print(s.episode.content)
                print("\n" + "=" * 60 + "\n")
            return 0

        if args.chat:
            episodes = index.get_episodes_for_chat(args.chat)[: args.limit]
        else:
            episodes = index.get_all_episodes()[: args.limit]

    if not episodes:
        print("No episodes found.")
        return 0

    print(f"Displaying {len(episodes)} Conversational Flow Episodes:\n")
    for ep in episodes:
        print(ep.header)
        print("-" * 50)
        print(ep.content)
        print("\n" + "=" * 60 + "\n")
    return 0


def run_purge(args: argparse.Namespace) -> int:
    db_path: Path = args.db
    if not db_path.exists():
        print(f"Error: Database '{db_path}' does not exist.", file=sys.stderr)
        return 1

    embedder_name = getattr(args, "embedder", "auto")
    host = _resolve_host(args)
    embedder = get_embedding_provider(embedder_name, host=host)

    if not args.message_id and not args.expired and not args.all_revoked:
        print(
            "Error: Please specify a message_id, --expired, or --all-revoked.",
            file=sys.stderr,
        )
        return 1

    with SearchIndex(db_path, embedder=embedder) as index:
        # Case 1: Purge specific message by PK or stanza_id
        if args.message_id:
            report = index.purge(args.message_id, embedder=embedder)
            target = (
                f"#{report.message_pk}" if report.message_pk else str(args.message_id)
            )
            if (
                report.raw_messages_purged > 0
                or report.episodes_updated > 0
                or report.episodes_deleted > 0
            ):
                print(f"Cryptographic purge successful for message {target}:")
                print(f"  Raw message removed: {report.raw_messages_purged}")
                print(f"  Episodes updated: {report.episodes_updated}")
                print(f"  Episodes deleted: {report.episodes_deleted}")
                print(
                    f"  Disk secure delete (zero-fill): {report.secure_delete_executed}"
                )
            else:
                print(f"Message {target} not found in index.")

        # Case 2: Purge expired disappearing messages
        if args.expired:
            expired_reports = index.purge_expired(embedder=embedder)
            print(f"Purged {len(expired_reports)} expired disappearing message(s).")

        # Case 3: Scan source ChatStorage for all revocations
        if args.all_revoked:
            source, contacts = _resolve_source_and_contacts(args.source, args.contacts)
            if not source:
                print(
                    "Error: ChatStorage.sqlite not found. Specify --source explicitly.",
                    file=sys.stderr,
                )
                return 1
            engine = IngestionEngine(source, contacts)
            rev_pks = engine.detect_revocation_events(index=index)
            rev_count = 0
            for pk in rev_pks:
                rep = index.purge(pk, embedder=embedder)
                if (
                    rep.raw_messages_purged > 0
                    or rep.episodes_updated > 0
                    or rep.episodes_deleted > 0
                ):
                    rev_count += 1
            print(f"Purged {rev_count} revoked message(s) identified from '{source}'.")

    return 0


def _resolve_host(
    args: argparse.Namespace, vault_cfg: VaultConfig | None = None
) -> str:
    host_arg = getattr(args, "ollama_host", None) or getattr(args, "host", None)
    if host_arg:
        return host_arg
    if vault_cfg and hasattr(vault_cfg, "ollama_host") and vault_cfg.ollama_host:
        return vault_cfg.ollama_host
    if DEFAULT_CONFIG_PATH.exists():
        app_cfg = AppConfig.load(DEFAULT_CONFIG_PATH)
        return app_cfg.ollama_host
    return "http://127.0.0.1:11434"


def _get_llm_adapter(
    provider_choice: str, model: str, host: str = "http://127.0.0.1:11434"
):
    if provider_choice == "mock":
        return DeterministicMockLLMAdapter(model=model)
    if provider_choice == "ollama":
        return OllamaLLMAdapter(host=host, model=model)

    # auto mode: try connecting to Ollama at host
    try:
        import urllib.request

        clean_host = host.rstrip("/")
        req = urllib.request.Request(f"{clean_host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            if resp.status == 200:
                return OllamaLLMAdapter(host=clean_host, model=model)
    except Exception:
        pass
    return DeterministicMockLLMAdapter(model=model)


def run_summarize(args: argparse.Namespace) -> int:
    if not args.chat and args.thread is None:
        print(
            "Error: Specify either --chat <chat_jid> or --thread <message_pk> to summarize.",
            file=sys.stderr,
        )
        return 1

    if not (0.0 <= args.temperature <= 0.1):
        print(
            f"Error: Summarization temperature must be in range [0.0, 0.1], got {args.temperature}",
            file=sys.stderr,
        )
        return 1

    host = _resolve_host(args)
    llm = _get_llm_adapter(args.llm, args.model, host=host)
    summarizer = Summarizer(llm=llm)

    if args.thread is not None:
        messages = []
        if args.source and args.source.exists():
            engine = IngestionEngine(args.source, args.contacts)
            messages, _ = engine.extract_messages(
                include_direct=True, include_groups=True
            )
        elif args.db and args.db.exists():
            with SearchIndex(args.db) as index:
                messages = index.get_all_messages()
        else:
            source, contacts = _resolve_source_and_contacts(args.source, args.contacts)
            if source:
                engine = IngestionEngine(source, contacts)
                messages, _ = engine.extract_messages(
                    include_direct=True, include_groups=True
                )

        if not messages:
            print("Error: No messages available to trace thread.", file=sys.stderr)
            return 1

        summary = summarizer.summarize_thread(
            messages=messages,
            root_pk=args.thread,
            temperature=args.temperature,
        )
    else:
        episodes = []
        if args.db and args.db.exists():
            with SearchIndex(args.db) as index:
                episodes = index.get_episodes_for_chat(args.chat)

        if not episodes:
            source, contacts = _resolve_source_and_contacts(args.source, args.contacts)
            if source:
                engine = IngestionEngine(source, contacts)
                messages, _ = engine.extract_messages(
                    include_direct=True, include_groups=True
                )
                chat_msgs = [m for m in messages if m.chat_jid == args.chat]
                if chat_msgs:
                    episodes = chunk_into_episodes(chat_msgs)

        if not episodes:
            print(f"Error: No episodes found for chat '{args.chat}'.", file=sys.stderr)
            return 1

        summary = summarizer.summarize(
            episodes=episodes,
            intent_notes=args.intent,
            temperature=args.temperature,
        )

    print("=" * 80)
    print(f"VERIFIED FACTUAL SUMMARY (Prefix NLI Grounded): {summary.title}")
    print(
        f"Chat: {summary.chat_jid} | Episodes: {summary.episodes_analyzed} | Messages: {summary.messages_analyzed}"
    )
    print("=" * 80 + "\n")
    print(summary.summary_text + "\n")
    print("-" * 80)
    print("VERIFICATION AUDIT REPORT")
    print(f"- Claims Entailed & Verified : {len(summary.claims)}")
    print(f"- Claims Discarded           : {len(summary.discarded_claims)}")
    if summary.discarded_claims:
        print("  Discarded Claims Audit:")
        for disc in summary.discarded_claims:
            print(f"    * [{disc.discard_reason}] {disc.text}")
    print("-" * 80)
    return 0


def run_draft(args: argparse.Namespace) -> int:
    review_box = DraftReviewBox(args.db)

    # 1. Review box operations
    if args.list_staged:
        staged_list = review_box.list_staged(args.chat)
        if not staged_list:
            print("No drafts currently staged for review.")
            return 0
        print("=" * 80)
        print("STAGED DRAFTS REVIEW BOX")
        print("=" * 80)
        for s in staged_list:
            print(f"Draft ID : {s.draft_id} | Created: {s.iso_date}")
            print(f"Chat     : {s.chat_jid}")
            print(f"Intent   : {', '.join(s.intent_bullets)}")
            print("Proposed Draft:")
            print(s.draft_text)
            print("-" * 80)
        return 0

    if getattr(args, "delete", None):
        deleted = review_box.delete(args.delete)
        if deleted:
            print(f"Draft '{args.delete}' deleted.")
            return 0
        else:
            print(f"Draft '{args.delete}' not found.", file=sys.stderr)
            return 1

    if args.approve:
        try:
            draft = review_box.approve(args.approve)
            print(
                f"Draft '{draft.draft_id}' ready for manual copy-paste into WhatsApp:"
            )
            print("-" * 40)
            print(draft.draft_text)
            print("-" * 40)
            return 0
        except KeyError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    if args.reject:
        try:
            draft = review_box.reject(args.reject)
            print(f"Draft '{draft.draft_id}' rejected and deleted.")
            return 0
        except KeyError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    # 2. Validation of chat/thread requirements
    if not args.chat and args.thread is None:
        print(
            "Error: Specify either --chat <chat_jid> or --thread <message_pk> to draft a reply.",
            file=sys.stderr,
        )
        return 1

    if not (0.0 <= args.temperature <= 0.2):
        print(
            f"Error: Drafting temperature must be in range [0.0, 0.2], got {args.temperature}",
            file=sys.stderr,
        )
        return 1

    # 3. Retrieve messages
    all_messages: list[NormalizedMessage] = []
    if args.db and args.db.exists():
        with SearchIndex(args.db) as index:
            all_messages = index.get_all_messages()
    elif args.source and args.source.exists():
        engine = IngestionEngine(args.source, args.contacts)
        all_messages, _ = engine.extract_messages(
            include_direct=True, include_groups=True
        )
    else:
        source, contacts = _resolve_source_and_contacts(args.source, args.contacts)
        if source:
            engine = IngestionEngine(source, contacts)
            all_messages, _ = engine.extract_messages(
                include_direct=True, include_groups=True
            )

    if not all_messages:
        print(
            "Error: No messages available in index or storage to draft reply.",
            file=sys.stderr,
        )
        return 1

    # Filter thread messages
    chat_jid = args.chat
    thread_messages: list[NormalizedMessage] = []
    if args.thread is not None:
        dag = ReplyDAG(all_messages)
        if args.thread not in dag.messages:
            print(
                f"Error: Message #{args.thread} not found to trace thread.",
                file=sys.stderr,
            )
            return 1
        thread_messages = dag.get_thread_messages(args.thread)
        if not chat_jid and thread_messages:
            chat_jid = thread_messages[0].chat_jid
    else:
        thread_messages = [m for m in all_messages if m.chat_jid == chat_jid]

    if not thread_messages:
        print(f"Error: No messages found for chat '{chat_jid}'.", file=sys.stderr)
        return 1

    # 4. Outgoing messages from user (is_from_me = 1) in that chat
    outgoing_history = [
        m for m in all_messages if m.chat_jid == chat_jid and m.is_from_me
    ]
    outgoing_history = outgoing_history[-10:]

    # 5. Detect unanswered questions
    unanswered = detect_unanswered_questions(thread_messages)

    # 6. Gather intent bullets
    intent_bullets = list(args.intent) if args.intent else []
    if not intent_bullets:
        if sys.stdin.isatty():
            print("\nIdentified unanswered question(s) in active thread:")
            if unanswered:
                for idx, uq in enumerate(unanswered, 1):
                    print(
                        f'  [{idx}] {uq.sender_name} (msg #{uq.message_pk}): "{uq.question_text}"'
                    )
            else:
                print("  (No direct questions found; drafting general response)")
            print(
                "\nEnter response intent points (one per line, empty line to finish):"
            )
            while True:
                try:
                    line = input("> ").strip()
                    if not line:
                        break
                    intent_bullets.append(line)
                except (EOFError, KeyboardInterrupt):
                    break
        if not intent_bullets:
            print(
                "Error: At least one intent bullet point is required to draft a reply (--intent / -i).",
                file=sys.stderr,
            )
            return 1

    # 7. Synthesize draft
    host = _resolve_host(args)
    llm = _get_llm_adapter(args.llm, args.model, host=host)
    drafter = ReplyDrafter(llm=llm, review_box=review_box)

    staged, verification = drafter.draft_reply(
        chat_jid=chat_jid,
        thread_messages=thread_messages,
        outgoing_history=outgoing_history,
        intent_bullets=intent_bullets,
        thread_root_pk=args.thread,
        temperature=args.temperature,
    )

    # 8. Render Staged Review Box
    print("=" * 80)
    print(
        "STAGED DRAFT REVIEW BOX (Manual Human Review Required - Autonomous Sending Prohibited)"
    )
    print(f"Draft ID: {staged.draft_id} | Chat: {staged.chat_jid}")
    print("=" * 80)
    print(
        f"Tone Formality : {staged.tone_profile.formality.value.capitalize()} (Learned from {staged.tone_profile.sample_count} recent outgoing messages)"
    )
    print(f"Greeting Style : {staged.tone_profile.typical_greeting or 'None'}")
    print(f"Sign-off Style : {staged.tone_profile.typical_signoff or 'None'}")
    print("\nUser Intent Bullets Addressed:")
    for b in staged.intent_bullets:
        print(f"  • {b}")

    if staged.unanswered_questions:
        print("\nQuestions Addressed:")
        for u in staged.unanswered_questions:
            print(f'  • {u.sender_name}: "{u.question_text}"')

    print("\n" + "-" * 80)
    print("PROPOSED DRAFT:")
    print(staged.draft_text)
    print("-" * 80 + "\n")

    print("INTENT VERIFICATION AUDIT REPORT")
    print(
        f"- Grounding Status       : {'VERIFIED & GROUNDED' if verification.is_valid else 'VIOLATIONS DETECTED'}"
    )
    print(f"- Grounding Score        : {verification.grounding_score:.2f}")
    print(f"- Unrequested Commitments: {len(verification.violations)}")
    if verification.violations:
        for v in verification.violations:
            print(f"  * {v}")
    print(
        "- Autonomous Sending     : STRUCTURALLY PROHIBITED (Manual Clipboard Action Only)"
    )
    print("=" * 80)

    return 0 if verification.is_valid else 1


def run_config(args: argparse.Namespace) -> int:
    config_path = args.config if getattr(args, "config", None) else DEFAULT_CONFIG_PATH
    action = getattr(args, "action", "show")

    if action == "init":
        if config_path.exists():
            print(f"Config file already exists at '{config_path}'")
            return 0
        cfg = AppConfig()
        cfg.save(config_path)
        print(f"Initialized default configuration at '{config_path}'")
        return 0

    cfg = AppConfig.load(config_path)

    if action == "set":
        changed = False
        new_host = getattr(args, "ollama_host", None) or getattr(args, "host", None)
        if new_host:
            cfg.ollama_host = new_host
            changed = True
        if getattr(args, "embedding_model", None):
            cfg.default_embedding_model = args.embedding_model
            changed = True
        if getattr(args, "llm_model", None):
            cfg.default_llm_model = args.llm_model
            changed = True
        if getattr(args, "group_gap", None) is not None:
            cfg.group_burst_minutes = args.group_gap
            changed = True
        if getattr(args, "direct_gap", None) is not None:
            cfg.direct_burst_minutes = args.direct_gap
            changed = True
        if getattr(args, "db", None) is not None:
            cfg.db_path = args.db
            changed = True
        if getattr(args, "vault", None) is not None:
            cfg.vault_path = args.vault
            changed = True

        if changed:
            cfg.save(config_path)
            print(f"Updated configuration saved to '{config_path}'")
        else:
            print("No configuration options provided to update.")
        return 0

    # action == "show"
    print("=" * 60)
    print("               TOTAL RECALL: CONFIGURATION                  ")
    print("=" * 60)
    print(f"Config File: {config_path} (exists: {config_path.exists()})")
    print(f"Database Path (db_path): {cfg.db_path}")
    print(f"Vault Config Path (vault_path): {cfg.vault_path}")
    print(f"Ollama Host (ollama_host): {cfg.ollama_host}")
    print(f"Default Embedding Model: {cfg.default_embedding_model}")
    print(f"Default LLM Model: {cfg.default_llm_model}")
    print(f"Group Burst Gap: {cfg.group_burst_minutes} minutes")
    print(f"Direct Burst Gap: {cfg.direct_burst_minutes} minutes")
    print("=" * 60)
    return 0


def run_mcp(args: argparse.Namespace) -> int:
    transport = getattr(args, "transport", "stdio")
    host = getattr(args, "host", "127.0.0.1")
    port = getattr(args, "port", 8765)
    token_str = getattr(args, "token", None)
    token_file = getattr(args, "token_file", DEFAULT_BRIDGE_TOKEN_PATH)
    db_path = getattr(args, "db", DEFAULT_DB_PATH)

    if not token_file.is_absolute() and db_path and db_path.is_absolute():
        token_file = db_path.parent / token_file

    if transport == "sse":
        try:
            validate_loopback_host(host)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    token = token_str or load_or_create_bridge_token(token_file)

    host_ollama = getattr(args, "host_ollama", None)
    if host_ollama and (
        host_ollama.startswith("http://") or host_ollama.startswith("https://")
    ):
        resolved_host = host_ollama
    else:
        resolved_host = _resolve_host(args)

    llm = _get_llm_adapter(
        getattr(args, "llm", "auto"),
        getattr(args, "model", "qwen2.5:7b"),
        host=resolved_host,
    )

    server = MCPServer(
        db_path=db_path,
        token=token,
        llm_adapter=llm,
        vault_path=getattr(args, "vault", None),
    )

    if transport == "stdio":
        MCPStdioTransport(server).run()
        return 0

    if transport == "sse":
        sse_server = MCPSSEServer(server, host=host, port=port)
        print(
            f"Total Recall MCP SSE Server running at http://{host}:{port}/sse (Auth token required)",
            file=sys.stderr,
        )
        print(f"Token file: {token_file}", file=sys.stderr)
        try:
            sse_server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            sse_server.shutdown_server()
        return 0

    print(f"Error: Unsupported transport '{transport}'", file=sys.stderr)
    return 1


def run_ui(args: argparse.Namespace) -> int:
    if args.build_only:
        print("Compiling native macOS menu bar app (WhatsAppTotalRecallBar)...")
        res = build_native_menu_bar_app()
        if res and res.exists():
            print(f"Success! Native app compiled at: {res}")
            return 0
        print("Error: Could not compile native menu bar app.", file=sys.stderr)
        return 1

    config = AppConfig.load()
    db_path = args.db or config.db_path
    vault_path = args.vault or config.vault_path

    try:
        url, server = run_ui_server(
            db_path=db_path,
            vault_path=vault_path,
            review_box_path=db_path,
            config=config,
            host=args.host,
            port=args.port,
            launch_desktop_app=not args.headless,
            open_browser=args.open_browser,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Total Recall Desktop UI daemon running at {url}")
    print("Press Ctrl+C to shut down daemon.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
    return 0


def _get_bundled_skill_content() -> str:
    """Read the bundled SKILL.md content from package resources or fallback locations."""
    # 1. Try importlib.resources
    try:
        from importlib.resources import files

        skill_ref = files("whatsapp_total_recall").joinpath(
            "skills/whatsapp-total-recall/SKILL.md"
        )
        if skill_ref.is_file():
            return skill_ref.read_text(encoding="utf-8")
    except Exception:
        pass

    # 2. Try relative to package directory
    pkg_skill = Path(__file__).parent / "skills" / "whatsapp-total-recall" / "SKILL.md"
    if pkg_skill.exists():
        return pkg_skill.read_text(encoding="utf-8")

    # 3. Try repo root skills/whatsapp-total-recall/SKILL.md
    repo_skill = (
        Path(__file__).resolve().parent.parent
        / "skills"
        / "whatsapp-total-recall"
        / "SKILL.md"
    )
    if repo_skill.exists():
        return repo_skill.read_text(encoding="utf-8")

    raise FileNotFoundError("Bundled whatsapp-total-recall SKILL.md not found.")


def run_install_skill(args: argparse.Namespace) -> int:
    try:
        content = _get_bundled_skill_content()
    except Exception as e:
        print(f"Error loading skill file: {e}", file=sys.stderr)
        return 1

    targets_map: dict[str, list[Path]] = {
        "claude": [
            Path.home() / ".claude" / "skills" / "whatsapp-total-recall" / "SKILL.md"
        ],
        "antigravity": [
            Path.home()
            / ".gemini"
            / "config"
            / "skills"
            / "whatsapp-total-recall"
            / "SKILL.md"
        ],
        "gemini": [
            Path.home()
            / ".gemini"
            / "config"
            / "skills"
            / "whatsapp-total-recall"
            / "SKILL.md"
        ],
        "cursor": [
            Path.cwd() / ".cursor" / "skills" / "whatsapp-total-recall" / "SKILL.md"
        ],
    }

    dest_paths: list[Path] = []
    if args.dest:
        dest_dir = Path(args.dest)
        dest_paths.append(
            dest_dir / "SKILL.md" if not str(dest_dir).endswith(".md") else dest_dir
        )
    elif args.target == "all":
        for key in ["claude", "antigravity", "cursor"]:
            dest_paths.extend(targets_map[key])
    elif args.target in targets_map:
        dest_paths.extend(targets_map[args.target])
    else:
        print(f"Error: Unknown target '{args.target}'", file=sys.stderr)
        return 1

    installed_count = 0
    for path in dest_paths:
        if path.exists() and not args.force:
            print(f"Skipping existing skill (use --force to overwrite): {path}")
            continue
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            print(f"Installed skill -> {path}")
            installed_count += 1
        except Exception as e:
            print(f"Failed to install to {path}: {e}", file=sys.stderr)

    print(f"Successfully installed skill to {installed_count} location(s).")
    return 0


def run_reset(args: argparse.Namespace) -> int:
    if not getattr(args, "force", False):
        try:
            choice = input(
                "Warning: This will irreversibly purge WhatsApp Total Recall databases, "
                "configurations, tokens, skills, and MCP registrations.\n"
                "Are you sure you want to proceed? [y/N]: "
            )
            if choice.strip().lower() not in {"y", "yes"}:
                print("Opt-out reset cancelled.")
                return 0
        except (EOFError, KeyboardInterrupt):
            print("\nOpt-out reset cancelled.")
            return 0

    purged_items: list[str] = []

    # 1. Database and local data files
    if not getattr(args, "keep_db", False):
        db_candidates: list[Path] = []
        if getattr(args, "db", None):
            db_candidates.append(Path(args.db))
        else:
            default_db = get_default_db_path()
            db_candidates.extend(
                [
                    default_db,
                    Path("whatsapp_total_recall.db"),
                    Path("total_recall.db"),
                ]
            )

        seen_dbs: set[Path] = set()
        for db_file in db_candidates:
            resolved = db_file.resolve()
            if resolved in seen_dbs or not db_file.exists():
                continue
            seen_dbs.add(resolved)
            try:
                with sqlite3.connect(db_file) as conn:
                    conn.execute("PRAGMA secure_delete = ON;")
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT name, type FROM sqlite_master WHERE type IN ('table', 'view', 'index');"
                    )
                    for name, item_type in cursor.fetchall():
                        if item_type == "table" and not name.startswith("sqlite_"):
                            cursor.execute(f'DROP TABLE IF EXISTS "{name}";')
                    conn.commit()
                    conn.execute("VACUUM;")
            except Exception:
                pass
            for suffix in ["", "-wal", "-shm"]:
                p = Path(str(db_file) + suffix)
                if p.exists():
                    p.unlink(missing_ok=True)
            purged_items.append(f"Database: {db_file}")

        # Wipe canonical directory ~/.whatsapp-total-recall
        storage_dir = get_canonical_storage_dir()
        if storage_dir.exists():
            shutil.rmtree(storage_dir, ignore_errors=True)
            purged_items.append(f"Directory: {storage_dir}")

        # Wipe local runtime config / token in CWD
        for local_name in ["vault_config.json", ".bridge-token", "config.json"]:
            lp = Path(local_name)
            if lp.exists():
                lp.unlink(missing_ok=True)
                purged_items.append(f"Config/Token: {lp}")

    # 2. Prune agent skills
    skill_dirs = [
        Path.home() / ".claude" / "skills" / "whatsapp-total-recall",
        Path.home() / ".gemini" / "config" / "skills" / "whatsapp-total-recall",
        Path.cwd() / ".cursor" / "skills" / "whatsapp-total-recall",
        Path.home() / ".cursor" / "skills" / "whatsapp-total-recall",
    ]
    for sdir in skill_dirs:
        if sdir.exists():
            shutil.rmtree(sdir, ignore_errors=True)
            purged_items.append(f"Skill: {sdir}")

    # 3. Prune MCP client config JSON files
    mcp_config_files = [
        Path.home()
        / "Library"
        / "Application Support"
        / "Claude"
        / "claude_desktop_config.json",
        Path.home() / ".cursor" / "mcp.json",
        Path.cwd() / ".cursor" / "mcp.json",
        Path.home() / ".codeium" / "windsurf" / "mcp_config.json",
    ]
    for cfg_file in mcp_config_files:
        if cfg_file.exists():
            try:
                data = json.loads(cfg_file.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "mcpServers" in data:
                    servers = data["mcpServers"]
                    if "whatsapp-total-recall" in servers:
                        del servers["whatsapp-total-recall"]
                        cfg_file.write_text(
                            json.dumps(data, indent=2), encoding="utf-8"
                        )
                        purged_items.append(f"MCP Config pruned: {cfg_file}")
            except Exception:
                pass

    # 4. Remove Claude Code CLI MCP registration
    if shutil.which("claude"):
        try:
            subprocess.run(
                ["claude", "mcp", "remove", "whatsapp-total-recall"],
                capture_output=True,
                check=False,
            )
            purged_items.append("Claude CLI MCP: whatsapp-total-recall removed")
        except Exception:
            pass

    print("WhatsApp Total Recall Reset Complete.")
    if purged_items:
        print(f"Purged {len(purged_items)} item(s):")
        for item in purged_items:
            print(f"  - {item}")
    else:
        print(
            "No active installations, databases, or MCP registrations found to purge."
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if (
        hasattr(args, "db")
        and args.db == DEFAULT_DB_PATH
        and not DEFAULT_DB_PATH.exists()
    ):
        args.db = get_default_db_path()
    if (
        hasattr(args, "vault")
        and args.vault == DEFAULT_VAULT_PATH
        and not DEFAULT_VAULT_PATH.exists()
    ):
        args.vault = get_default_vault_path()

    if args.command == "ingest":
        return run_ingest(args)
    if args.command == "sync":
        return run_sync(args)
    if args.command == "search":
        return run_search(args)
    if args.command == "thread":
        return run_thread(args)
    if args.command == "vault":
        return run_vault(args)
    if args.command == "episodes":
        return run_episodes(args)
    if args.command == "purge":
        return run_purge(args)
    if args.command == "summarize":
        return run_summarize(args)
    if args.command == "draft":
        return run_draft(args)
    if args.command == "config":
        return run_config(args)
    if args.command == "mcp":
        return run_mcp(args)
    if args.command == "ui":
        return run_ui(args)
    if args.command == "install-skill":
        return run_install_skill(args)
    if args.command == "reset":
        return run_reset(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
