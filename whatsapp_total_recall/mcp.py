from __future__ import annotations

import contextlib
import hmac
import http.server
import json
import os
import queue
import secrets
import sys
import threading
import urllib.parse
from pathlib import Path
from typing import Any

from whatsapp_total_recall.config import get_default_db_path
from whatsapp_total_recall.dag import ReplyDAG
from whatsapp_total_recall.drafter import (
    AutonomousSendingProhibitedError,
    DraftReviewBox,
    ReplyDrafter,
)
from whatsapp_total_recall.search import SearchIndex
from whatsapp_total_recall.summarizer import DeterministicMockLLMAdapter, LLMAdapter
from whatsapp_total_recall.vault import VaultConfig

DEFAULT_BRIDGE_TOKEN_PATH = Path(".bridge-token")
MCP_PROTOCOL_VERSION = "2024-11-05"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def validate_loopback_host(host: str) -> None:
    """Validate that the host binds exclusively to loopback (127.0.0.1 / localhost)."""
    clean_host = host.strip().lower()
    if clean_host not in LOOPBACK_HOSTS and not clean_host.startswith("127."):
        raise ValueError(
            f"Invalid MCP host '{host}'. MCP server must bind exclusively to 127.0.0.1 "
            "to prevent local network exposure."
        )


def load_or_create_bridge_token(path: Path | str | None = None) -> str:
    """Load existing bearer token from file or generate a cryptographically secure token with 0o600 permissions."""
    token_path = Path(path) if path is not None else DEFAULT_BRIDGE_TOKEN_PATH
    try:
        if token_path.exists():
            token = token_path.read_text(encoding="utf-8").strip()
            if token:
                return token

        token = secrets.token_hex(32)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        # Enforce POSIX 0o600 (read/write by owner only)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = os.open(str(token_path), flags, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(token + "\n")
        return token
    except OSError:
        # Fallback if spawned in read-only cwd (e.g. root '/' by desktop GUI clients)
        try:
            home_token = Path.home() / ".whatsapp-total-recall" / ".bridge-token"
            if not home_token.exists():
                fallback_old = Path.home() / ".total-recall" / ".bridge-token"
                if fallback_old.exists():
                    home_token = fallback_old
            if home_token.resolve() != token_path.resolve():
                return load_or_create_bridge_token(home_token)
        except OSError:
            pass
        return secrets.token_hex(32)


class MCPServer:
    """Core local-first Model Context Protocol (MCP) server engine."""

    def __init__(
        self,
        db_path: Path | str | None = None,
        token: str | None = None,
        token_path: Path | str | None = None,
        review_box_path: Path | str | None = None,
        llm_adapter: LLMAdapter | None = None,
        vault_path: Path | str | None = None,
    ):
        if db_path is not None:
            candidate = Path(db_path)
            if str(candidate) == "whatsapp_total_recall.db" and not candidate.exists():
                self.db_path = get_default_db_path()
            else:
                self.db_path = candidate
        else:
            self.db_path = get_default_db_path()

        self.review_box_path = (
            Path(review_box_path) if review_box_path else self.db_path
        )
        self.token = token or load_or_create_bridge_token(token_path)
        self.llm_adapter = llm_adapter or DeterministicMockLLMAdapter()
        self.vault_path = Path(vault_path) if vault_path else None

    def _get_vault_config(self) -> VaultConfig | None:
        if self.vault_path and self.vault_path.exists():
            try:
                return VaultConfig.load(self.vault_path)
            except Exception:
                pass
        return None

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return registered tool schemas. Autonomous send and presence tracking are stripped/omitted."""
        return [
            {
                "name": "list_chats",
                "description": (
                    "List indexed WhatsApp conversations with contact/group name, JID, message count, "
                    "and last activity timestamp. Use as step 1 in chat exploration to discover available "
                    "conversations, resolve JIDs, or find specific people/groups."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Optional search term to filter chat names or JIDs.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of chats to return (default: 20, max: 100).",
                            "default": 20,
                        },
                    },
                },
            },
            {
                "name": "search_messages",
                "description": (
                    "Search indexed WhatsApp messages using local full-text BM25 retrieval. "
                    "Returns matching message PKs, senders, timestamps, and text. Use this to locate key anchor "
                    "messages or keywords. You can optionally scope the search to a specific chat_id. "
                    "Follow up with get_chat_history to read chronological context around matching messages."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query terms or keywords.",
                        },
                        "chat_id": {
                            "type": "string",
                            "description": "Optional chat JID to scope search strictly within one conversation.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of messages to return (default: 10, max: 100).",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_conversations",
                "description": (
                    "Semantic hybrid search (RRF: BM25 lexical + dense vector embeddings) across multi-turn conversational flow episodes. "
                    "Returns cohesive dialogue bursts with summaries, intent notes, timestamps, and participants. "
                    "Use this as the primary semantic RAG tool for conceptual questions, decisions, topics, or plans "
                    "where exact keywords may not match raw messages."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query, topic, or conceptual question to search.",
                        },
                        "chat_id": {
                            "type": "string",
                            "description": "Optional chat JID to scope semantic search strictly within one conversation.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of conversational episodes to return (default: 5, max: 20).",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_chat_history",
                "description": (
                    "Retrieve a sequential chronological message stream for a specific chat. "
                    "Pass around_message_id to expand context before and after a search hit, or omit it to fetch "
                    "the most recent messages. Unlike get_thread_context (which only traces quoted reply DAGs), "
                    "get_chat_history reveals natural, unquoted back-and-forth dialogue, agreements, and context."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "chat_id": {
                            "type": "string",
                            "description": "The target chat JID.",
                        },
                        "around_message_id": {
                            "type": "integer",
                            "description": "Optional message primary key ID. If provided, fetches a chronological window centered on this message.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of chronological messages to return (default: 20, max: 100).",
                            "default": 20,
                        },
                    },
                    "required": ["chat_id"],
                },
            },
            {
                "name": "get_thread_context",
                "description": (
                    "Retrieve the full parent-child reply DAG (Directed Acyclic Graph) for a given message ID. "
                    "Traces quoted reply roots, branches, and discussion hierarchies. Use when a message references "
                    "an earlier comment via WhatsApp's quote-reply feature or during complex multi-turn group debates."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message_id": {
                            "type": "integer",
                            "description": "The message primary key ID to trace.",
                        },
                    },
                    "required": ["message_id"],
                },
            },
            {
                "name": "stage_reply_draft",
                "description": (
                    "Synthesize and stage an intent-constrained reply draft matching user's historical tone. "
                    "Drafts are staged strictly in the local review box for manual human approval. Autonomous sending is prohibited."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "chat_id": {
                            "type": "string",
                            "description": "The target chat JID to reply to.",
                        },
                        "intent_points": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of intent bullet points strictly bounding reply content.",
                        },
                    },
                    "required": ["chat_id", "intent_points"],
                },
            },
        ]

    def verify_token(self, token: str | None) -> bool:
        """Verify bearer token using constant-time comparison."""
        if not token:
            return False
        return hmac.compare_digest(token.strip(), self.token.strip())

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch tool execution with strict safety boundaries."""
        # Safety guard: Autonomous send and presence tracking methods are stripped/omitted
        prohibited_methods = {
            "send_message",
            "send_reply",
            "stage_and_send",
            "set_presence",
            "track_presence",
            "get_presence",
            "get_online_status",
        }
        if name in prohibited_methods:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Safety Violation: Tool '{name}' is strictly prohibited. "
                            "Autonomous sending and presence tracking methods are omitted by system architecture "
                            "to prevent WhatsApp account bans and user surveillance."
                        ),
                    }
                ],
                "isError": True,
            }

        try:
            if name == "list_chats":
                query = str(arguments.get("query", "")).strip()
                limit = int(arguments.get("limit", 20))
                return self._tool_list_chats(query, limit)
            elif name == "search_messages":
                query = str(arguments.get("query", "")).strip()
                chat_id = arguments.get("chat_id")
                limit = int(arguments.get("limit", 10))
                return self._tool_search_messages(query, limit, chat_id=chat_id)
            elif name == "search_conversations":
                query = str(arguments.get("query", "")).strip()
                chat_id = arguments.get("chat_id")
                limit = int(arguments.get("limit", 5))
                return self._tool_search_conversations(query, limit, chat_id=chat_id)
            elif name == "get_chat_history":
                if "chat_id" not in arguments:
                    raise ValueError("Argument 'chat_id' is required.")
                chat_id = str(arguments["chat_id"]).strip()
                around_id = arguments.get("around_message_id")
                around_message_id = int(around_id) if around_id is not None else None
                limit = int(arguments.get("limit", 20))
                return self._tool_get_chat_history(
                    chat_id, around_message_id=around_message_id, limit=limit
                )
            elif name == "get_thread_context":
                if "message_id" not in arguments:
                    raise ValueError("Argument 'message_id' is required.")
                message_id = int(arguments["message_id"])
                return self._tool_get_thread_context(message_id)
            elif name == "stage_reply_draft":
                chat_id = str(arguments.get("chat_id", "")).strip()
                intent_points = arguments.get("intent_points", [])
                if isinstance(intent_points, str):
                    intent_points = [intent_points]
                return self._tool_stage_reply_draft(chat_id, list(intent_points))
            else:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error: Unknown tool '{name}'.",
                        }
                    ],
                    "isError": True,
                }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Error executing '{name}': {e}"}],
                "isError": True,
            }

    def _tool_list_chats(self, query: str = "", limit: int = 20) -> dict[str, Any]:
        limit = max(1, min(limit, 100))
        if not self.db_path.exists():
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"error": f"Database '{self.db_path}' not found."}
                        ),
                    }
                ],
                "isError": True,
            }

        with SearchIndex(self.db_path) as index:
            cur = index.con.cursor()
            cur.execute(
                """
                SELECT
                    m.chat_jid,
                    coalesce(
                        (SELECT chat_name FROM episodes WHERE chat_jid = m.chat_jid AND chat_name IS NOT NULL AND chat_name != '' LIMIT 1),
                        (SELECT sender_name FROM messages WHERE chat_jid = m.chat_jid AND sender_name IS NOT NULL AND is_from_me = 0 ORDER BY unix_timestamp DESC LIMIT 1),
                        m.chat_jid
                    ) AS chat_name,
                    count(m.pk) AS message_count,
                    max(m.unix_timestamp) AS last_timestamp,
                    max(m.iso_date) AS last_iso_date
                FROM messages m
                GROUP BY m.chat_jid
                ORDER BY last_timestamp DESC
                """
            )
            rows = cur.fetchall()

            vault = self._get_vault_config()
            chats = []
            for r in rows:
                chat_jid = str(r["chat_jid"])
                if vault is not None and not vault.is_chat_approved(chat_jid):
                    continue
                name = str(r["chat_name"])
                if query:
                    q_lower = query.lower()
                    if q_lower not in chat_jid.lower() and q_lower not in name.lower():
                        continue
                chats.append(
                    {
                        "chat_jid": chat_jid,
                        "chat_name": name,
                        "is_group": "@g.us" in chat_jid,
                        "message_count": int(r["message_count"]),
                        "last_active": str(r["last_iso_date"]),
                    }
                )
                if len(chats) >= limit:
                    break

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(chats, indent=2, ensure_ascii=False),
                    }
                ],
                "isError": False,
            }

    def _tool_search_messages(
        self, query: str, limit: int, chat_id: str | None = None
    ) -> dict[str, Any]:
        if not query:
            return {
                "content": [{"type": "text", "text": json.dumps([])}],
                "isError": False,
            }
        limit = max(1, min(limit, 100))
        if not self.db_path.exists():
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"error": f"Database '{self.db_path}' not found."}
                        ),
                    }
                ],
                "isError": True,
            }

        with SearchIndex(self.db_path) as index:
            results = index.search(query=query, limit=max(limit * 5, 50))
            vault = self._get_vault_config()
            if vault is not None:
                results = [r for r in results if vault.is_chat_approved(r.chat_jid)]
            if chat_id:
                clean_chat = chat_id.strip()
                results = [r for r in results if r.chat_jid == clean_chat]
            results = results[:limit]
            data = [
                {
                    "message_pk": r.message_pk,
                    "chat_jid": r.chat_jid,
                    "sender_name": r.sender_name,
                    "iso_date": r.iso_date,
                    "text": r.text,
                    "bm25_score": round(r.bm25_score, 4),
                }
                for r in results
            ]
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(data, indent=2, ensure_ascii=False),
                    }
                ],
                "isError": False,
            }

    def _tool_search_conversations(
        self, query: str, limit: int = 5, chat_id: str | None = None
    ) -> dict[str, Any]:
        if not query:
            return {
                "content": [{"type": "text", "text": json.dumps([])}],
                "isError": False,
            }
        limit = max(1, min(limit, 20))
        if not self.db_path.exists():
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"error": f"Database '{self.db_path}' not found."}
                        ),
                    }
                ],
                "isError": True,
            }

        with SearchIndex(self.db_path) as index:
            try:
                results = index.search_hybrid(query=query, limit=max(limit * 3, 20))
            except Exception:
                try:
                    results = index.search_episodes(
                        query=query, limit=max(limit * 3, 20)
                    )
                except Exception:
                    results = []

            vault = self._get_vault_config()
            if vault is not None:
                results = [
                    r for r in results if vault.is_chat_approved(r.episode.chat_jid)
                ]
            if chat_id:
                clean_chat = chat_id.strip()
                results = [r for r in results if r.episode.chat_jid == clean_chat]

            results = results[:limit]
            data = [
                {
                    "episode_id": r.episode.episode_id,
                    "chat_jid": r.episode.chat_jid,
                    "chat_name": r.episode.chat_name,
                    "participants": list(r.episode.participants),
                    "start_date": r.episode.start_iso,
                    "end_date": r.episode.end_iso,
                    "header": r.episode.header,
                    "rrf_score": round(r.rrf_score, 4),
                    "dialogue": r.episode.content,
                }
                for r in results
            ]
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(data, indent=2, ensure_ascii=False),
                    }
                ],
                "isError": False,
            }

    def _tool_get_chat_history(
        self,
        chat_id: str,
        around_message_id: int | None = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        limit = max(1, min(limit, 100))
        if not self.db_path.exists():
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"error": f"Database '{self.db_path}' not found."}
                        ),
                    }
                ],
                "isError": True,
            }

        vault = self._get_vault_config()
        if vault is not None and not vault.is_chat_approved(chat_id):
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "error": f"Chat '{chat_id}' is not approved in Opt-In Chat Vault."
                            }
                        ),
                    }
                ],
                "isError": True,
            }

        with SearchIndex(self.db_path) as index:
            cur = index.con.cursor()
            if around_message_id is not None:
                target = index.get_message(around_message_id)
                if not target:
                    return {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "error": f"Message #{around_message_id} not found."
                                    }
                                ),
                            }
                        ],
                        "isError": True,
                    }
                if target.chat_jid != chat_id:
                    chat_id = target.chat_jid
                    if vault is not None and not vault.is_chat_approved(chat_id):
                        return {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(
                                        {
                                            "error": f"Chat '{chat_id}' is not approved in Opt-In Chat Vault."
                                        }
                                    ),
                                }
                            ],
                            "isError": True,
                        }

                half = limit // 2
                cur.execute(
                    """
                    SELECT * FROM messages
                    WHERE chat_jid = ? AND (unix_timestamp < ? OR (unix_timestamp = ? AND pk < ?))
                    ORDER BY unix_timestamp DESC, pk DESC
                    LIMIT ?
                    """,
                    (
                        chat_id,
                        target.unix_timestamp,
                        target.unix_timestamp,
                        target.pk,
                        half,
                    ),
                )
                before_rows = cur.fetchall()
                before_msgs = [
                    index._row_to_normalized_message(r) for r in reversed(before_rows)
                ]

                cur.execute(
                    """
                    SELECT * FROM messages
                    WHERE chat_jid = ? AND (unix_timestamp > ? OR (unix_timestamp = ? AND pk >= ?))
                    ORDER BY unix_timestamp ASC, pk ASC
                    LIMIT ?
                    """,
                    (
                        chat_id,
                        target.unix_timestamp,
                        target.unix_timestamp,
                        target.pk,
                        limit - len(before_msgs),
                    ),
                )
                after_rows = cur.fetchall()
                after_msgs = [index._row_to_normalized_message(r) for r in after_rows]
                messages = before_msgs + after_msgs
            else:
                cur.execute(
                    """
                    SELECT * FROM messages
                    WHERE chat_jid = ?
                    ORDER BY unix_timestamp DESC, pk DESC
                    LIMIT ?
                    """,
                    (chat_id, limit),
                )
                rows = cur.fetchall()
                messages = [index._row_to_normalized_message(r) for r in reversed(rows)]

            payload = {
                "chat_jid": chat_id,
                "message_count": len(messages),
                "target_message_id": around_message_id,
                "messages": [
                    {
                        "pk": m.pk,
                        "sender_name": m.sender_name,
                        "is_from_me": m.is_from_me,
                        "parent_pk": m.parent_pk,
                        "iso_date": m.iso_date,
                        "text": m.text,
                        "is_target": (m.pk == around_message_id)
                        if around_message_id is not None
                        else False,
                    }
                    for m in messages
                ],
            }
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(payload, indent=2, ensure_ascii=False),
                    }
                ],
                "isError": False,
            }

    def _tool_get_thread_context(self, message_id: int) -> dict[str, Any]:
        if not self.db_path.exists():
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {"error": f"Database '{self.db_path}' not found."}
                        ),
                    }
                ],
                "isError": True,
            }

        with SearchIndex(self.db_path) as index:
            target_msg = index.get_message(message_id)
            if not target_msg:
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {"error": f"Message #{message_id} not found."}
                            ),
                        }
                    ],
                    "isError": True,
                }

            vault = self._get_vault_config()
            if vault is not None and not vault.is_chat_approved(target_msg.chat_jid):
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "error": f"Chat '{target_msg.chat_jid}' is not approved in Opt-In Chat Vault."
                                }
                            ),
                        }
                    ],
                    "isError": True,
                }

            chat_messages = index.get_messages_for_chat(target_msg.chat_jid)
            dag = ReplyDAG(chat_messages)
            root = dag.get_root_for(message_id)
            thread_messages = dag.get_thread_messages(root.pk)
            rendered_tree = dag.render_thread_tree(root.pk)

            payload = {
                "target_message_id": message_id,
                "chat_jid": target_msg.chat_jid,
                "root_id": root.pk,
                "thread_length": len(thread_messages),
                "rendered_tree": rendered_tree,
                "messages": [
                    {
                        "pk": m.pk,
                        "sender_name": m.sender_name,
                        "is_from_me": m.is_from_me,
                        "parent_pk": m.parent_pk,
                        "iso_date": m.iso_date,
                        "text": m.text,
                    }
                    for m in thread_messages
                ],
            }
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(payload, indent=2, ensure_ascii=False),
                    }
                ],
                "isError": False,
            }

    def _tool_stage_reply_draft(
        self, chat_id: str, intent_points: list[str]
    ) -> dict[str, Any]:
        if not chat_id:
            raise ValueError("Parameter 'chat_id' is required.")
        if not intent_points or not any(p.strip() for p in intent_points):
            raise ValueError("At least one intent point is required.")

        vault = self._get_vault_config()
        if vault is not None and not vault.is_chat_approved(chat_id):
            raise ValueError(f"Chat '{chat_id}' is not approved in Opt-In Chat Vault.")

        clean_intents = [p.strip() for p in intent_points if p.strip()]

        chat_messages = []
        outgoing_history = []
        if self.db_path.exists():
            with SearchIndex(self.db_path) as index:
                chat_messages = index.get_messages_for_chat(chat_id)
                outgoing_history = index.get_recent_outgoing_messages(
                    chat_jid=chat_id, limit=10
                )

        review_box = DraftReviewBox(self.review_box_path)
        drafter = ReplyDrafter(llm=self.llm_adapter, review_box=review_box)

        # Thread messages (last 25 or available)
        recent_thread = chat_messages[-25:] if chat_messages else []

        staged, verification = drafter.draft_reply(
            chat_jid=chat_id,
            thread_messages=recent_thread,
            outgoing_history=outgoing_history,
            intent_bullets=clean_intents,
            temperature=0.0,
        )

        # Safety verification: ensure review_box prohibits sending
        with contextlib.suppress(AutonomousSendingProhibitedError):
            review_box.send(staged.draft_id)

        response_data = {
            "draft_id": staged.draft_id,
            "chat_id": staged.chat_jid,
            "status": staged.status.value,
            "draft_text": staged.draft_text,
            "intent_bullets": list(staged.intent_bullets),
            "tone_profile": {
                "formality": staged.tone_profile.formality.value,
                "typical_greeting": staged.tone_profile.typical_greeting,
                "typical_signoff": staged.tone_profile.typical_signoff,
            },
            "autonomous_sending_prohibited": True,
            "notice": (
                "Draft has been staged in the local Review Box for manual approval. "
                "Autonomous message sending is structurally prohibited by system architecture."
            ),
        }
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(response_data, indent=2, ensure_ascii=False),
                }
            ],
            "isError": False,
        }

    def handle_message(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Handle incoming JSON-RPC 2.0 message."""
        if not isinstance(request, dict):
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {
                    "code": -32600,
                    "message": "Invalid Request: expected object",
                },
            }

        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}

        # Handle notifications (no response expected)
        if method == "notifications/initialized":
            return None

        if not method:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32600, "message": "Invalid Request: missing method"},
            }

        if method in {"initialize", "server/discover"}:
            client_version = (
                params.get("protocolVersion")
                or params.get("_meta", {}).get(
                    "io.modelcontextprotocol/protocolVersion"
                )
                or MCP_PROTOCOL_VERSION
            )
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": client_version,
                    "supportedVersions": ["2024-11-05", "2026-07-28"],
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "whatsapp-total-recall-mcp",
                        "version": "0.2.2",
                    },
                },
            }

        elif method == "ping":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {},
            }

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": self.get_tool_definitions()},
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments") or {}
            if not tool_name:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32602,
                        "message": "Invalid params: 'name' is required for tools/call",
                    },
                }

            result = self.execute_tool(tool_name, tool_args)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": result,
            }

        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: '{method}'",
                },
            }


class MCPStdioTransport:
    """Standard I/O (stdio) transport for MCP."""

    def __init__(self, server: MCPServer):
        self.server = server

    def run(
        self,
        reader: Any = None,
        writer: Any = None,
        error_writer: Any = None,
    ) -> None:
        """Process line-delimited JSON-RPC messages over stdio."""
        in_stream = reader or sys.stdin
        out_stream = writer or sys.stdout
        err_stream = error_writer or sys.stderr

        for line in in_stream:
            line_str = (
                line.strip() if isinstance(line, str) else line.decode("utf-8").strip()
            )
            if not line_str:
                continue

            try:
                msg = json.loads(line_str)
            except Exception as e:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {e}"},
                }
                out_stream.write(json.dumps(err_resp) + "\n")
                out_stream.flush()
                continue

            try:
                resp = self.server.handle_message(msg)
                if resp is not None:
                    out_stream.write(json.dumps(resp) + "\n")
                    out_stream.flush()
            except Exception as e:
                err_stream.write(f"Error handling MCP message: {e}\n")
                err_stream.flush()


class MCPHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    """HTTP Request Handler implementing MCP SSE and message routing with Bearer auth."""

    server: MCPSSEServer  # type hint for parent server
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress standard HTTP server access logs to stderr in production."""
        pass

    def _extract_token(self) -> str | None:
        """Extract bearer token from Authorization header or URL query string."""
        auth_header = self.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:].strip()

        # Check query parameter ?token=
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        if "token" in query and query["token"]:
            return query["token"][0].strip()

        return None

    def _send_unauthorized(self) -> None:
        self.close_connection = True
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("WWW-Authenticate", 'Bearer realm="whatsapp-total-recall-mcp"')
        self.send_header("Connection", "close")
        self.end_headers()
        err_payload = {
            "jsonrpc": "2.0",
            "error": {
                "code": -32000,
                "message": "Unauthorized: missing or invalid bearer token.",
            },
        }
        self.wfile.write(json.dumps(err_payload).encode("utf-8"))

    def do_GET(self) -> None:
        """Handle SSE connection setup and health checks."""
        token = self._extract_token()
        if not self.server.mcp_server.verify_token(token):
            self._send_unauthorized()
            return

        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/health":
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b'{"status":"ok","transport":"sse"}')
            return

        if parsed.path in {"/sse", "/"}:
            session_id = secrets.token_hex(16)
            session_queue: queue.Queue[str | None] = queue.Queue()
            self.server.register_session(session_id, session_queue)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            # Emit endpoint event
            endpoint_uri = f"/messages?session_id={session_id}"
            initial_event = f"event: endpoint\ndata: {endpoint_uri}\n\n"
            try:
                self.wfile.write(initial_event.encode("utf-8"))
                self.wfile.flush()

                while not self.server.is_stopped():
                    try:
                        msg_str = session_queue.get(timeout=0.5)
                        if msg_str is None:  # Shutdown sentinel
                            break
                        event_payload = f"event: message\ndata: {msg_str}\n\n"
                        self.wfile.write(event_payload.encode("utf-8"))
                        self.wfile.flush()
                    except queue.Empty:
                        continue
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                self.server.unregister_session(session_id)
            return

        self.close_connection = True
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        """Handle JSON-RPC messages routed via HTTP POST."""
        self.close_connection = True
        token = self._extract_token()
        if not self.server.mcp_server.verify_token(token):
            self._send_unauthorized()
            return

        parsed = urllib.parse.urlparse(self.path)
        if not parsed.path.startswith("/messages"):
            self.send_response(404)
            self.end_headers()
            return

        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode("utf-8")

        try:
            req_data = json.loads(body)
        except Exception as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": f"Parse error: {e}"},
                    }
                ).encode("utf-8")
            )
            return

        query_params = urllib.parse.parse_qs(parsed.query)
        session_id = query_params.get("session_id", [None])[0]

        resp = self.server.mcp_server.handle_message(req_data)

        # Dispatch response to active SSE stream if session exists
        if resp is not None:
            resp_str = json.dumps(resp)
            if session_id:
                self.server.send_to_session(session_id, resp_str)

            # Also return 200/202 with JSON-RPC payload in POST response for client compatibility
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(resp_str.encode("utf-8"))
        else:
            self.send_response(204)
            self.end_headers()


class MCPSSEServer(http.server.ThreadingHTTPServer):
    """Threading HTTP server bound strictly to loopback 127.0.0.1 for MCP SSE transport."""

    def __init__(
        self,
        mcp_server: MCPServer,
        host: str = "127.0.0.1",
        port: int = 8765,
    ):
        validate_loopback_host(host)
        self.mcp_server = mcp_server
        self._stopped = False
        self._serving = False
        self._sessions: dict[str, queue.Queue[str | None]] = {}
        self._lock = threading.Lock()
        super().__init__((host, port), MCPHTTPRequestHandler)

    def serve_forever(self, poll_interval: float = 0.5) -> None:
        self._serving = True
        try:
            super().serve_forever(poll_interval=poll_interval)
        finally:
            self._serving = False

    def is_stopped(self) -> bool:
        return self._stopped

    def register_session(self, session_id: str, q: queue.Queue[str | None]) -> None:
        with self._lock:
            self._sessions[session_id] = q

    def unregister_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def send_to_session(self, session_id: str, message: str) -> bool:
        with self._lock:
            q = self._sessions.get(session_id)
            if q is not None:
                q.put(message)
                return True
        return False

    def shutdown_server(self) -> None:
        """Signal all sessions and terminate HTTP listener."""
        self._stopped = True
        with self._lock:
            for q in self._sessions.values():
                q.put(None)
        if getattr(self, "_serving", False):
            self.shutdown()
        self.server_close()
