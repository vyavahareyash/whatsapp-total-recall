from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from whatsapp_total_recall.config import (
    AppConfig,
    get_default_db_path,
    get_default_vault_path,
)
from whatsapp_total_recall.dag import ReplyDAG
from whatsapp_total_recall.discovery import discover_whatsapp_containers
from whatsapp_total_recall.drafter import DraftReviewBox, ReplyDrafter
from whatsapp_total_recall.ingestion import IngestionEngine
from whatsapp_total_recall.mcp import validate_loopback_host
from whatsapp_total_recall.search import SearchIndex
from whatsapp_total_recall.summarizer import (
    OllamaLLMAdapter,
    format_citation,
)
from whatsapp_total_recall.vault import DiscoveredChat, VaultConfig

logger = logging.getLogger("whatsapp_total_recall.ui")

STATIC_DIR = Path(__file__).parent / "static"
APPS_DIR = Path(__file__).parent.parent / "apps" / "WhatsAppTotalRecallBar"
FALLBACK_APPS_DIR = Path(__file__).parent.parent / "apps" / "TotalRecallBar"


class UIRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler serving both the desktop web assets and the REST API."""

    protocol_version = "HTTP/1.1"
    server_context: UIServerContext

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._serve_index_html()
        elif path == "/api/status":
            self._handle_get_status()
        elif path == "/api/models":
            self._handle_get_models()
        elif path == "/api/vault":
            self._handle_get_vault()
        elif path == "/api/search":
            q = query.get("q", [""])[0]
            limit_str = query.get("limit", ["10"])[0]
            try:
                limit = int(limit_str)
            except ValueError:
                limit = 10
            self._handle_search(q, limit)
        elif path.startswith("/api/thread/"):
            msg_id_str = path[len("/api/thread/") :]
            try:
                msg_id = int(msg_id_str)
                self._handle_get_thread(msg_id)
            except ValueError:
                self._send_json({"error": "Invalid message ID."}, status=400)
        elif path == "/api/drafts":
            self._handle_get_drafts()
        else:
            self._send_json({"error": f"Path '{path}' not found."}, status=404)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body_bytes = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            body = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except json.JSONDecodeError:
            self._send_json({"error": "Malformed JSON body."}, status=400)
            return

        if path == "/api/vault/approve":
            self._handle_vault_approve(body)
        elif path == "/api/vault/bulk":
            self._handle_vault_bulk(body)
        elif path == "/api/models":
            self._handle_set_model(body)
        elif path == "/api/vault/workspace":
            self._handle_vault_workspace(body)
        elif path == "/api/drafts/stage":
            self._handle_stage_draft(body)
        elif path.startswith("/api/drafts/") and path.endswith("/approve"):
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[1] == "drafts" and parts[3] == "approve":
                draft_id = parts[2]
                self._handle_approve_draft(draft_id)
            else:
                self._send_json({"error": "Invalid approve endpoint."}, status=400)
        elif path.startswith("/api/drafts/") and path.endswith("/reject"):
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[1] == "drafts" and parts[3] == "reject":
                draft_id = parts[2]
                self._handle_reject_draft(draft_id)
            else:
                self._send_json({"error": "Invalid reject endpoint."}, status=400)
        elif path.startswith("/api/drafts/") and path.endswith("/delete"):
            parts = path.strip("/").split("/")
            if len(parts) == 4 and parts[1] == "drafts" and parts[3] == "delete":
                draft_id = parts[2]
                self._handle_delete_draft(draft_id)
            else:
                self._send_json({"error": "Invalid delete endpoint."}, status=400)
        elif path == "/api/open-privacy-settings":
            self._handle_open_privacy_settings()
        else:
            self._send_json({"error": f"Path '{path}' not found."}, status=404)

    def do_DELETE(self) -> None:
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        if path.startswith("/api/drafts/"):
            parts = path.strip("/").split("/")
            if len(parts) == 3 and parts[1] == "drafts":
                draft_id = parts[2]
                self._handle_delete_draft(draft_id)
                return
        self._send_json({"error": f"Path '{path}' not found."}, status=404)

    def _serve_index_html(self) -> None:
        html_file = STATIC_DIR / "index.html"
        if html_file.exists():
            html_bytes = html_file.read_bytes()
        else:
            html_bytes = b"<html><body><h1>Total Recall</h1><p>UI assets missing.</p></body></html>"

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html_bytes)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(html_bytes)

    def _handle_get_status(self) -> None:
        containers = discover_whatsapp_containers()
        db_path = self.server_context.db_path

        indexed_messages = 0
        if db_path.exists():
            try:
                with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as con:
                    cur = con.cursor()
                    cur.execute("SELECT count(*) FROM messages")
                    row = cur.fetchone()
                    if row:
                        indexed_messages = int(row[0])
            except Exception:
                indexed_messages = 0

        vault = self.server_context.load_vault()
        approved_count = len(vault.approved_chat_jids)

        data = {
            "ok": True,
            "has_permission": containers.has_permission,
            "group_dir": str(containers.group_dir) if containers.group_dir else None,
            "chat_storage": (
                str(containers.chat_storage) if containers.chat_storage else None
            ),
            "contacts": str(containers.contacts) if containers.contacts else None,
            "indexed_messages": indexed_messages,
            "approved_chats": approved_count,
        }
        self._send_json(data)

    def _handle_get_models(self) -> None:
        models = self.server_context.get_available_models()
        self._send_json(
            {
                "ok": True,
                "current_model": self.server_context.active_model,
                "models": models,
            }
        )

    def _handle_set_model(self, body: dict[str, Any]) -> None:
        model = str(body.get("model", "")).strip()
        if not model:
            self._send_json({"error": "Missing 'model' parameter."}, status=400)
            return
        self.server_context.set_active_model(model)
        self._send_json(
            {
                "ok": True,
                "current_model": self.server_context.active_model,
            }
        )

    def _handle_get_vault(self) -> None:
        vault = self.server_context.load_vault()
        containers = discover_whatsapp_containers()

        discovered: list[DiscoveredChat] = []
        if (
            containers.chat_storage
            and containers.chat_storage.exists()
            and containers.has_permission
        ):
            try:
                engine = IngestionEngine(
                    chat_storage_path=containers.chat_storage,
                    contacts_path=containers.contacts,
                )
                report = engine.discover_chats(vault_config=vault)
                discovered = list(report.chats)
            except Exception as e:
                logger.warning(f"Error discovering chats: {e}")

        # If live discovery found 0 or wasn't available, fall back to database records
        if not discovered and self.server_context.db_path.exists():
            try:
                with sqlite3.connect(
                    f"file:{self.server_context.db_path}?mode=ro", uri=True
                ) as con:
                    cur = con.cursor()
                    cur.execute(
                        "SELECT m.chat_jid, coalesce(e.chat_name, m.chat_jid) as chat_name, "
                        "count(m.pk) as cnt, coalesce(min(m.chat_session_id), 1) as session_id "
                        "FROM messages m "
                        "LEFT JOIN episodes e ON e.chat_jid = m.chat_jid "
                        "GROUP BY m.chat_jid"
                    )
                    for r in cur.fetchall():
                        jid = str(r[0])
                        name = str(r[1])
                        count = int(r[2])
                        sess_id = int(r[3])
                        discovered.append(
                            DiscoveredChat(
                                chat_session_id=sess_id,
                                chat_jid=jid,
                                chat_name=name,
                                is_group=jid.endswith("@g.us"),
                                message_count=count,
                                is_approved=vault.is_chat_approved(jid),
                                workspaces=tuple(vault.get_workspaces_for_chat(jid)),
                            )
                        )
            except Exception:
                pass

        # Also ensure all vault-approved or workspace chats are surfaced
        known_jids = {c.chat_jid for c in discovered}
        for jid in vault.approved_chat_jids:
            if jid not in known_jids:
                discovered.append(
                    DiscoveredChat(
                        chat_session_id=len(discovered) + 1,
                        chat_jid=jid,
                        chat_name=jid.split("@")[0],
                        is_group=jid.endswith("@g.us"),
                        message_count=0,
                        is_approved=True,
                        workspaces=tuple(vault.get_workspaces_for_chat(jid)),
                    )
                )

        data = {
            "ok": True,
            "approved_chat_jids": sorted(vault.approved_chat_jids),
            "active_workspaces": sorted(vault.active_workspaces),
            "workspaces": {k: sorted(v) for k, v in sorted(vault.workspaces.items())},
            "discovered_chats": [
                {
                    "chat_session_id": c.chat_session_id,
                    "chat_jid": c.chat_jid,
                    "chat_name": c.chat_name,
                    "is_group": c.is_group,
                    "message_count": c.message_count,
                    "is_approved": c.is_approved,
                    "workspaces": list(c.workspaces),
                }
                for c in discovered
            ],
        }
        self._send_json(data)

    def _handle_vault_approve(self, body: dict[str, Any]) -> None:
        chat_jid = body.get("chat_jid", "").strip()
        approved = bool(body.get("approved", True))
        if not chat_jid:
            self._send_json({"error": "Missing 'chat_jid' parameter."}, status=400)
            return

        vault = self.server_context.load_vault()
        if approved:
            vault.approve(chat_jid)
        else:
            vault.revoke(chat_jid)
        self.server_context.save_vault(vault)

        self._send_json(
            {
                "ok": True,
                "chat_jid": chat_jid,
                "is_approved": vault.is_chat_approved(chat_jid),
                "approved_chat_jids": sorted(vault.approved_chat_jids),
            }
        )

    def _handle_vault_workspace(self, body: dict[str, Any]) -> None:
        name = body.get("name", "").strip()
        if not name:
            self._send_json({"error": "Missing workspace 'name'."}, status=400)
            return

        vault = self.server_context.load_vault()
        active = vault.toggle_workspace(name)
        self.server_context.save_vault(vault)

        self._send_json(
            {
                "ok": True,
                "name": name,
                "active": active,
                "active_workspaces": sorted(vault.active_workspaces),
            }
        )

    def _handle_vault_bulk(self, body: dict[str, Any]) -> None:
        approved = bool(body.get("approved", True))
        chat_jids = body.get("chat_jids", [])
        vault = self.server_context.load_vault()

        if approved:
            if chat_jids:
                for jid in chat_jids:
                    vault.approve(jid)
        else:
            if chat_jids:
                for jid in chat_jids:
                    vault.revoke(jid)
            else:
                vault.approved_chat_jids.clear()
                for ws in vault.workspaces.values():
                    ws.clear()
        self.server_context.save_vault(vault)

        self._send_json(
            {
                "ok": True,
                "approved": approved,
                "count": len(vault.approved_chat_jids),
                "approved_chat_jids": sorted(vault.approved_chat_jids),
            }
        )

    def _handle_search(self, query: str, limit: int) -> None:
        if not query:
            self._send_json({"ok": True, "count": 0, "results": []})
            return

        if not self.server_context.db_path.exists():
            self._send_json({"ok": True, "count": 0, "results": []})
            return

        vault = self.server_context.load_vault()
        candidate_limit = max(limit * 3, 30)

        with SearchIndex(self.server_context.db_path) as index:
            try:
                results = index.search_hybrid(query=query, limit=candidate_limit)
            except Exception:
                results = []
            if not results:
                results = index.search_episodes(query=query, limit=candidate_limit)

            formatted_results = []
            for scored in results:
                ep = scored.episode
                if not vault.is_chat_approved(ep.chat_jid):
                    continue
                citations = []
                for m in ep.messages[:5]:
                    cit_str = format_citation(
                        ep.chat_name, m.iso_date, m.sender_name, m.pk
                    )
                    citations.append(
                        {
                            "citation_str": cit_str,
                            "message_pk": m.pk,
                            "sender_name": m.sender_name,
                            "iso_date": m.iso_date,
                            "text": m.text,
                        }
                    )

                formatted_results.append(
                    {
                        "episode_id": ep.episode_id,
                        "chat_jid": ep.chat_jid,
                        "chat_name": ep.chat_name,
                        "start_iso": ep.start_iso,
                        "end_iso": ep.end_iso,
                        "rrf_score": scored.rrf_score,
                        "bm25_score": scored.bm25_score,
                        "header": ep.header,
                        "content": ep.content,
                        "citations": citations,
                        "message_ids": [m.pk for m in ep.messages],
                    }
                )
                if len(formatted_results) >= limit:
                    break

        self._send_json(
            {
                "ok": True,
                "query": query,
                "count": len(formatted_results),
                "results": formatted_results,
            }
        )

    def _handle_get_thread(self, message_id: int) -> None:
        if not self.server_context.db_path.exists():
            self._send_json(
                {"error": f"Database '{self.server_context.db_path}' not found."},
                status=404,
            )
            return

        vault = self.server_context.load_vault()
        with SearchIndex(self.server_context.db_path) as index:
            target_msg = index.get_message(message_id)
            if not target_msg:
                self._send_json(
                    {"error": f"Message #{message_id} not found."}, status=404
                )
                return

            if not vault.is_chat_approved(target_msg.chat_jid):
                self._send_json(
                    {
                        "error": f"Chat '{target_msg.chat_jid}' is not approved in Vault."
                    },
                    status=403,
                )
                return

            chat_messages = index.get_messages_for_chat(target_msg.chat_jid)
            dag = ReplyDAG(chat_messages)
            root = dag.get_root_for(message_id)
            thread_messages = dag.get_thread_messages(root.pk)
            rendered_tree = dag.render_thread_tree(root.pk)

            self._send_json(
                {
                    "ok": True,
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
            )

    def _handle_get_drafts(self) -> None:
        review_box = DraftReviewBox(self.server_context.review_box_path)
        drafts = review_box.list_staged()

        data = {
            "ok": True,
            "drafts": [
                {
                    "draft_id": d.draft_id,
                    "chat_jid": d.chat_jid,
                    "created_at": d.created_at,
                    "iso_date": d.iso_date,
                    "status": d.status.value,
                    "draft_text": d.draft_text,
                    "tone_profile": {
                        "formality": d.tone_profile.formality.value,
                        "greeting": d.tone_profile.typical_greeting,
                        "signoff": d.tone_profile.typical_signoff,
                    },
                    "tone_summary": f"{d.tone_profile.formality.value} (greeting: {d.tone_profile.typical_greeting}, signoff: {d.tone_profile.typical_signoff})",
                    "intent_bullets": list(d.intent_bullets),
                    "unanswered_questions": [u.text for u in d.unanswered_questions],
                }
                for d in drafts
            ],
        }
        self._send_json(data)

    def _handle_stage_draft(self, body: dict[str, Any]) -> None:
        chat_jid = body.get("chat_jid", "").strip()
        intent_bullets = body.get("intent_bullets", [])
        if isinstance(intent_bullets, str):
            intent_bullets = [intent_bullets]

        clean_intents = [b.strip() for b in intent_bullets if b and b.strip()]
        if not chat_jid:
            self._send_json({"error": "Missing 'chat_jid'."}, status=400)
            return
        if not clean_intents:
            self._send_json(
                {"error": "At least one intent bullet is required."}, status=400
            )
            return

        vault = self.server_context.load_vault()
        if not vault.is_chat_approved(chat_jid):
            self._send_json(
                {"error": f"Chat '{chat_jid}' is not approved in Opt-In Chat Vault."},
                status=403,
            )
            return

        chat_messages = []
        outgoing_history = []
        if self.server_context.db_path.exists():
            with SearchIndex(self.server_context.db_path) as index:
                chat_messages = index.get_messages_for_chat(chat_jid)
                outgoing_history = index.get_recent_outgoing_messages(
                    chat_jid=chat_jid, limit=10
                )

        review_box = DraftReviewBox(self.server_context.review_box_path)

        model_override = body.get("model", "").strip()
        if model_override:
            llm = OllamaLLMAdapter(
                host=self.server_context.config.ollama_host,
                model=model_override,
            )
        else:
            llm = self.server_context.llm_adapter

        drafter = ReplyDrafter(llm=llm, review_box=review_box)

        recent_thread = chat_messages[-25:] if chat_messages else []

        try:
            staged, verification = drafter.draft_reply(
                chat_jid=chat_jid,
                thread_messages=recent_thread,
                outgoing_history=outgoing_history,
                intent_bullets=clean_intents,
                temperature=0.0,
            )
        except Exception as exc:
            logger.error(f"Draft synthesis error: {exc}")
            self._send_json({"error": str(exc)}, status=502)
            return

        self._send_json(
            {
                "ok": True,
                "draft_id": staged.draft_id,
                "chat_jid": staged.chat_jid,
                "status": staged.status.value,
                "draft_text": staged.draft_text,
                "intent_bullets": list(staged.intent_bullets),
                "tone_profile": {
                    "formality": staged.tone_profile.formality.value,
                    "greeting": staged.tone_profile.typical_greeting,
                    "signoff": staged.tone_profile.typical_signoff,
                },
                "tone_summary": f"{staged.tone_profile.formality.value} (greeting: {staged.tone_profile.typical_greeting}, signoff: {staged.tone_profile.typical_signoff})",
            }
        )

    def _handle_approve_draft(self, draft_id: str) -> None:
        review_box = DraftReviewBox(self.server_context.review_box_path)
        try:
            draft = review_box.approve(draft_id)
            self._send_json(
                {
                    "ok": True,
                    "draft_id": draft.draft_id,
                    "status": draft.status.value,
                    "draft_text": draft.draft_text,
                }
            )
        except KeyError:
            self._send_json({"error": f"Draft '{draft_id}' not found."}, status=404)

    def _handle_reject_draft(self, draft_id: str) -> None:
        review_box = DraftReviewBox(self.server_context.review_box_path)
        try:
            draft = review_box.reject(draft_id)
            self._send_json(
                {
                    "ok": True,
                    "draft_id": draft.draft_id,
                    "deleted": True,
                }
            )
        except KeyError:
            self._send_json({"error": f"Draft '{draft_id}' not found."}, status=404)

    def _handle_delete_draft(self, draft_id: str) -> None:
        review_box = DraftReviewBox(self.server_context.review_box_path)
        deleted = review_box.delete(draft_id)
        if deleted:
            self._send_json({"ok": True, "draft_id": draft_id, "deleted": True})
        else:
            self._send_json({"error": f"Draft '{draft_id}' not found."}, status=404)

    def _handle_open_privacy_settings(self) -> None:
        if sys.platform == "darwin":
            subprocess.Popen(
                [
                    "open",
                    "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles",
                ]
            )
        self._send_json({"ok": True})

    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy standard HTTP access logs
        pass


class UIServerContext:
    """Shared application context passed to UI request handler."""

    def __init__(
        self,
        db_path: Path,
        vault_path: Path,
        review_box_path: Path,
        config: AppConfig | None = None,
        llm_adapter: Any = None,
    ) -> None:
        self.db_path = db_path
        self.vault_path = vault_path
        self.review_box_path = review_box_path
        self.config = config or AppConfig.load()
        self.active_model = self.config.default_llm_model
        self.llm_adapter = llm_adapter or self._init_default_llm()

    def _init_default_llm(self) -> Any:
        return OllamaLLMAdapter(
            host=self.config.ollama_host,
            model=self.active_model,
        )

    def get_available_models(self) -> list[str]:
        """Fetch list of locally installed models from Ollama."""
        host = self.config.ollama_host.rstrip("/")
        try:
            req = urllib.request.Request(
                f"{host}/api/tags", headers={"User-Agent": "TotalRecall"}
            )
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = []
                for m in data.get("models", []):
                    name = str(m.get("name", ""))
                    # Filter out dedicated embedding models
                    if "embed" in name.lower() or "bge" in name.lower():
                        continue
                    if name:
                        models.append(name)
                if models:
                    if self.active_model not in models:
                        self.set_active_model(models[0])
                    return models
        except Exception:
            pass
        return [self.active_model]

    def set_active_model(self, model_name: str) -> None:
        """Update runtime active LLM model."""
        self.active_model = model_name
        self.llm_adapter = OllamaLLMAdapter(
            host=self.config.ollama_host,
            model=model_name,
        )

    def load_vault(self) -> VaultConfig:
        return VaultConfig.load(self.vault_path)

    def save_vault(self, vault: VaultConfig) -> None:
        vault.save(self.vault_path)


class UIServer(ThreadingHTTPServer):
    """Threaded HTTP server serving the Total Recall desktop UI and API."""

    def __init__(
        self,
        server_address: tuple[str, int],
        context: UIServerContext,
    ) -> None:
        validate_loopback_host(server_address[0])
        self.context = context
        super().__init__(server_address, UIRequestHandler)

    def finish_request(self, request: Any, client_address: Any) -> None:
        self.RequestHandlerClass.server_context = self.context
        super().finish_request(request, client_address)


def build_native_menu_bar_app(
    source_dir: Path | None = None,
    output_path: Path | None = None,
) -> Path | None:
    """Compile the native macOS Swift menu bar application via swiftc."""
    if sys.platform != "darwin":
        return None

    swiftc = shutil.which("swiftc")
    if not swiftc:
        return None

    src_dir = source_dir or (APPS_DIR if APPS_DIR.exists() else FALLBACK_APPS_DIR)
    swift_source = src_dir / "main.swift"
    if not swift_source.exists():
        return None

    target = output_path or (src_dir / "WhatsAppTotalRecallBar")
    target.parent.mkdir(parents=True, exist_ok=True)

    cache_dir = Path("/tmp/swift-cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        swiftc,
        "-module-cache-path",
        str(cache_dir),
        "-O",
        str(swift_source),
        "-o",
        str(target),
        "-framework",
        "Cocoa",
        "-framework",
        "WebKit",
    ]

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode == 0 and target.exists():
            return target
        logger.warning(f"swiftc compilation failed: {proc.stderr}")
        return None
    except Exception as e:
        logger.warning(f"Could not build Swift menu bar app: {e}")
        return None


def run_ui_server(
    db_path: Path | None = None,
    vault_path: Path | None = None,
    review_box_path: Path | None = None,
    config: AppConfig | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    launch_desktop_app: bool = True,
    open_browser: bool = False,
    llm_adapter: Any = None,
) -> tuple[str, UIServer]:
    """Initialize and start the local Total Recall UI server."""
    validate_loopback_host(host)

    target_db = db_path if db_path is not None else get_default_db_path()
    target_vault = vault_path if vault_path is not None else get_default_vault_path()
    target_review = review_box_path if review_box_path is not None else target_db

    context = UIServerContext(
        db_path=target_db,
        vault_path=target_vault,
        review_box_path=target_review,
        config=config,
        llm_adapter=llm_adapter,
    )

    server = UIServer((host, port), context)
    actual_port = server.server_address[1]
    url = f"http://{host}:{actual_port}"

    if launch_desktop_app and sys.platform == "darwin":
        binary = (
            APPS_DIR / "WhatsAppTotalRecallBar"
            if (APPS_DIR / "WhatsAppTotalRecallBar").exists()
            else (APPS_DIR / "TotalRecallBar")
        )
        if not binary.exists():
            binary = build_native_menu_bar_app()
        if not binary or not binary.exists():
            candidates = [
                shutil.which("WhatsAppTotalRecallBar"),
                shutil.which("TotalRecallBar"),
                Path(
                    "/Applications/WhatsAppTotalRecallBar.app/Contents/MacOS/WhatsAppTotalRecallBar"
                ),
                Path("/Applications/TotalRecallBar.app/Contents/MacOS/TotalRecallBar"),
                Path.home()
                / "Applications/WhatsAppTotalRecallBar.app/Contents/MacOS/WhatsAppTotalRecallBar",
                Path.home()
                / "Applications/TotalRecallBar.app/Contents/MacOS/TotalRecallBar",
            ]
            for cand in candidates:
                if cand:
                    p = Path(cand)
                    if p.exists() and os.access(p, os.X_OK):
                        binary = p
                        break
        if binary and binary.exists() and os.access(binary, os.X_OK):
            try:
                subprocess.Popen([str(binary), url])
            except Exception as e:
                logger.warning(f"Could not spawn menu bar app: {e}")
                if open_browser:
                    import webbrowser

                    webbrowser.open(url)
        elif open_browser:
            import webbrowser

            webbrowser.open(url)
    elif open_browser:
        import webbrowser

        webbrowser.open(url)

    return url, server
