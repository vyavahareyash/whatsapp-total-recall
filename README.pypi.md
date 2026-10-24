# WhatsApp Total Recall

> **A Zero-Cost, Privacy-First Conversational Intelligence Engine & MCP Server for WhatsApp (macOS)**

[![PyPI](https://img.shields.io/pypi/v/whatsapp-total-recall)](https://pypi.org/project/whatsapp-total-recall/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/vyavahareyash/whatsapp-total-recall/blob/main/LICENSE)
[![GitHub Release](https://img.shields.io/github/v/release/vyavahareyash/whatsapp-total-recall)](https://github.com/vyavahareyash/whatsapp-total-recall/releases)

WhatsApp Total Recall is an edge-native conversational intelligence platform and Model Context Protocol (MCP) server for macOS. It passively reads local WhatsApp Catalyst SQLite archives (`ChatStorage.sqlite`) with **zero cloud egress, zero network traffic, and zero account ban risk**.

> [!TIP]
> **Looking for the native Mac Menu Bar App?**
> Download the pre-compiled **[WhatsAppTotalRecall-macos.dmg](https://github.com/vyavahareyash/whatsapp-total-recall/releases/latest)** installer directly from GitHub Releases.

---

## What This Package Delivers

Installing `whatsapp-total-recall` via `pip`, `pipx`, or `uv` provides:
1. **Model Context Protocol (MCP) Server:** Plug into Claude Desktop, Cursor, Windsurf, or Claude Code for live conversational search across personal and group chats.
2. **Local Hybrid Search Engine:** Single-file local SQLite storage combining keyword search (`FTS5` BM25) and dense semantic vector search (`sqlite-vec` SIMD KNN) via Reciprocal Rank Fusion (RRF).
3. **Privacy Vault Perimeter:** Opt-In Chat Vault with strict contact isolation and cryptographic revocation purges (`PRAGMA secure_delete`).
4. **Local Web UI & API Server:** Embedded zero-dependency web interface running locally on `http://127.0.0.1:8765`.

---

## Prerequisites: Granting macOS Full Disk Access

WhatsApp on macOS stores its SQLite message database in a protected App Group sandbox:
```text
~/Library/Group Containers/group.net.whatsapp.WhatsApp.shared/ChatStorage.sqlite
```

Because macOS restricts access to group containers by default, you must grant **Full Disk Access**:

1. Open macOS **System Settings** > **Privacy & Security** > **Full Disk Access**.
2. Click the **`+`** icon (or toggle **ON**) for:
   - Your Terminal app (**Terminal**, **iTerm2**, **Ghostty**, **Warp**).
   - Your AI desktop client if running MCP stdio (**Claude Desktop**, **Cursor**, **Windsurf**).
3. **Verify permission via CLI:**
   ```bash
   ls -la ~/Library/Group\ Containers/group.net.whatsapp.WhatsApp.shared/ChatStorage.sqlite
   ```
   *If the command prints file details, permission is active. If you see `Operation not permitted`, restart your terminal application.*

---

## Local Embeddings Requirement (Ollama)

WhatsApp Total Recall powers hybrid search (RRF) by combining lexical BM25 (`FTS5`) with dense vector embeddings (`sqlite-vec`, 768-dim):

1. **Install Ollama** (if not already installed):
   ```bash
   brew install ollama
   ```
   *(Or download installer from [ollama.com](https://ollama.com)).*

2. **Pull the default local embedding model:**
   ```bash
   ollama pull nomic-embed-text
   ```

3. Ensure Ollama is running in the background (`ollama serve` or Ollama app).

> [!NOTE]
> **Graceful Fallback:** If Ollama is unavailable, WhatsApp Total Recall automatically runs in fast lexical-only mode using SQLite `FTS5` BM25 keyword search without crashing.

---

## Installation

Install globally using `uv` (recommended) or `pipx`:

```bash
# With SIMD-accelerated vector search (recommended):
uv tool install "whatsapp-total-recall[vec]"

# Or via pipx:
pipx install "whatsapp-total-recall[vec]"

# Or standard pip:
pip install "whatsapp-total-recall[vec]"
```

---

## Model Context Protocol (MCP) Integration

Connect WhatsApp Total Recall directly to your AI desktop clients without manual environment setup:

### Claude Desktop
Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "whatsapp-total-recall": {
      "command": "uvx",
      "args": ["--from", "whatsapp-total-recall[vec]", "whatsapp-total-recall", "mcp"]
    }
  }
}
```

*(Or use `"command": "whatsapp-total-recall"` if installed globally).*

### Cursor & Windsurf
In Cursor (`Settings > Features > MCP`) or `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "whatsapp-total-recall": {
      "command": "uvx",
      "args": ["--from", "whatsapp-total-recall[vec]", "whatsapp-total-recall", "mcp"]
    }
  }
}
```

### Claude Code (CLI)
```bash
claude mcp add whatsapp-total-recall -- uvx --from "whatsapp-total-recall[vec]" whatsapp-total-recall mcp
```

### Exposed MCP Tools
* `list_chats`: Enumerate indexed conversations with JID, name, message count, and last active timestamp.
* `search_messages`: Hybrid lexical + vector search across approved conversations (with optional `chat_id` filter).
* `get_chat_history`: Retrieve sequential chronological message streams before/after a message ID or latest conversation turns.
* `get_thread_context`: Reconstruct multi-turn reply DAGs and conversation branches around specific messages.
* `stage_reply_draft`: Synthesize tone-matched reply drafts staged in the local Review Box (no autonomous sending).

### Agent Skills Installation (Claude Code, Antigravity, Cursor)
Install the official Total Recall skill to equip AI coding assistants with deep chat exploration workflows:
```bash
# Install to all detected agent harnesses (Claude Code, Antigravity, Cursor):
whatsapp-total-recall install-skill

# Or install to a specific agent harness:
whatsapp-total-recall install-skill --target claude
```

---

## CLI Quick Start

### 1. Approve Chats in Privacy Vault
```bash
# Quick approve all discovered WhatsApp chats at once:
whatsapp-total-recall vault approve --all

# Or approve a specific contact or group JID:
whatsapp-total-recall vault approve "123456789@s.whatsapp.net"
```

### 2. Ingest Messages
```bash
# Ingest approved 1:1 chats:
whatsapp-total-recall ingest

# Ingest approved group chats with reply DAG attribution:
whatsapp-total-recall ingest --include-groups
```

### 3. Search Locally
```bash
whatsapp-total-recall search "project deadline" --k 5
```

### 4. Launch Desktop Web UI
```bash
whatsapp-total-recall ui
```

---

## Links & Resources

* **GitHub Repository:** [https://github.com/vyavahareyash/whatsapp-total-recall](https://github.com/vyavahareyash/whatsapp-total-recall)
* **Mac Menu Bar DMG Installer:** [GitHub Releases](https://github.com/vyavahareyash/whatsapp-total-recall/releases)
* **Architecture Documentation:** [docs/architecture.md](https://github.com/vyavahareyash/whatsapp-total-recall/blob/main/docs/architecture.md)
* **Issues & Feedback:** [GitHub Issues](https://github.com/vyavahareyash/whatsapp-total-recall/issues)
* **License:** [MIT License](https://github.com/vyavahareyash/whatsapp-total-recall/blob/main/LICENSE)
