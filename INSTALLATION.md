# Installation & Setup Guide

This guide covers all installation methods, macOS permissions, local embedding setup, and AI assistant configurations for **WhatsApp Total Recall**.

---

## ⚡ Quick 1-Liner Agent Installation

If you are using an AI agent or coding assistant (Claude Code, Cursor, Windsurf, Antigravity, Roo Code), paste this single prompt into your chat:

```text
Please read https://raw.githubusercontent.com/vyavahareyash/whatsapp-total-recall/main/AGENT_INSTALL.md and install + configure WhatsApp Total Recall MCP for this environment.
```

---

## Prerequisites

1. **Operating System:** macOS 12.0+ (Apple Silicon or Intel).
2. **WhatsApp Desktop:** Official macOS desktop application (Catalyst build) installed and active.
3. **Full Disk Access:** Terminal / AI client granted Full Disk Access in macOS to read WhatsApp's App Group container.
4. **Python Runtime:** Python 3.11+ (if installing via CLI, uv, or pip).
5. **Local Inference (Optional):** [Ollama](https://ollama.com/) running locally for vector embeddings (`nomic-embed-text`) and summarization models (`qwen2.5:7b` / `llama3.2:3b`).

---

## Step 1: Grant macOS Full Disk Access

WhatsApp Catalyst stores its SQLite message database in a sandboxed App Group container:
```text
~/Library/Group Containers/group.net.whatsapp.WhatsApp.shared/ChatStorage.sqlite
```

Because macOS restricts access to group containers by default, you must grant **Full Disk Access (FDA)** to any application reading this database:

### 1. Open macOS Full Disk Access Settings
Run this command in Terminal to open the settings pane immediately:
```bash
open "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles"
```
*(Or navigate manually to **System Settings** > **Privacy & Security** > **Full Disk Access**).*

### 2. Enable Permissions
Toggle **ON** (or click **`+`** to add):
- **Terminal app** (**Terminal**, **iTerm2**, **Ghostty**, or **Warp**) — required for CLI commands and ingestion jobs.
- **AI client or IDE** (**Claude Desktop**, **Cursor**, **Windsurf**, or **VS Code**) — required when MCP servers are spawned by desktop assistants.
- **WhatsAppTotalRecallBar.app** (if running the menu bar app).

> [!IMPORTANT]
> **Restart Required:** After toggling Full Disk Access, you **must completely quit and restart** your terminal app or AI desktop client for macOS TCC permission changes to take effect.

### 3. Verify Permission via CLI
Run this quick diagnostic check:
```bash
test -r ~/Library/Group\ Containers/group.net.whatsapp.WhatsApp.shared/ChatStorage.sqlite && echo "✅ Full Disk Access granted" || echo "❌ Access denied - Grant Full Disk Access in macOS Settings"
```

---

## Step 2: Local Embedding Engine Setup (Ollama)

To enable dense semantic vector search alongside lexical BM25 (`sqlite-vec` RRF hybrid ranking):

```bash
# 1. Install Ollama (via Homebrew or ollama.com)
brew install ollama

# 2. Pull the default 768-dim embedding model
ollama pull nomic-embed-text

# 3. Ensure Ollama service is active
ollama serve
```

> [!NOTE]
> **Graceful Fallback:** If Ollama is not running, WhatsApp Total Recall operates in high-speed lexical-only mode using SQLite `FTS5` BM25 without interruption.

---

## Installation Methods

### Method A: Python CLI via `uv` or `pipx` (Recommended for CLI / Headless)

Install `whatsapp-total-recall` globally from [PyPI](https://pypi.org/project/whatsapp-total-recall/):

```bash
# Using uv (recommended)
uv tool install "whatsapp-total-recall[vec]"

# Or using pipx
pipx install "whatsapp-total-recall[vec]"

# Or standard pip inside an active virtual environment
pip install "whatsapp-total-recall[vec]"
```

> [!TIP]
> The `[vec]` extra installs `sqlite-vec>=0.1.6` for local SIMD-accelerated KNN vector search. Omit `[vec]` for a pure standard-library zero-dependency installation (BM25 lexical search only).
> Both `whatsapp-total-recall` and `total-recall` executable aliases are available.

---

### Method B: Standalone macOS Menu Bar App (GUI Users)

1. Download the latest `WhatsAppTotalRecall-macos.dmg` from [GitHub Releases](https://github.com/vyavahareyash/whatsapp-total-recall/releases/latest).
2. Open `WhatsAppTotalRecall-macos.dmg` and drag `WhatsAppTotalRecallBar.app` into `/Applications`.
3. Launch `WhatsAppTotalRecallBar.app`. An icon appears in your menu bar.
4. Ensure `WhatsAppTotalRecallBar.app` has Full Disk Access granted in macOS Settings.

---

### Method C: Headless Model Context Protocol (MCP) Setup

`WhatsAppTotalRecallBar.app` is **not** required to use the MCP server. MCP runs completely headless:

> [!TIP]
> You only need `uvx` (or the Python CLI) and a one-time message ingestion (`whatsapp-total-recall vault approve --all` followed by `whatsapp-total-recall ingest`). If invoking directly from Claude Desktop / Cursor, grant Full Disk Access to that application.
>
> **Where does it pull from?**
> * **PyPI (Standard):** `uvx --from "whatsapp-total-recall[vec]"` downloads from PyPI.
> * **Direct from GitHub (Bleeding-edge):** Replace `"whatsapp-total-recall[vec]"` with `"git+https://github.com/vyavahareyash/whatsapp-total-recall.git"`.
> * **Local Machine:** Use `"command": "whatsapp-total-recall"` if already installed locally via `uv tool install` or `pipx`.

#### Claude Desktop

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

*(Or use `"command": "whatsapp-total-recall"` if already installed globally).*

#### Cursor & Windsurf

In Cursor settings (`Settings > Features > MCP`) or `.cursor/mcp.json`:

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

#### Claude Code (CLI)

```bash
claude mcp add whatsapp-total-recall -- uvx --from "whatsapp-total-recall[vec]" whatsapp-total-recall mcp
```

#### Install Agent Skills (Claude Code, Antigravity, Cursor)

Install the bundled agent exploration skill across your local agent harnesses:
```bash
whatsapp-total-recall install-skill
```

#### SSE Transport for Web / Network Agents

```bash
# Starts SSE daemon with bearer bridge-token authentication
whatsapp-total-recall mcp --transport sse --port 8766
```

---

### Method D: Install From Source

```bash
# Clone the repository
git clone https://github.com/vyavahareyash/whatsapp-total-recall.git
cd whatsapp-total-recall

# Install dependencies using uv
uv sync --extra vec

# Or build the native menu bar DMG installer locally
make -C apps/WhatsAppTotalRecallBar dmg
```

---

## Post-Installation Quick Start

### 1. Configure the Opt-In Chat Vault
WhatsApp Total Recall enforces a strict privacy perimeter. You can approve all chats at once or pick specific contacts:

```bash
# Quick approve all discovered chats (Recommended for instant MCP use)
whatsapp-total-recall vault approve --all

# Or approve a specific contact or group JID
whatsapp-total-recall vault approve "123456789@s.whatsapp.net"

# Verify active approved chats
whatsapp-total-recall vault list-approved
```

### 2. Ingest Messages
Ingest messages from the local WhatsApp Catalyst storage into the local index:

```bash
# Ingest approved 1:1 chats
whatsapp-total-recall ingest

# Include approved group chats with reply DAG attribution
whatsapp-total-recall ingest --include-groups

# Or bypass vault checks entirely and ingest all chats directly:
whatsapp-total-recall ingest --all --include-groups
```

### 3. Incrementally Sync New Messages
After initial ingestion, update your local index with new messages without re-indexing your entire archive:

```bash
# Incrementally sync new messages across approved chats
whatsapp-total-recall sync

# Sync a specific conversation only
whatsapp-total-recall sync --chat <jid>

# Force a full backfill re-sync
whatsapp-total-recall sync --force
```

### 4. Search Conversations
Perform hybrid BM25 + vector search fused with Reciprocal Rank Fusion (RRF):

```bash
whatsapp-total-recall search "contract renewal deadline" --k 5
```

### 5. Launch Local Web UI & Menu Bar Companion
Start the local UI server (`http://127.0.0.1:8765`):

```bash
whatsapp-total-recall ui
```

---

## Troubleshooting & Reset

### Permission Denied (`Operation not permitted`)
If you see an error reading `ChatStorage.sqlite`:
1. Check Full Disk Access in **System Settings** > **Privacy & Security** > **Full Disk Access**.
2. Fully quit and restart your terminal app or AI client (`Cmd+Q`).
3. Verify via `test -r ~/Library/Group\ Containers/group.net.whatsapp.WhatsApp.shared/ChatStorage.sqlite && echo "OK"`.

### Database Reset & Secure Purge
To cleanly purge all indexed data, configuration, skills, and MCP registrations:

```bash
# Securely purge index database, configs, skills, and MCP registrations:
whatsapp-total-recall reset

# Bypass confirmation prompt:
whatsapp-total-recall reset --force
```
