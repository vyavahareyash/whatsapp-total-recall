# WhatsApp Total Recall

> **A Zero-Cost, Privacy-First Conversational Intelligence Engine for WhatsApp (macOS)**

WhatsApp Total Recall is an edge-native conversational intelligence platform designed to synthesize, search, and retrieve information across personal WhatsApp chats, group discussions, and multi-tiered communities on macOS—with **zero cloud costs, zero network packet egress, and zero account ban risk**.

---

## Key Highlights

* **Zero Ban Risk & Zero Pairing Friction:** Passively reads local unencrypted CoreData databases (`ChatStorage.sqlite`) from the macOS WhatsApp Catalyst App Group container. No companion QR pairing, no Puppeteer/Selenium sessions, and no risk of automated anti-bot bans.
* **Lock-Free Concurrency:** Uses immutable read-only connections (`mode=ro&immutable=1`) and snapshot isolation to prevent SQLite write-lock contention with WhatsApp Desktop's active WAL writer, avoiding OS `0xDEAD10CC` crashes.
* **Conversational Flow Graph (CFG):** Reconstructs multi-turn reply DAGs (`ZPARENTMESSAGE`), attributes group senders (`ZWAGROUPMEMBER` + `ContactsV2.sqlite`), and chunks fragmented burst chatter into coherent 5–25 message conversational episodes.
* **Zero-Cost Local Hybrid Search:** Single-file local SQLite storage (`total_recall.db`) combining keyword search (`FTS5` BM25) and dense semantic vector search (`sqlite-vec` SIMD KNN), fused via Reciprocal Rank Fusion (RRF).
* **Cryptographic Privacy & Epistemic Neutrality:** Mandatory hard purging on message revocation (`PRAGMA secure_delete = ON`), Opt-In Chat Vault perimeters, court-reporter citation ledgers (`[Chat | Date | Sender | msg_id]`), and prefix-level NLI verification (`MiniTruePrefixes`).
* **Dual Delivery (Single Binary):** Runs as a standalone macOS menu-bar / tray application (Wails / Tauri) while concurrently serving as a local Model Context Protocol (MCP) server for Claude Desktop, Cursor, and local AI agents.
* **Phase 2 Multimodal Roadmap:** On-device CPU voice note transcription via quantized `faster-whisper` + Silero VAD.

---

## Deep Module Architecture

The system is engineered around four deep modules with minimal interfaces and high leverage:

| Module | Interface Surface | Seam & Implementation |
| :--- | :--- | :--- |
| **IngestionEngine** | `sync(vaultConfig)`, `watch(vaultConfig)` | Local-substitutable; reads CoreData App Group containers; normalizes Apple Epoch (+978,307,200s); filters 32-bit sentinels; builds reply DAGs. |
| **SearchIndex** | `index(episodes)`, `search(query, k)`, `purge(msgId)` | Local-substitutable; manages FTS5 virtual tables, `sqlite-vec` SIMD KNN, SQL RRF scoring, and `secure_delete` purges. |
| **Summarizer** | `summarize(episodes, intentNotes?)` | Real seam (Ollama adapter vs test mock); enforces citation ledgers, sentiment normalization, and prefix-level NLI verification. |
| **AudioTranscriber** *(Phase 2)* | `transcribe(audioFilePath)` | Real seam (faster-whisper adapter vs test fake); handles Opus decoding, Silero VAD, and quantized speech-to-text. |

---

## Documentation Navigation

* **[CONTEXT.md](CONTEXT.md):** Problem space, macOS Catalyst constraints, user personas, and project boundaries.
* **[GLOSSARY.md](GLOSSARY.md):** Canonical domain language, strict definitions, and `_Avoid_` rules.
* **[docs/concepts.md](docs/concepts.md):** Deep explanation of core concepts: episodes, reply DAGs, Apple Epoch, sentinels, RRF, and epistemic neutrality.
* **[docs/architecture.md](docs/architecture.md):** Detailed macOS architecture, deep modules, seams, SQL RRF formulas, and concurrency mechanics.
* **[docs/spec.md](docs/spec.md):** Comprehensive product specification, 25 user stories, implementation decisions, and test plan.
* **[docs/adr/0001-macos-app-group-sqlite-ingestion.md](docs/adr/0001-macos-app-group-sqlite-ingestion.md):** Architecture Decision Record establishing macOS Catalyst SQLite snapshot ingestion.
* **[docs/research/research.md](docs/research/research.md):** Upstream forensic, architectural, benchmark, and protocol analysis from the research paper.


---

## Prerequisites

* **Operating System:** macOS 12.0+ (Apple Silicon or Intel).
* **WhatsApp Desktop:** Official macOS desktop application (Catalyst build) installed and active.
* **Full Disk Access:** Terminal / Python / IDE granted Full Disk Access in macOS to read WhatsApp's App Group container.
* **Python Runtime:** Python 3.11+ (if installing via CLI/uv/pip).
* **Local Inference (Optional):** [Ollama](https://ollama.com/) running locally for vector embeddings (`nomic-embed-text`) and summarization models (`qwen2.5:7b` / `llama3.2:3b`).

### Granting macOS Full Disk Access

WhatsApp Catalyst database files are located in `~/Library/Group Containers/group.net.whatsapp.WhatsApp.shared/ChatStorage.sqlite`. macOS requires Full Disk Access to read this container:

1. Open **System Settings** > **Privacy & Security** > **Full Disk Access**.
2. Unlock with your password or Touch ID.
3. Toggle ON (or click **+** to add) your terminal emulator (**Terminal**, **iTerm2**, **Ghostty**, or **Warp**), **WhatsAppTotalRecallBar.app**, and your AI client (e.g. **Claude Desktop**, **Cursor**, **VS Code**).
4. Verify permission in terminal:
   ```bash
   ls -la ~/Library/Group\ Containers/group.net.whatsapp.WhatsApp.shared/ChatStorage.sqlite
   ```

### Local Embedding Engine Setup (Ollama)

To enable dense semantic vector search alongside lexical BM25 (`sqlite-vec` RRF hybrid ranking):

```bash
# 1. Install Ollama (via Homebrew or ollama.com)
brew install ollama

# 2. Pull the default 768-dim embedding model
ollama pull nomic-embed-text

# 3. Ensure Ollama service is active
ollama serve
```

*(If Ollama is not running, WhatsApp Total Recall operates in high-speed lexical-only mode using SQLite `FTS5` BM25 without interruption).*

---

## Installation

### Option 1: Standalone macOS Menu Bar App (Recommended for GUI Users)

1. Download the latest `WhatsAppTotalRecall-macos.dmg` from [GitHub Releases](https://github.com/vyavahareyash/whatsapp-total-recall/releases).
2. Open `WhatsAppTotalRecall-macos.dmg` and drag `WhatsAppTotalRecallBar.app` into `/Applications`.
3. Launch `WhatsAppTotalRecallBar.app`. An icon appears in your menu bar.

### Option 2: Python CLI Installation via `uv` or `pipx`

Install the `whatsapp-total-recall` CLI globally from PyPI:

```bash
# Using uv (recommended)
uv tool install "whatsapp-total-recall[vec]"

# Or using pipx
pipx install "whatsapp-total-recall[vec]"

# Or inside an existing virtual environment
pip install "whatsapp-total-recall[vec]"
```

> [!NOTE]
> The `[vec]` extra installs `sqlite-vec>=0.1.6` for local SIMD-accelerated KNN vector search. Omit `[vec]` for a pure standard-library zero-dependency installation (BM25 lexical search only).
> Both `whatsapp-total-recall` and `total-recall` executable commands are available.

### Option 3: Direct MCP Server Install (for Claude Desktop & Cursor)

`WhatsAppTotalRecallBar.app` is **not** required to use the MCP server. MCP runs completely headless:

> [!TIP]
> You only need `uvx` (or the Python CLI) and a one-time message ingestion (`whatsapp-total-recall vault approve --all` followed by `whatsapp-total-recall ingest`). If invoking directly from Claude Desktop / Cursor, grant Full Disk Access to that application.
>
> **Where does it pull from?**
> * **PyPI (Standard):** `uvx --from "whatsapp-total-recall[vec]"` downloads from PyPI.
> * **Direct from GitHub (Pre-release / bleeding-edge):** Replace `"whatsapp-total-recall[vec]"` with `"git+https://github.com/vyavahareyash/whatsapp-total-recall.git"`.
> * **Local Machine:** Use `"command": "whatsapp-total-recall"` if already installed locally via `uv tool install` or `pipx`.

Configure WhatsApp Total Recall directly in your AI client without manual virtualenv management using `uvx`:

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

*(Or use `"command": "whatsapp-total-recall"` if already installed globally via `uv tool install`)*

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

### Option 4: From Source

```bash
git clone https://github.com/vyavahareyash/whatsapp-total-recall.git
cd whatsapp-total-recall

# Install dependencies using uv
uv sync --extra vec

# Or build the native menu bar DMG installer
make -C apps/WhatsAppTotalRecallBar dmg
```

---

## Quick Start

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

Ingest messages from the local WhatsApp Catalyst storage into the local encrypted SQLite index:

```bash
# Ingest approved 1:1 chats
whatsapp-total-recall ingest

# Include approved group chats with reply DAG attribution
whatsapp-total-recall ingest --include-groups

# Or bypass vault checks entirely and ingest all chats directly:
whatsapp-total-recall ingest --all --include-groups
```

### 3. Search Conversations

Perform hybrid BM25 + vector search fused with Reciprocal Rank Fusion (RRF):

```bash
whatsapp-total-recall search "contract renewal deadline" --k 5
```

### 4. Launch Desktop Menu Bar & Local UI

Start the local UI server and native menu bar companion (`http://127.0.0.1:8765`):

```bash
whatsapp-total-recall ui
```

### 5. Connect as an MCP Server

WhatsApp Total Recall provides a Model Context Protocol (MCP) server for Claude Desktop, Cursor, and local agent orchestrators.

#### Claude Desktop Integration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "whatsapp-total-recall": {
      "command": "whatsapp-total-recall",
      "args": ["mcp"]
    }
  }
}
```

#### Exposed MCP Tools

Once connected, your AI assistant gains access to five privacy-scoped tools:
* `list_chats`: Enumerate indexed conversations with JID, name, message count, and last active timestamp.
* `search_messages`: Hybrid lexical + vector search over approved chats with RRF ranking (with optional `chat_id` filter).
* `get_chat_history`: Retrieve sequential chronological message streams before/after a message ID or latest conversation turns.
* `get_thread_context`: Reconstruct reply DAGs and episode context around specific message IDs.
* `stage_reply_draft`: Create tone-matched drafts staged in the local Review Box (no autonomous sending).

#### Agent Skills Installation (Claude Code, Antigravity, Cursor)

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

## Release Pipeline

Total Recall includes an automated GitHub Actions release workflow (`.github/workflows/release.yml`):

* **Trigger:** Push a semantic tag (e.g. `git tag v0.1.0 && git push origin v0.1.0`) or run manually via `workflow_dispatch`.
* **Quality Gate:** Automated linting (`ruff`), format validation, and matrix test execution on Ubuntu & macOS.
* **Build Outputs:**
  * Universal macOS DMG installer (`WhatsAppTotalRecall-macos.dmg`) supporting both Apple Silicon (`arm64`) and Intel (`x86_64`).
  * Python Wheel (`.whl`) and Source Distribution (`.tar.gz`).
  * Cryptographic checksum ledger (`SHA256SUMS.txt`).
* **Publishing:** Creates GitHub Releases with changelogs and optionally publishes packages to PyPI via trusted publishing.

---

## License

MIT License. See LICENSE for details.
