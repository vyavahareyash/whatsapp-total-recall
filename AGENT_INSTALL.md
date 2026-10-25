# Agent Setup Instructions: WhatsApp Total Recall

> **Agent Operating Directive:**
> 1. **User Visibility (Verbose Output):** Always inform the user in clear, plain language what step you are performing, what files/paths you are inspecting, and the outcome of each operation before moving to the next.
> 2. **Sensitive Action Approvals:** Always pause and ask for explicit user confirmation before executing any sensitive action (installing packages, modifying desktop client configs, downloading models, opening system settings, or wiping databases).
> 3. **Ollama Awareness:** Clearly explain local embedding options (hybrid RRF vector search vs. lexical-only BM25) and guide the user on Ollama requirements.

---

## Agent Setup Protocol

When instructed by the user to set up WhatsApp Total Recall, follow these phases sequentially:

---

### Phase 1: Environment & WhatsApp Catalyst Verification

**Visibility Output:** Explain to the user:
> *"I am checking your operating system and verifying that the official WhatsApp Catalyst desktop archive exists on this machine."*

1. Verify macOS:
   Ensure the host platform is macOS (`darwin`).
2. Verify WhatsApp App Group sandbox directory:
   ```bash
   test -d "$HOME/Library/Group Containers/group.net.whatsapp.WhatsApp.shared" && echo "FOUND" || echo "NOT_FOUND"
   ```
   - **If NOT_FOUND:** Halt setup and explain that WhatsApp Total Recall requires the official macOS desktop WhatsApp application (which creates the App Group container).

---

### Phase 2: Full Disk Access (FDA) Permission Check

**Visibility Output:** Explain to the user:
> *"Checking if the active terminal or assistant process has macOS Full Disk Access to read `ChatStorage.sqlite`."*

1. Test file readability:
   ```bash
   test -r "$HOME/Library/Group Containers/group.net.whatsapp.WhatsApp.shared/ChatStorage.sqlite" && echo "FDA_OK" || echo "FDA_DENIED"
   ```
2. **If `FDA_DENIED`:**
   > ⚠️ **SENSITIVE ACTION: Prompt User for Permission Before Opening Settings**
   Ask:
   > *"macOS blocks applications from reading WhatsApp message archives by default. Would you like me to open macOS System Settings > Full Disk Access so you can enable permission for this terminal/app?"*
   - Once the user confirms, run:
     ```bash
     open "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles"
     ```
   - Instruct the user:
     > *"1. In the opened window, toggle ON (or click `+` to add) your terminal app (Terminal, iTerm2, Ghostty) or your AI IDE (Cursor, Windsurf, Claude Desktop).\n2. **Important:** You must quit and restart the application after enabling this setting.\n3. Reply once restarted to resume setup."*
   - Once resumed, re-verify with `test -r`.

---

### Phase 3: Ollama & Semantic Vector Search Awareness

**Visibility Output:** Explain to the user:
> *"Checking local inference capabilities (Ollama) for dense vector search and hybrid RRF rankings."*

1. **Context for User:**
   - WhatsApp Total Recall features **Hybrid Search (Reciprocal Rank Fusion)**:
     - **Lexical BM25 (`FTS5`):** Instant keyword matching. Always available with zero dependencies.
     - **Vector Search (`sqlite-vec` + Ollama `nomic-embed-text`):** High-accuracy semantic search understanding context, synonyms, and paraphrasing.
   - If Ollama is unavailable, the system automatically runs in fast BM25 lexical-only mode without breaking or throwing errors.

2. **Inspect Ollama State:**
   - Check binary:
     ```bash
     which ollama
     ```
   - Check if Ollama daemon is actively running:
     ```bash
     curl -s http://127.0.0.1:11434/api/tags
     ```
   - Check if `nomic-embed-text` embedding model is pulled:
     ```bash
     ollama list 2>/dev/null | grep -q "nomic-embed-text" && echo "MODEL_AVAILABLE" || echo "MODEL_MISSING"
     ```

3. **User Choice & Permission:**
   - **Case A (Ollama Running & Model Present):** Inform user that hybrid semantic search is fully ready.
   - **Case B (Ollama Installed but `nomic-embed-text` Missing):**
     > ⚠️ **SENSITIVE ACTION: Ask Permission to Download Model (~274MB)**
     > *"Ollama is installed, but the embedding model `nomic-embed-text` (~274 MB) is not present. Would you like me to pull it to enable semantic vector search, or continue with BM25 keyword search?"*
     If approved, run:
     ```bash
     ollama pull nomic-embed-text
     ```
   - **Case C (Ollama Not Installed or Daemon Stopped):**
     Inform user:
     > *"Ollama is currently not running or not installed. WhatsApp Total Recall will operate using SQLite BM25 keyword search. (Optional: You can install Ollama later via `brew install ollama && ollama pull nomic-embed-text` to enable vector search anytime)."*

---

### Phase 4: Package Installation

> ⚠️ **SENSITIVE ACTION: Ask User Permission to Install Global Tool**

**Visibility Output:** Announce to the user:
> *"I am ready to install `whatsapp-total-recall[vec]` from PyPI. This provides the CLI engine, local hybrid search, and MCP server."*

Ask:
> *"Would you like me to proceed with installing `whatsapp-total-recall[vec]` via `uv` (or `pipx`)?"*

Upon user approval:
```bash
# Preferred: uv
if command -v uv >/dev/null 2>&1; then
    uv tool install --force "whatsapp-total-recall[vec]"
# Fallback: pipx
elif command -v pipx >/dev/null 2>&1; then
    pipx install --force "whatsapp-total-recall[vec]"
# Fallback: standard pip
else
    pip install "whatsapp-total-recall[vec]"
fi
```

Verify CLI installation:
```bash
whatsapp-total-recall --help
```

---

### Phase 5: MCP Client Configuration

> ⚠️ **SENSITIVE ACTION: Ask User Permission Before Modifying Client Config Files**

**Visibility Output:** Inform user:
> *"I will now configure the `whatsapp-total-recall` Model Context Protocol (MCP) server so your AI assistant can query your WhatsApp history."*

Detect the target client and show the exact config changes:

#### Option A: Claude Code (CLI)
Ask: *"Add `whatsapp-total-recall` to Claude Code CLI MCP list?"*
If approved:
```bash
claude mcp add whatsapp-total-recall -- uvx --from "whatsapp-total-recall[vec]" whatsapp-total-recall mcp
```

#### Option B: Claude Desktop
Target file: `~/Library/Application Support/Claude/claude_desktop_config.json`
Show user the planned entry:
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
Ask: *"May I update `~/Library/Application Support/Claude/claude_desktop_config.json` with this configuration?"*
If approved, read existing JSON, merge `"whatsapp-total-recall"` under `"mcpServers"`, and save.

#### Option C: Cursor
Target file: `~/.cursor/mcp.json` or `.cursor/mcp.json`
Show user the planned JSON entry and ask permission before modifying.

#### Option D: Windsurf
Target file: `~/.codeium/windsurf/mcp_config.json`
Show user the planned JSON entry and ask permission before modifying.

---

### Phase 6: Agent Skill Installation

**Visibility Output:** Inform user:
> *"Installing the official Total Recall agent workflow skill into your agent harness so I can execute deep conversational search, DAG thread reconstruction, and review box staging."*

Run:
```bash
whatsapp-total-recall install-skill --force
```

---

### Phase 7: Verification & Final Report

**Visibility Output:** Run a quick health check and present a complete summary:
1. Verify tool responsiveness:
   ```bash
   whatsapp-total-recall config --help >/dev/null && echo "CLI_READY"
   ```
2. Report to user:
   - ✅ **Full Disk Access:** Confirmed active.
   - ✅ **Search Mode:** Hybrid RRF (BM25 + `nomic-embed-text`) OR Lexical BM25 (FTS5).
   - ✅ **MCP Tools Registered:**
     - `list_chats`: Enumerate conversations and message statistics.
     - `search_conversations`: Semantic hybrid RAG search over multi-turn episodes.
     - `search_messages`: Fast BM25 lexical keyword search across individual messages.
     - `get_chat_history`: Surrounding chronological message streams.
     - `get_thread_context`: Quote reply DAG attribution.
     - `stage_reply_draft`: Local tone-matched reply staging (never auto-sends).
     - `sync_chats`: Incremental delta sync of messages received since last sync date.
   - 💡 **Next Steps:** Suggest the user approve their desired chats via `whatsapp-total-recall vault approve --all`, run initial backfill with `whatsapp-total-recall ingest`, and use `whatsapp-total-recall sync` (or the `sync_chats` tool) for subsequent incremental updates.

---

## Opt-Out, Reset & Uninstallation Protocol

> ⚠️ **CRITICAL SENSITIVE ACTION: Permanent Data & Config Deletion**

If the user requests to remove, uninstall, or opt out of WhatsApp Total Recall:

1. **Verbose Warning & Permission:**
   Explicitly warn the user:
   > *"This will permanently wipe all indexed local SQLite databases, remove all configuration files and bridge tokens in `~/.whatsapp-total-recall/`, prune MCP server registrations from your AI clients (Claude Desktop, Cursor, Windsurf), and delete installed skills. Your original WhatsApp messages remain untouched in WhatsApp."*
   Ask:
   > *"Are you sure you want to completely purge and reset WhatsApp Total Recall? [y/N]"*

2. **Execute Cryptographic Reset:**
   Upon explicit affirmative confirmation:
   ```bash
   whatsapp-total-recall reset --force
   ```

3. **Uninstall Global Tool:**
   Ask permission to remove the package:
   ```bash
   uv tool uninstall whatsapp-total-recall 2>/dev/null || pipx uninstall whatsapp-total-recall 2>/dev/null || pip uninstall -y whatsapp-total-recall
   ```
