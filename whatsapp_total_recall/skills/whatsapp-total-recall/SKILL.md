---
name: whatsapp-total-recall
description: Query local WhatsApp history, inspect chronological conversation streams, unknot reply DAG threads, and synthesize intent-constrained reply drafts via WhatsApp Total Recall's local MCP server.
---

# WhatsApp Total Recall MCP Skill

Autonomous AI agent guide for deep exploration across local WhatsApp conversations using the local-first WhatsApp Total Recall Model Context Protocol (MCP) server.

---

## 1. Available Tools

The MCP server exposes 5 privacy-preserving local tools:

| Tool | Purpose | Primary Use Case |
| :--- | :--- | :--- |
| `list_chats` | Enumerate indexed chats with JID, name, message count, and last activity | Discovery, mapping topology, finding contacts/groups |
| `search_messages` | BM25 full-text search across messages, with optional `chat_id` filter | Locating anchor messages, keywords, or topics |
| `get_chat_history` | Sequential chronological message stream before/after an anchor message | Reading unquoted back-and-forth dialogue & context |
| `get_thread_context`| Reconstruct parent-child quote reply DAG hierarchy | Unknotting explicit replies, questions, and group debates |
| `stage_reply_draft` | Synthesize tone-matched response strictly bound to intent bullets | Staging draft in local review box (manual send only) |

---

## 2. Deep Agentic Exploration Workflow Patterns

When investigating a question, finding a past commitment, or exploring conversation history, avoid relying solely on keyword search snippets. Follow these workflow patterns:

### Pattern A: Locate → Expand → Resolve (Single-Chat Deep Dive)

1. **Locate Anchor Message (`search_messages`)**:
   Query keywords, names, or topics to find relevant candidate message PKs and their `chat_jid`.
   ```json
   {
     "name": "search_messages",
     "arguments": { "query": "deployment budget server", "limit": 10 }
   }
   ```
2. **Determine Thread Type**:
   - If the candidate message is part of an unquoted sequential stream (common in WhatsApp):
     Call **`get_chat_history`** with `around_message_id`.
   - If the candidate message is an explicit quote reply (has a quoted parent):
     Call **`get_thread_context`** with `message_id`.
3. **Expand Sequential Stream (`get_chat_history`)**:
   Fetch the surrounding chronological window (10–30 messages) around the target PK to read the immediate conversation before and after.
   ```json
   {
     "name": "get_chat_history",
     "arguments": {
       "chat_id": "1234567890-group@g.us",
       "around_message_id": 482,
       "limit": 20
     }
     ```
   *Why this matters*: WhatsApp users rarely quote every reply. Crucial context (clarifications, emoji agreements, counter-proposals) lives in the sequential messages immediately following the target.

4. **Trace Complex Reply DAGs (`get_thread_context`)**:
   If messages quote earlier messages across interleaving chat bursts, invoke `get_thread_context` to inspect the tree representation and root message.
   ```json
   {
     "name": "get_thread_context",
     "arguments": { "message_id": 482 }
   }
   ```

---

### Pattern B: Cross-Chat Pivot & Entity Tracing

When researching an entity, person, or project across multiple groups or 1:1 discussions:

1. **Discover Relevant Chats (`list_chats`)**:
   Search for candidate contacts or groups:
   ```json
   {
     "name": "list_chats",
     "arguments": { "query": "Finance", "limit": 10 }
   }
   ```
2. **Execute Scoped Searches (`search_messages` with `chat_id`)**:
   Search within specific relevant chats to compare discussions across channels (e.g., what was discussed in the team group vs. what was agreed in 1:1 with the manager):
   ```json
   {
     "name": "search_messages",
     "arguments": {
       "query": "deadline contract",
       "chat_id": "manager@s.whatsapp.net"
     }
   }
   ```
3. **Correlate Timelines**:
   Compare `iso_date` timestamps across chats to establish chronological causality (e.g., "Feature approved in 1:1 chat at 14:15, then announced to group chat at 14:30").

---

### Pattern C: Safe Reply Drafting & Review

When assisting the user in drafting a reply:

1. **Gather Preceding Context**:
   Use `get_chat_history` to inspect the last 15–25 messages in the target chat to capture active topics, recent tone, and pending questions.
2. **Formulate Strict Intent Points**:
   Extract or clarify exact bullet points from the user. Never infer or invent unconfirmed commitments.
3. **Stage Draft (`stage_reply_draft`)**:
   ```json
   {
     "name": "stage_reply_draft",
     "arguments": {
       "chat_id": "colleague@s.whatsapp.net",
       "intent_points": [
         "Confirm received revised spreadsheet",
         "Note that numbers look aligned with Q3 estimates",
         "Will send sign-off by 3 PM"
       ]
     }
   }
   ```
4. **Instruct the User**:
   Report `draft_text` and `tone_profile` to the user and prompt them to copy it to their WhatsApp clipboard.

---

## 3. Invariants & Safety Guardrails

- **Zero Autonomous Sending**: The server structurally prohibits outgoing network calls to WhatsApp Web/Catalyst. All drafts must be manually approved and pasted by the user.
- **Zero Presence Snooping**: Presence tracking, typing status, and last-seen spying APIs are omitted.
- **Opt-In Chat Vault Enforcement**: If a chat is not approved in `vault_config.json`, queries return access denied. Agents must respect this boundary and never attempt to bypass vault restrictions.
