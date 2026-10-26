---
description: Search local WhatsApp messages or conversation episodes
argument-name: query
---
Search the local WhatsApp database for "$ARGUMENTS".

1. **Tool Selection Strategy**:
   - For broad topics, conceptual questions, commitments, or decisions, call `search_conversations`.
   - For exact terms, codes, phone numbers, addresses, or identifiers, call `search_messages`.
2. **Context Resolution**:
   - If a candidate message needs surrounding dialogue context, invoke `get_chat_history`.
   - If resolving a back-and-forth reply or quote thread in a group, invoke `get_thread_context`.
3. **Synthesis**:
   - Synthesize findings concisely with timestamps, contact/group names, and speaker attribution.
