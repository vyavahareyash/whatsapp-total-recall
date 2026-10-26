#!/usr/bin/env bash
# Pre-flight environment check for WhatsApp Total Recall Claude Plugin

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "⚠️  [WhatsApp Total Recall] Note: macOS is required to read native WhatsApp desktop databases."
  exit 0
fi

SHARED_DIR="$HOME/Library/Group Containers/group.net.whatsapp.WhatsApp.shared"
DB_FILE="$SHARED_DIR/ChatStorage.sqlite"

if [[ ! -d "$SHARED_DIR" ]]; then
  echo "⚠️  [WhatsApp Total Recall] WhatsApp desktop shared container not found at '$SHARED_DIR'."
  echo "    Ensure official WhatsApp for macOS (Catalyst) is installed and logged in."
  exit 0
fi

if [[ ! -r "$DB_FILE" ]]; then
  echo "⚠️  [WhatsApp Total Recall] Full Disk Access (FDA) required to read ChatStorage.sqlite."
  echo "    Please grant Full Disk Access to your terminal or Claude Code:"
  echo "    System Settings -> Privacy & Security -> Full Disk Access -> enable Claude/Terminal."
  exit 0
fi

exit 0
