from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from whatsapp_total_recall.cli import main as cli_main


def test_reset_help():
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["reset", "--help"])
    assert exc_info.value.code == 0


def test_reset_aborted(tmp_path: Path):
    db_file = tmp_path / "test.db"
    db_file.write_text("dummy")

    with patch("builtins.input", return_value="n"):
        code = cli_main(["reset", "--db", str(db_file)])
        assert code == 0
        assert db_file.exists()


def test_reset_force_purges_db_and_mcp(tmp_path: Path, monkeypatch):
    db_file = tmp_path / "test_whatsapp.db"
    with sqlite3.connect(db_file) as conn:
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, text TEXT);")
        conn.execute("INSERT INTO messages VALUES (1, 'secret message');")
        conn.commit()

    # Mock canonical storage directory
    fake_storage = tmp_path / ".fake_whatsapp_storage"
    fake_storage.mkdir()
    (fake_storage / "sample.txt").write_text("hello")
    monkeypatch.setattr(
        "whatsapp_total_recall.cli.get_canonical_storage_dir", lambda: fake_storage
    )

    # Mock home directory for skill and MCP config
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    # Mock Claude desktop config
    claude_cfg_dir = fake_home / "Library" / "Application Support" / "Claude"
    claude_cfg_dir.mkdir(parents=True)
    claude_cfg = claude_cfg_dir / "claude_desktop_config.json"
    claude_cfg.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "whatsapp-total-recall": {"command": "uvx"},
                    "other-tool": {"command": "node"},
                }
            }
        )
    )

    # Mock installed skill
    fake_skill_dir = fake_home / ".claude" / "skills" / "whatsapp-total-recall"
    fake_skill_dir.mkdir(parents=True)
    (fake_skill_dir / "SKILL.md").write_text("skill docs")

    code = cli_main(["reset", "--force", "--db", str(db_file)])
    assert code == 0

    # Verify db and storage removed
    assert not db_file.exists()
    assert not fake_storage.exists()

    # Verify skill removed
    assert not fake_skill_dir.exists()

    # Verify MCP pruned without deleting other servers
    mcp_data = json.loads(claude_cfg.read_text(encoding="utf-8"))
    assert "whatsapp-total-recall" not in mcp_data["mcpServers"]
    assert "other-tool" in mcp_data["mcpServers"]


def test_reset_keep_db(tmp_path: Path, monkeypatch):
    db_file = tmp_path / "retained.db"
    db_file.write_text("keep me")

    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))

    fake_skill_dir = fake_home / ".claude" / "skills" / "whatsapp-total-recall"
    fake_skill_dir.mkdir(parents=True)
    (fake_skill_dir / "SKILL.md").write_text("skill docs")

    code = cli_main(["reset", "--force", "--keep-db", "--db", str(db_file)])
    assert code == 0

    assert db_file.exists()
    assert not fake_skill_dir.exists()
