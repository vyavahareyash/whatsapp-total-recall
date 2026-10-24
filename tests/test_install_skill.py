from __future__ import annotations

from pathlib import Path

from whatsapp_total_recall.cli import main as cli_main


def test_install_skill_custom_dest(tmp_path: Path):
    dest_dir = tmp_path / "custom_agent" / "skills"
    code = cli_main(["install-skill", "--dest", str(dest_dir)])
    assert code == 0

    installed_file = dest_dir / "SKILL.md"
    assert installed_file.exists()
    content = installed_file.read_text(encoding="utf-8")
    assert "name: whatsapp-total-recall" in content
    assert "list_chats" in content
    assert "get_chat_history" in content
    assert "Pattern A: Locate → Expand → Resolve" in content

    # Test idempotency without force skips overwrite
    code_idempotent = cli_main(["install-skill", "--dest", str(dest_dir)])
    assert code_idempotent == 0

    # Test force overwrite
    code_force = cli_main(["install-skill", "--dest", str(dest_dir), "--force"])
    assert code_force == 0
