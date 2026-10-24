from __future__ import annotations

import contextlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from whatsapp_total_recall.vault import VaultConfig

DEFAULT_CONFIG_PATH = Path("config.json")


def get_canonical_storage_dir() -> Path:
    """Return ~/.whatsapp-total-recall directory, ensuring it exists."""
    env_dir = os.environ.get("WHATSAPP_TOTAL_RECALL_DIR")
    if env_dir:
        dir_path = Path(env_dir)
    elif "PYTEST_CURRENT_TEST" in os.environ:
        dir_path = Path(".")
    else:
        dir_path = Path.home() / ".whatsapp-total-recall"
    with contextlib.suppress(OSError):
        dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_default_db_path() -> Path:
    """Resolve the default SQLite index database path.

    Prioritizes:
    1. Environment override WHATSAPP_TOTAL_RECALL_DIR / WHATSAPP_TOTAL_RECALL_DB
    2. In pytest: local whatsapp_total_recall.db
    3. Canonical ~/.whatsapp-total-recall/whatsapp_total_recall.db (if it exists)
    4. Local ./whatsapp_total_recall.db / ./total_recall.db (if it exists in CWD)
    5. Canonical ~/.whatsapp-total-recall/whatsapp_total_recall.db (default for new databases)
    """
    env_db = os.environ.get("WHATSAPP_TOTAL_RECALL_DB")
    if env_db:
        return Path(env_db)
    env_dir = os.environ.get("WHATSAPP_TOTAL_RECALL_DIR")
    if env_dir:
        return Path(env_dir) / "whatsapp_total_recall.db"

    if "PYTEST_CURRENT_TEST" in os.environ:
        return Path("whatsapp_total_recall.db")

    canonical = Path.home() / ".whatsapp-total-recall" / "whatsapp_total_recall.db"
    if canonical.exists():
        return canonical
    for local_name in ("whatsapp_total_recall.db", "total_recall.db"):
        local_path = Path(local_name)
        if local_path.exists():
            return local_path.resolve()
    get_canonical_storage_dir()
    return canonical


def get_default_vault_path() -> Path:
    """Resolve default vault config path.

    Prioritizes:
    1. Environment override WHATSAPP_TOTAL_RECALL_DIR
    2. In pytest: local vault_config.json
    3. Canonical ~/.whatsapp-total-recall/vault_config.json (if it exists)
    4. Local ./vault_config.json / ./vault_perimeter.json (if in CWD)
    5. Canonical ~/.whatsapp-total-recall/vault_config.json
    """
    env_dir = os.environ.get("WHATSAPP_TOTAL_RECALL_DIR")
    if env_dir:
        return Path(env_dir) / "vault_config.json"

    if "PYTEST_CURRENT_TEST" in os.environ:
        return Path("vault_config.json")

    canonical = Path.home() / ".whatsapp-total-recall" / "vault_config.json"
    if canonical.exists():
        return canonical
    for local_name in ("vault_config.json", "vault_perimeter.json"):
        local_path = Path(local_name)
        if local_path.exists():
            return local_path.resolve()
    get_canonical_storage_dir()
    return canonical


@dataclass
class AppConfig:
    """Unified application configuration and user preferences for Total Recall."""

    db_path: Path = Path("whatsapp_total_recall.db")
    vault_path: Path = Path("vault_config.json")
    ollama_host: str = "http://127.0.0.1:11434"
    default_embedding_model: str = "nomic-embed-text"
    default_llm_model: str = "qwen2.5:7b"
    group_burst_minutes: float = 15.0
    direct_burst_minutes: float = 45.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a JSON dictionary."""
        return {
            "version": 1,
            "db_path": str(self.db_path),
            "vault_path": str(self.vault_path),
            "ollama_host": self.ollama_host,
            "default_embedding_model": self.default_embedding_model,
            "default_llm_model": self.default_llm_model,
            "group_burst_minutes": self.group_burst_minutes,
            "direct_burst_minutes": self.direct_burst_minutes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppConfig:
        """Deserialize configuration from dictionary."""
        return cls(
            db_path=Path(data.get("db_path", "whatsapp_total_recall.db")),
            vault_path=Path(data.get("vault_path", "vault_config.json")),
            ollama_host=str(data.get("ollama_host", "http://127.0.0.1:11434")),
            default_embedding_model=str(
                data.get("default_embedding_model", "nomic-embed-text")
            ),
            default_llm_model=str(data.get("default_llm_model", "qwen2.5:7b")),
            group_burst_minutes=float(data.get("group_burst_minutes", 15.0)),
            direct_burst_minutes=float(data.get("direct_burst_minutes", 45.0)),
        )

    def save(self, path: Path | str = DEFAULT_CONFIG_PATH) -> None:
        """Save configuration to JSON file."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG_PATH) -> AppConfig:
        """Load configuration from JSON file. Returns default if not found."""
        file_path = Path(path)
        if not file_path.exists():
            return cls()
        data = json.loads(file_path.read_text())
        return cls.from_dict(data)

    def load_vault_config(self) -> VaultConfig:
        """Load VaultConfig from vault_path or initialize empty."""
        if self.vault_path.exists():
            cfg = VaultConfig.load(self.vault_path)
            # Sync burst minutes and host if not custom set in vault
            cfg.group_burst_minutes = self.group_burst_minutes
            cfg.direct_burst_minutes = self.direct_burst_minutes
            cfg.ollama_host = self.ollama_host
            return cfg
        return VaultConfig(
            group_burst_minutes=self.group_burst_minutes,
            direct_burst_minutes=self.direct_burst_minutes,
            ollama_host=self.ollama_host,
        )
