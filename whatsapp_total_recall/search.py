from __future__ import annotations

import json
import math
import re
import sqlite3
import struct
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from whatsapp_total_recall.chunking import rebuild_episode_without_message
from whatsapp_total_recall.embeddings import EmbeddingProvider, get_embedding_provider
from whatsapp_total_recall.types import (
    ConversationalFlowEpisode,
    CrossBurstQuoteContext,
    NormalizedMessage,
    PurgeReport,
    ScoredEpisode,
    SearchResult,
)


def cosine_distance(vec_a_bytes: bytes, vec_b_bytes: bytes) -> float:
    """Compute cosine distance (1 - cosine_similarity) between two packed float32 vectors."""
    n_floats = len(vec_a_bytes) // 4
    if n_floats == 0 or len(vec_b_bytes) // 4 != n_floats:
        return 1.0
    vec_a = struct.unpack(f"{n_floats}f", vec_a_bytes)
    vec_b = struct.unpack(f"{n_floats}f", vec_b_bytes)

    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    sim = dot / (norm_a * norm_b)
    # Numerical stability clamp
    sim = max(-1.0, min(1.0, sim))
    return max(0.0, 1.0 - sim)


class SearchIndex:
    """Manages SQLite FTS5 lexical retrieval and SIMD vector similarity with unified RRF scoring."""

    def __init__(
        self,
        db_path: Path | str = ":memory:",
        dimension: int = 384,
        embedder: EmbeddingProvider | None = None,
    ):
        self.db_path = str(db_path) if isinstance(db_path, Path) else db_path
        if embedder is not None and (dimension == 384 or dimension is None):
            self.dimension = embedder.dimension
        else:
            self.dimension = dimension
        self.embedder = embedder
        self.con = sqlite3.connect(self.db_path)
        self.con.row_factory = sqlite3.Row
        self.con.execute("PRAGMA secure_delete = ON")

        self._sqlite_vec_available = False
        self._init_sqlite_vec_or_fallback()
        self._init_schema()

    def _init_sqlite_vec_or_fallback(self) -> None:
        """Attempt to load native sqlite-vec extension or register portable SIMD fallback."""
        if hasattr(self.con, "enable_load_extension"):
            try:
                import sqlite_vec

                self.con.enable_load_extension(True)
                sqlite_vec.load(self.con)
                self.con.enable_load_extension(False)
                self._sqlite_vec_available = True
            except Exception:
                self._sqlite_vec_available = False

        if not self._sqlite_vec_available:
            self.con.create_function("cosine_dist", 2, cosine_distance)

    def _ensure_vector_table_dimension(
        self, cur: sqlite3.Cursor, target_dim: int
    ) -> bool:
        """Ensure the episode_vectors table matches target_dim.

        If native sqlite-vec is active and the existing table's vector dimension differs,
        recreates the table with target_dim so that model switches don't crash.
        """
        self.dimension = target_dim
        if not self._sqlite_vec_available:
            return True

        cur.execute(
            "SELECT type, sql FROM sqlite_master WHERE name = 'episode_vectors'"
        )
        row = cur.fetchone()
        if row and row["sql"]:
            m = re.search(r"float\[(\d+)\]", row["sql"])
            if m:
                existing_dim = int(m.group(1))
                if existing_dim == target_dim:
                    return True
                # Dimension mismatch: table is float[existing_dim], incoming is float[target_dim]
                cur.execute("DROP TABLE IF EXISTS episode_vectors")

        cur.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS episode_vectors USING vec0(
                episode_id TEXT PRIMARY KEY,
                embedding float[{target_dim}]
            );
            """
        )
        return True

    def _init_schema(self) -> None:
        """Create normalized messages, episodes, FTS5 virtual tables, and vector tables."""
        cur = self.con.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                pk INTEGER PRIMARY KEY,
                chat_session_id INTEGER,
                chat_jid TEXT,
                stanza_id TEXT,
                is_from_me INTEGER,
                sender_name TEXT,
                sender_jid TEXT,
                apple_date REAL,
                unix_timestamp REAL,
                iso_date TEXT,
                text TEXT,
                parent_pk INTEGER,
                expire_timer INTEGER,
                expires_at REAL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                pk UNINDEXED,
                chat_jid,
                sender_name,
                text,
                tokenize = 'porter unicode61'
            );

            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                chat_jid TEXT,
                chat_name TEXT,
                start_timestamp REAL,
                end_timestamp REAL,
                start_iso TEXT,
                end_iso TEXT,
                header TEXT,
                content TEXT,
                message_count INTEGER,
                messages_json TEXT,
                cross_burst_quotes_json TEXT,
                min_expires_at REAL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
                episode_id UNINDEXED,
                chat_jid,
                chat_name,
                header,
                content,
                tokenize = 'porter unicode61'
            );

            CREATE TABLE IF NOT EXISTS sync_state (
                chat_jid TEXT PRIMARY KEY,
                last_message_pk INTEGER NOT NULL,
                last_apple_date REAL NOT NULL,
                last_unix_timestamp REAL NOT NULL,
                last_synced_at REAL NOT NULL,
                total_messages_synced INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_messages_expires_at ON messages(expires_at) WHERE expires_at IS NOT NULL;
            CREATE INDEX IF NOT EXISTS idx_episodes_expires_at ON episodes(min_expires_at) WHERE min_expires_at IS NOT NULL;
            """
        )

        # Migrate existing schema if columns are missing
        cur.execute("PRAGMA table_info(messages)")
        msg_columns = [row["name"] for row in cur.fetchall()]
        if "parent_pk" not in msg_columns:
            cur.execute("ALTER TABLE messages ADD COLUMN parent_pk INTEGER")
        if "expire_timer" not in msg_columns:
            cur.execute("ALTER TABLE messages ADD COLUMN expire_timer INTEGER")
        if "expires_at" not in msg_columns:
            cur.execute("ALTER TABLE messages ADD COLUMN expires_at REAL")

        cur.execute("PRAGMA table_info(episodes)")
        ep_columns = [row["name"] for row in cur.fetchall()]
        if "min_expires_at" not in ep_columns:
            cur.execute("ALTER TABLE episodes ADD COLUMN min_expires_at REAL")

        # Initialize vector store
        cur.execute(
            "SELECT type, sql FROM sqlite_master WHERE name = 'episode_vectors'"
        )
        existing_vec = cur.fetchone()
        if existing_vec is not None and existing_vec["sql"]:
            m = re.search(r"float\[(\d+)\]", existing_vec["sql"])
            if m:
                existing_dim = int(m.group(1))
                if self.embedder is None and self.dimension == 384:
                    self.dimension = existing_dim
        elif existing_vec is None:
            if self._sqlite_vec_available:
                self._ensure_vector_table_dimension(cur, self.dimension)
            else:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS episode_vectors (
                        episode_id TEXT PRIMARY KEY,
                        embedding BLOB
                    );
                    """
                )

        self.con.commit()

    def _get_embedder(
        self, override: EmbeddingProvider | None = None
    ) -> EmbeddingProvider:
        if override is not None:
            return override
        if self.embedder is not None:
            return self.embedder
        self.embedder = get_embedding_provider("auto")
        return self.embedder

    def index_messages(
        self,
        messages: list[NormalizedMessage],
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        """Batch index normalized messages into both storage and FTS5 tables."""
        if not messages:
            return 0

        cur = self.con.cursor()
        total = len(messages)
        step = max(1, min(100, total // 50))
        for idx, msg in enumerate(messages, 1):
            cur.execute(
                """
                INSERT OR REPLACE INTO messages (
                    pk, chat_session_id, chat_jid, stanza_id, is_from_me,
                    sender_name, sender_jid, apple_date, unix_timestamp,
                    iso_date, text, parent_pk, expire_timer, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    msg.pk,
                    msg.chat_session_id,
                    msg.chat_jid,
                    msg.stanza_id,
                    1 if msg.is_from_me else 0,
                    msg.sender_name,
                    msg.sender_jid,
                    msg.apple_date,
                    msg.unix_timestamp,
                    msg.iso_date,
                    msg.text,
                    msg.parent_pk,
                    msg.expire_timer,
                    msg.expires_at,
                ),
            )

            cur.execute("DELETE FROM messages_fts WHERE pk = ?", (str(msg.pk),))
            cur.execute(
                """
                INSERT INTO messages_fts (pk, chat_jid, sender_name, text)
                VALUES (?, ?, ?, ?)
                """,
                (str(msg.pk), msg.chat_jid, msg.sender_name, msg.text),
            )
            if progress_callback and (idx % step == 0 or idx == total):
                progress_callback(idx, total)

        self.con.commit()
        return len(messages)

    def index_episodes(
        self,
        episodes: list[ConversationalFlowEpisode],
        index_constituent_messages: bool = True,
        embedder: EmbeddingProvider | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        """Batch index Conversational Flow Episodes into FTS5 and vector tables."""
        if not episodes:
            return 0

        if index_constituent_messages:
            all_msgs: dict[int, NormalizedMessage] = {}
            for ep in episodes:
                for m in ep.messages:
                    all_msgs[m.pk] = m
            self.index_messages(list(all_msgs.values()))

        active_embedder = self._get_embedder(embedder)

        cur = self.con.cursor()
        if episodes:
            self._ensure_vector_table_dimension(cur, active_embedder.dimension)

        total = len(episodes)
        for idx, ep in enumerate(episodes, 1):
            msgs_payload = [
                {
                    "pk": m.pk,
                    "chat_session_id": m.chat_session_id,
                    "chat_jid": m.chat_jid,
                    "stanza_id": m.stanza_id,
                    "is_from_me": m.is_from_me,
                    "sender_name": m.sender_name,
                    "sender_jid": m.sender_jid,
                    "apple_date": m.apple_date,
                    "unix_timestamp": m.unix_timestamp,
                    "iso_date": m.iso_date,
                    "text": m.text,
                    "parent_pk": m.parent_pk,
                    "expire_timer": m.expire_timer,
                    "expires_at": m.expires_at,
                }
                for m in ep.messages
            ]

            quotes_payload = [
                {
                    "child_pk": q.child_pk,
                    "parent_pk": q.parent_pk,
                    "parent_sender_name": q.parent_sender_name,
                    "parent_text": q.parent_text,
                    "parent_iso_date": q.parent_iso_date,
                }
                for q in ep.cross_burst_quotes
            ]

            cur.execute(
                """
                INSERT OR REPLACE INTO episodes (
                    episode_id, chat_jid, chat_name, start_timestamp,
                    end_timestamp, start_iso, end_iso, header, content,
                    message_count, messages_json, cross_burst_quotes_json,
                    min_expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ep.episode_id,
                    ep.chat_jid,
                    ep.chat_name,
                    ep.start_timestamp,
                    ep.end_timestamp,
                    ep.start_iso,
                    ep.end_iso,
                    ep.header,
                    ep.content,
                    len(ep.messages),
                    json.dumps(msgs_payload),
                    json.dumps(quotes_payload),
                    ep.min_expires_at,
                ),
            )

            cur.execute(
                "DELETE FROM episodes_fts WHERE episode_id = ?", (ep.episode_id,)
            )
            cur.execute(
                """
                INSERT INTO episodes_fts (episode_id, chat_jid, chat_name, header, content)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ep.episode_id, ep.chat_jid, ep.chat_name, ep.header, ep.content),
            )

            # Compute and store dense vector embedding
            text_to_embed = f"{ep.header}\n{ep.content}"
            vec = active_embedder.embed_text(text_to_embed)
            if len(vec) != self.dimension:
                self._ensure_vector_table_dimension(cur, len(vec))
            vec_bytes = struct.pack(f"{len(vec)}f", *vec)

            cur.execute(
                "DELETE FROM episode_vectors WHERE episode_id = ?",
                (ep.episode_id,),
            )
            cur.execute(
                """
                INSERT INTO episode_vectors (episode_id, embedding)
                VALUES (?, ?)
                """,
                (ep.episode_id, vec_bytes),
            )
            if progress_callback:
                progress_callback(idx, total)

        self.con.commit()
        return len(episodes)

    def get_sync_state(self, chat_jid: str | None = None) -> dict[str, Any] | None:
        """Retrieve sync state record(s). If chat_jid is provided, returns single record dict or None."""
        cur = self.con.cursor()
        if chat_jid is not None:
            cur.execute(
                """
                SELECT chat_jid, last_message_pk, last_apple_date, last_unix_timestamp, last_synced_at, total_messages_synced
                FROM sync_state WHERE chat_jid = ?
                """,
                (chat_jid,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

        cur.execute(
            """
            SELECT chat_jid, last_message_pk, last_apple_date, last_unix_timestamp, last_synced_at, total_messages_synced
            FROM sync_state
            """
        )
        return {row["chat_jid"]: dict(row) for row in cur.fetchall()}

    def update_sync_state(
        self,
        chat_jid: str,
        last_message_pk: int,
        last_apple_date: float,
        last_unix_timestamp: float,
        synced_count: int = 0,
    ) -> None:
        """Upsert sync state for a chat."""
        cur = self.con.cursor()
        now_ts = datetime.now(UTC).timestamp()
        cur.execute(
            """
            INSERT INTO sync_state (
                chat_jid, last_message_pk, last_apple_date, last_unix_timestamp, last_synced_at, total_messages_synced
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(chat_jid) DO UPDATE SET
                last_message_pk = excluded.last_message_pk,
                last_apple_date = excluded.last_apple_date,
                last_unix_timestamp = excluded.last_unix_timestamp,
                last_synced_at = excluded.last_synced_at,
                total_messages_synced = sync_state.total_messages_synced + excluded.total_messages_synced
            """,
            (
                chat_jid,
                last_message_pk,
                last_apple_date,
                last_unix_timestamp,
                now_ts,
                synced_count,
            ),
        )
        self.con.commit()

    @staticmethod
    def sanitize_fts_query(query: str) -> str:
        """Sanitize raw user query into safe FTS5 query expression."""
        tokens = re.findall(r"\w+", query)
        if not tokens:
            return ""
        return " OR ".join(f'"{token}"' for token in tokens)

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Perform BM25 ranked lexical search over indexed messages."""
        fts_query = self.sanitize_fts_query(query)
        if not fts_query:
            return []

        cur = self.con.cursor()
        cur.execute(
            """
            SELECT
                m.pk,
                m.chat_jid,
                m.sender_name,
                m.iso_date,
                m.text,
                bm25(messages_fts) AS bm25_score
            FROM messages_fts f
            JOIN messages m ON m.pk = CAST(f.pk AS INTEGER)
            WHERE messages_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, limit),
        )

        results: list[SearchResult] = []
        for row in cur.fetchall():
            results.append(
                SearchResult(
                    message_pk=int(row["pk"]),
                    chat_jid=str(row["chat_jid"]),
                    sender_name=str(row["sender_name"]),
                    iso_date=str(row["iso_date"]),
                    text=str(row["text"]),
                    bm25_score=float(row["bm25_score"]),
                )
            )
        return results

    def _row_to_episode(self, row: sqlite3.Row) -> ConversationalFlowEpisode:
        raw_msgs = json.loads(row["messages_json"] or "[]")
        msgs = tuple(
            NormalizedMessage(
                pk=int(d["pk"]),
                chat_session_id=int(d["chat_session_id"]),
                chat_jid=str(d["chat_jid"]),
                stanza_id=str(d["stanza_id"]),
                is_from_me=bool(d["is_from_me"]),
                sender_name=str(d["sender_name"]),
                sender_jid=str(d["sender_jid"]),
                apple_date=float(d["apple_date"]),
                unix_timestamp=float(d["unix_timestamp"]),
                iso_date=str(d["iso_date"]),
                text=str(d["text"]),
                parent_pk=int(d["parent_pk"])
                if d.get("parent_pk") is not None
                else None,
                expire_timer=int(d["expire_timer"])
                if d.get("expire_timer") is not None
                else None,
                expires_at=float(d["expires_at"])
                if d.get("expires_at") is not None
                else None,
            )
            for d in raw_msgs
        )

        raw_quotes = json.loads(row["cross_burst_quotes_json"] or "[]")
        quotes = tuple(
            CrossBurstQuoteContext(
                child_pk=int(q["child_pk"]),
                parent_pk=int(q["parent_pk"]),
                parent_sender_name=str(q["parent_sender_name"]),
                parent_text=str(q["parent_text"]),
                parent_iso_date=str(q["parent_iso_date"]),
            )
            for q in raw_quotes
        )

        seen_participants: dict[str, None] = {}
        for m in msgs:
            seen_participants[m.sender_name] = None

        keys = set(row.keys())
        min_exp = (
            float(row["min_expires_at"])
            if ("min_expires_at" in keys and row["min_expires_at"] is not None)
            else None
        )

        return ConversationalFlowEpisode(
            episode_id=str(row["episode_id"]),
            chat_jid=str(row["chat_jid"]),
            chat_name=str(row["chat_name"]),
            start_timestamp=float(row["start_timestamp"]),
            end_timestamp=float(row["end_timestamp"]),
            start_iso=str(row["start_iso"]),
            end_iso=str(row["end_iso"]),
            participants=tuple(seen_participants.keys()),
            messages=msgs,
            cross_burst_quotes=quotes,
            header=str(row["header"]),
            content=str(row["content"]),
            min_expires_at=min_exp,
        )

    def search_episodes(self, query: str, limit: int = 10) -> list[ScoredEpisode]:
        """Perform BM25 ranked lexical search over indexed Conversational Flow Episodes."""
        fts_query = self.sanitize_fts_query(query)
        if not fts_query:
            return []

        cur = self.con.cursor()
        cur.execute(
            """
            SELECT
                e.*,
                bm25(episodes_fts) AS bm25_score
            FROM episodes_fts f
            JOIN episodes e ON e.episode_id = f.episode_id
            WHERE episodes_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_query, limit),
        )

        results: list[ScoredEpisode] = []
        for idx, row in enumerate(cur.fetchall(), 1):
            episode = self._row_to_episode(row)
            results.append(
                ScoredEpisode(
                    episode=episode,
                    bm25_score=float(row["bm25_score"]),
                    fts_rank=idx,
                )
            )
        return results

    def search_episodes_vector(
        self,
        query: str,
        limit: int = 10,
        embedder: EmbeddingProvider | None = None,
    ) -> list[ScoredEpisode]:
        """Perform vector KNN similarity search over indexed Conversational Flow Episodes."""
        active_embedder = self._get_embedder(embedder)
        query_vec = active_embedder.embed_text(query)
        query_bytes = struct.pack(f"{len(query_vec)}f", *query_vec)

        cur = self.con.cursor()
        if self._sqlite_vec_available:
            cur.execute("SELECT sql FROM sqlite_master WHERE name = 'episode_vectors'")
            row = cur.fetchone()
            if row and row["sql"]:
                m = re.search(r"float\[(\d+)\]", row["sql"])
                if m and int(m.group(1)) != len(query_vec):
                    return []

            cur.execute(
                """
                SELECT
                    e.*,
                    v.distance AS vector_distance,
                    ROW_NUMBER() OVER (ORDER BY v.distance ASC) AS rank_vec
                FROM episode_vectors v
                JOIN episodes e ON e.episode_id = v.episode_id
                WHERE v.embedding MATCH ? AND k = ?
                """,
                (query_bytes, limit),
            )
        else:
            cur.execute(
                """
                SELECT
                    e.*,
                    cosine_dist(v.embedding, ?) AS vector_distance,
                    ROW_NUMBER() OVER (ORDER BY cosine_dist(v.embedding, ?) ASC) AS rank_vec
                FROM episode_vectors v
                JOIN episodes e ON e.episode_id = v.episode_id
                ORDER BY vector_distance ASC
                LIMIT ?
                """,
                (query_bytes, query_bytes, limit),
            )

        results: list[ScoredEpisode] = []
        for row in cur.fetchall():
            episode = self._row_to_episode(row)
            vec_rank = int(row["rank_vec"])
            dist = float(row["vector_distance"])
            results.append(
                ScoredEpisode(
                    episode=episode,
                    rrf_score=1.0 / (60.0 + vec_rank),
                    vec_rank=vec_rank,
                    vector_distance=dist,
                )
            )
        return results

    def search_hybrid(
        self,
        query: str,
        limit: int = 10,
        k_rrf: float = 60.0,
        embedder: EmbeddingProvider | None = None,
        candidate_limit: int = 20,
    ) -> list[ScoredEpisode]:
        """Execute unified SQL Reciprocal Rank Fusion (RRF) query merging FTS5 BM25 and vector similarity."""
        active_embedder = self._get_embedder(embedder)
        query_vec = active_embedder.embed_text(query)
        query_bytes = struct.pack(f"{len(query_vec)}f", *query_vec)

        fts_query = self.sanitize_fts_query(query)
        has_fts = bool(fts_query)

        cur = self.con.cursor()

        if self._sqlite_vec_available:
            vec_cte = """
            vec_matches AS (
                SELECT
                    episode_id,
                    distance AS vector_distance,
                    ROW_NUMBER() OVER (ORDER BY distance ASC) AS rank_vec
                FROM episode_vectors
                WHERE embedding MATCH ? AND k = ?
            )
            """
            vec_params = (query_bytes, candidate_limit)
        else:
            vec_cte = """
            vec_matches AS (
                SELECT
                    episode_id,
                    cosine_dist(embedding, ?) AS vector_distance,
                    ROW_NUMBER() OVER (ORDER BY cosine_dist(embedding, ?) ASC) AS rank_vec
                FROM episode_vectors
                ORDER BY vector_distance ASC
                LIMIT ?
            )
            """
            vec_params = (query_bytes, query_bytes, candidate_limit)

        if has_fts:
            fts_cte = """
            fts_matches AS (
                SELECT
                    episode_id,
                    rank AS bm25_score,
                    ROW_NUMBER() OVER (ORDER BY rank) AS rank_fts
                FROM episodes_fts
                WHERE episodes_fts MATCH ?
                LIMIT ?
            ),
            """
            fts_params = (fts_query, candidate_limit)
            fts_join = "LEFT JOIN fts_matches f ON f.episode_id = c.episode_id"
            fts_union = "SELECT episode_id FROM fts_matches UNION "
            fts_score_expr = f"coalesce(1.0 / ({k_rrf} + f.rank_fts), 0.0)"
            bm25_col = "coalesce(f.bm25_score, 0.0) AS bm25_score,"
            rank_fts_col = "f.rank_fts,"
        else:
            fts_cte = ""
            fts_params = ()
            fts_join = ""
            fts_union = ""
            fts_score_expr = "0.0"
            bm25_col = "0.0 AS bm25_score,"
            rank_fts_col = "NULL AS rank_fts,"

        vec_score_expr = f"coalesce(1.0 / ({k_rrf} + v.rank_vec), 0.0)"

        sql = f"""
        WITH {fts_cte}
        {vec_cte},
        candidates AS (
            {fts_union}
            SELECT episode_id FROM vec_matches
        )
        SELECT
            e.*,
            {rank_fts_col}
            v.rank_vec,
            v.vector_distance,
            {bm25_col}
            {fts_score_expr} AS fts_rrf_score,
            {vec_score_expr} AS vec_rrf_score,
            ({fts_score_expr} + {vec_score_expr}) AS rrf_score
        FROM candidates c
        JOIN episodes e ON e.episode_id = c.episode_id
        {fts_join}
        LEFT JOIN vec_matches v ON v.episode_id = c.episode_id
        ORDER BY rrf_score DESC
        LIMIT ?;
        """

        all_params = fts_params + vec_params + (limit,)
        cur.execute(sql, all_params)

        results: list[ScoredEpisode] = []
        for row in cur.fetchall():
            episode = self._row_to_episode(row)
            results.append(
                ScoredEpisode(
                    episode=episode,
                    bm25_score=float(row["bm25_score"] or 0.0),
                    rrf_score=float(row["rrf_score"]),
                    fts_rank=int(row["rank_fts"])
                    if row["rank_fts"] is not None
                    else None,
                    vec_rank=int(row["rank_vec"])
                    if row["rank_vec"] is not None
                    else None,
                    vector_distance=float(row["vector_distance"])
                    if row["vector_distance"] is not None
                    else None,
                )
            )
        return results

    def purge(
        self,
        message_pk: int | str,
        embedder: EmbeddingProvider | None = None,
    ) -> PurgeReport:
        """Perform hard cryptographic purge (PRAGMA secure_delete = ON) of a revoked message."""
        cur = self.con.cursor()

        # Enforce secure delete
        cur.execute("PRAGMA secure_delete = ON")

        target_pk: int | None = None
        target_stanza_id: str | None = None

        if isinstance(message_pk, int):
            target_pk = message_pk
            cur.execute("SELECT stanza_id FROM messages WHERE pk = ?", (target_pk,))
            row = cur.fetchone()
            if row:
                target_stanza_id = row["stanza_id"]
        elif isinstance(message_pk, str) and message_pk.isdigit():
            target_pk = int(message_pk)
            cur.execute("SELECT stanza_id FROM messages WHERE pk = ?", (target_pk,))
            row = cur.fetchone()
            if row:
                target_stanza_id = row["stanza_id"]
        else:
            target_stanza_id = str(message_pk)
            cur.execute(
                "SELECT pk FROM messages WHERE stanza_id = ?", (target_stanza_id,)
            )
            row = cur.fetchone()
            if row:
                target_pk = int(row["pk"])

        raw_purged = 0
        if target_pk is not None:
            cur.execute("SELECT pk FROM messages WHERE pk = ?", (target_pk,))
            if cur.fetchone():
                raw_purged = 1
            cur.execute("DELETE FROM messages WHERE pk = ?", (target_pk,))
            cur.execute("DELETE FROM messages_fts WHERE pk = ?", (str(target_pk),))

        # Scan episodes referencing this message
        cur.execute("SELECT * FROM episodes")
        all_ep_rows = cur.fetchall()

        episodes_updated = 0
        episodes_deleted = 0
        active_embedder = self._get_embedder(embedder)

        for row in all_ep_rows:
            ep = self._row_to_episode(row)
            has_match = False
            if target_pk is not None:
                has_match = any(m.pk == target_pk for m in ep.messages)
            if not has_match and target_stanza_id:
                has_match = any(m.stanza_id == target_stanza_id for m in ep.messages)
                if has_match and target_pk is None:
                    matched = [
                        m for m in ep.messages if m.stanza_id == target_stanza_id
                    ]
                    if matched:
                        target_pk = matched[0].pk

            if not has_match and target_pk is not None:
                has_match = any(q.parent_pk == target_pk for q in ep.cross_burst_quotes)

            if not has_match:
                continue

            ep_id = ep.episode_id
            rebuilt = (
                rebuild_episode_without_message(ep, target_pk)
                if target_pk is not None
                else None
            )

            if rebuilt is None:
                # Purge empty episode completely
                cur.execute("DELETE FROM episodes WHERE episode_id = ?", (ep_id,))
                cur.execute("DELETE FROM episodes_fts WHERE episode_id = ?", (ep_id,))
                cur.execute(
                    "DELETE FROM episode_vectors WHERE episode_id = ?", (ep_id,)
                )
                episodes_deleted += 1
            else:
                msgs_payload = [
                    {
                        "pk": m.pk,
                        "chat_session_id": m.chat_session_id,
                        "chat_jid": m.chat_jid,
                        "stanza_id": m.stanza_id,
                        "is_from_me": m.is_from_me,
                        "sender_name": m.sender_name,
                        "sender_jid": m.sender_jid,
                        "apple_date": m.apple_date,
                        "unix_timestamp": m.unix_timestamp,
                        "iso_date": m.iso_date,
                        "text": m.text,
                        "parent_pk": m.parent_pk,
                        "expire_timer": m.expire_timer,
                        "expires_at": m.expires_at,
                    }
                    for m in rebuilt.messages
                ]
                quotes_payload = [
                    {
                        "child_pk": q.child_pk,
                        "parent_pk": q.parent_pk,
                        "parent_sender_name": q.parent_sender_name,
                        "parent_text": q.parent_text,
                        "parent_iso_date": q.parent_iso_date,
                    }
                    for q in rebuilt.cross_burst_quotes
                ]

                cur.execute(
                    """
                    UPDATE episodes
                    SET header = ?, content = ?, start_timestamp = ?,
                        end_timestamp = ?, start_iso = ?, end_iso = ?,
                        message_count = ?, messages_json = ?,
                        cross_burst_quotes_json = ?, min_expires_at = ?
                    WHERE episode_id = ?
                    """,
                    (
                        rebuilt.header,
                        rebuilt.content,
                        rebuilt.start_timestamp,
                        rebuilt.end_timestamp,
                        rebuilt.start_iso,
                        rebuilt.end_iso,
                        len(rebuilt.messages),
                        json.dumps(msgs_payload),
                        json.dumps(quotes_payload),
                        rebuilt.min_expires_at,
                        ep_id,
                    ),
                )

                cur.execute("DELETE FROM episodes_fts WHERE episode_id = ?", (ep_id,))
                cur.execute(
                    """
                    INSERT INTO episodes_fts (episode_id, chat_jid, chat_name, header, content)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        ep_id,
                        rebuilt.chat_jid,
                        rebuilt.chat_name,
                        rebuilt.header,
                        rebuilt.content,
                    ),
                )

                # Re-calculate vector embedding
                text_to_embed = f"{rebuilt.header}\n{rebuilt.content}"
                new_vec = active_embedder.embed_text(text_to_embed)
                if len(new_vec) != self.dimension:
                    self._ensure_vector_table_dimension(cur, len(new_vec))
                new_vec_bytes = struct.pack(f"{len(new_vec)}f", *new_vec)
                cur.execute(
                    "DELETE FROM episode_vectors WHERE episode_id = ?", (ep_id,)
                )
                cur.execute(
                    """
                    INSERT INTO episode_vectors (episode_id, embedding)
                    VALUES (?, ?)
                    """,
                    (ep_id, new_vec_bytes),
                )
                episodes_updated += 1

        self.con.commit()

        # Vacuum to overwrite cleared space
        cur.execute("PRAGMA incremental_vacuum")

        return PurgeReport(
            message_pk=target_pk if target_pk is not None else 0,
            episodes_updated=episodes_updated,
            episodes_deleted=episodes_deleted,
            raw_messages_purged=raw_purged,
            secure_delete_executed=True,
            stanza_id=target_stanza_id,
        )

    def purge_by_stanza_id(
        self, stanza_id: str, embedder: EmbeddingProvider | None = None
    ) -> PurgeReport:
        """Purge message by its WhatsApp stanza ID."""
        return self.purge(stanza_id, embedder=embedder)

    def purge_expired(
        self,
        current_timestamp: float | None = None,
        embedder: EmbeddingProvider | None = None,
    ) -> list[PurgeReport]:
        """Purge all expired disappearing messages based on expires_at TTL metadata."""
        if current_timestamp is None:
            current_timestamp = datetime.now(UTC).timestamp()

        cur = self.con.cursor()
        cur.execute(
            "SELECT pk FROM messages WHERE expires_at IS NOT NULL AND expires_at <= ? ORDER BY pk ASC",
            (current_timestamp,),
        )
        pks_to_purge = [int(row["pk"]) for row in cur.fetchall()]
        reports: list[PurgeReport] = []
        for pk in pks_to_purge:
            rep = self.purge(pk, embedder=embedder)
            reports.append(rep)
        return reports

    def get_all_message_pks(self) -> set[int]:
        """Return set of all indexed message primary keys."""
        cur = self.con.cursor()
        cur.execute("SELECT pk FROM messages")
        return {int(row["pk"]) for row in cur.fetchall()}

    def get_expired_messages(
        self, current_timestamp: float | None = None
    ) -> list[NormalizedMessage]:
        """Return list of messages whose expires_at is past the timestamp."""
        if current_timestamp is None:
            current_timestamp = datetime.now(UTC).timestamp()
        cur = self.con.cursor()
        cur.execute(
            "SELECT * FROM messages WHERE expires_at IS NOT NULL AND expires_at <= ? ORDER BY unix_timestamp ASC, pk ASC",
            (current_timestamp,),
        )
        return [self._row_to_normalized_message(row) for row in cur.fetchall()]

    def get_episode(self, episode_id: str) -> ConversationalFlowEpisode | None:
        """Fetch a single episode by episode ID."""
        cur = self.con.cursor()
        cur.execute("SELECT * FROM episodes WHERE episode_id = ?", (episode_id,))
        row = cur.fetchone()
        return self._row_to_episode(row) if row else None

    def get_episodes_for_chat(self, chat_jid: str) -> list[ConversationalFlowEpisode]:
        """Fetch all episodes for a specific chat ordered chronologically."""
        cur = self.con.cursor()
        cur.execute(
            "SELECT * FROM episodes WHERE chat_jid = ? ORDER BY start_timestamp ASC",
            (chat_jid,),
        )
        return [self._row_to_episode(row) for row in cur.fetchall()]

    def get_all_episodes(self) -> list[ConversationalFlowEpisode]:
        """Fetch all indexed episodes ordered chronologically."""
        cur = self.con.cursor()
        cur.execute("SELECT * FROM episodes ORDER BY start_timestamp ASC")
        return [self._row_to_episode(row) for row in cur.fetchall()]

    @staticmethod
    def _row_to_normalized_message(row: sqlite3.Row) -> NormalizedMessage:
        keys = set(row.keys())
        return NormalizedMessage(
            pk=int(row["pk"]),
            chat_session_id=int(row["chat_session_id"]),
            chat_jid=str(row["chat_jid"]),
            stanza_id=str(row["stanza_id"] or ""),
            is_from_me=bool(row["is_from_me"]),
            sender_name=str(row["sender_name"]),
            sender_jid=str(row["sender_jid"]),
            apple_date=float(row["apple_date"]),
            unix_timestamp=float(row["unix_timestamp"]),
            iso_date=str(row["iso_date"]),
            text=str(row["text"]),
            parent_pk=int(row["parent_pk"]) if row["parent_pk"] is not None else None,
            expire_timer=int(row["expire_timer"])
            if ("expire_timer" in keys and row["expire_timer"] is not None)
            else None,
            expires_at=float(row["expires_at"])
            if ("expires_at" in keys and row["expires_at"] is not None)
            else None,
        )

    def get_message(self, pk: int) -> NormalizedMessage | None:
        """Fetch a single message by PK."""
        cur = self.con.cursor()
        cur.execute("SELECT * FROM messages WHERE pk = ?", (pk,))
        row = cur.fetchone()
        return self._row_to_normalized_message(row) if row else None

    def get_messages_for_chat(self, chat_jid: str) -> list[NormalizedMessage]:
        """Fetch all messages for a specific chat JID ordered chronologically."""
        cur = self.con.cursor()
        cur.execute(
            "SELECT * FROM messages WHERE chat_jid = ? ORDER BY unix_timestamp ASC, pk ASC",
            (chat_jid,),
        )
        return [self._row_to_normalized_message(row) for row in cur.fetchall()]

    def get_recent_outgoing_messages(
        self, chat_jid: str, limit: int = 10
    ) -> list[NormalizedMessage]:
        """Fetch recent outgoing user messages (is_from_me = 1) for chat in chronological order."""
        cur = self.con.cursor()
        cur.execute(
            """
            SELECT * FROM messages
            WHERE chat_jid = ? AND is_from_me = 1
            ORDER BY unix_timestamp DESC, pk DESC
            LIMIT ?
            """,
            (chat_jid, limit),
        )
        rows = cur.fetchall()
        return [self._row_to_normalized_message(row) for row in reversed(rows)]

    def get_all_messages(self) -> list[NormalizedMessage]:
        """Fetch all indexed messages ordered chronologically."""
        cur = self.con.cursor()
        cur.execute("SELECT * FROM messages ORDER BY unix_timestamp ASC, pk ASC")
        return [self._row_to_normalized_message(row) for row in cur.fetchall()]

    def close(self) -> None:
        """Close database connection."""
        self.con.close()

    def __enter__(self) -> SearchIndex:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
