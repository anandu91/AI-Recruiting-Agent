# app/core/storage.py
from __future__ import annotations

import os
import json
from typing import Dict, List, Tuple, Any, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Embeddings for Chroma indexing / queries
try:
    from app.core.embeddings import embed_texts
    _HAS_EMB = True
except Exception:
    _HAS_EMB = False

# ---- Single DB URL used everywhere
DB_URL = (
    os.getenv("DATABASE_URL") or os.getenv("DB_URL") or os.getenv("SQLITE_URL") or "sqlite:///app.db"
)
_engine: Optional[Engine] = None


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(DB_URL, future=True)
    return _engine


# ---- helpers to inspect/patch schema (used for safe migrations)
def _table_exists(conn, name: str) -> bool:
    row = conn.execute(text(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=:n"
    ), {"n": name}).fetchone()
    return bool(row)

def _column_names(conn, table: str) -> List[str]:
    try:
        return [r[1] for r in conn.execute(text(f"PRAGMA table_info({table});")).fetchall()]
    except Exception:
        return []

def _ensure_skill_meta_schema(conn) -> None:
    """
    Ensure candidate_skill_meta exists and has the expected columns.
    Adds missing columns (SQLite supports ADD COLUMN).
    """
    # create table if missing
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS candidate_skill_meta (
            resume_id TEXT,
            skill TEXT,
            months_since_used INTEGER,
            used_in_work INTEGER,
            PRIMARY KEY (resume_id, skill)
        );
    """))
    cols = set(_column_names(conn, "candidate_skill_meta"))
    # add missing columns idempotently
    if "skill" not in cols:
        conn.execute(text("ALTER TABLE candidate_skill_meta ADD COLUMN skill TEXT;"))
    if "months_since_used" not in cols:
        conn.execute(text("ALTER TABLE candidate_skill_meta ADD COLUMN months_since_used INTEGER;"))
    if "used_in_work" not in cols:
        conn.execute(text("ALTER TABLE candidate_skill_meta ADD COLUMN used_in_work INTEGER;"))


# ---- bootstrap / schema
def init_sqlite() -> None:
    """
    Creates tables if missing. Includes a rank cache and meta KV store for invalidation.
    Also ensures candidate_skill_meta has required columns (safe migration).
    """
    eng = _get_engine()
    with eng.begin() as conn:
        # candidates holds raw text plus normalized fields
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS candidates (
                resume_id TEXT PRIMARY KEY,
                name TEXT,
                email TEXT,
                education TEXT,
                experience REAL,
                raw_text TEXT
            );
        """))
        # helpful index for email lookups / dedup
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_candidates_email ON candidates(email);
        """))
        # NEW: functional index to speed case-insensitive lookups
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_candidates_email_lower ON candidates(lower(email));
        """))

        # skills (one-to-many)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS candidate_skills (
                resume_id TEXT,
                skill TEXT,
                strength REAL,
                PRIMARY KEY (resume_id, skill)
            );
        """))
        # NEW: skills metadata for boosts (recency + work); ensure schema fully present
        _ensure_skill_meta_schema(conn)

        # notes/submissions/stages (optional ATS artifacts)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS notes (
                job_id TEXT,
                resume_id TEXT,
                note TEXT,
                ts DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS submissions (
                job_id TEXT,
                resume_id TEXT,
                status TEXT,
                ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (job_id, resume_id)
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS stages (
                job_id TEXT,
                resume_id TEXT,
                stage TEXT,
                meta TEXT,
                ts DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (job_id, resume_id)
            );
        """))
        # minimal jobs table (for ATS flow notifications / client email)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                title TEXT,
                client_email TEXT,
                jd_text TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """))
        # meta KV store (for data_version, etc.)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """))
        # initialize data_version if missing
        conn.execute(text("""
            INSERT INTO meta(key, value) VALUES('data_version','0')
            ON CONFLICT(key) DO NOTHING;
        """))
        # rank cache (cache per JD vector + mix weights for current data_version)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS rank_cache (
                cache_key TEXT PRIMARY KEY,  -- hash of canonicalized JD vector + mix weights + recency + data_version
                data_version INTEGER,
                payload_json TEXT,           -- JSON list of ranked rows (top-N)
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """))


def migrate_add_email() -> None:
    """Idempotent: ensure 'email' column exists on candidates."""
    eng = _get_engine()
    with eng.begin() as conn:
        tables = [r[0] for r in conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )).fetchall()]
        if "candidates" not in tables:
            return
        cols = [r[1] for r in conn.execute(text("PRAGMA table_info(candidates);")).fetchall()]
        if "email" not in cols:
            conn.execute(text("ALTER TABLE candidates ADD COLUMN email TEXT;"))


# ---- meta helpers ----
def _get_meta(key: str, default: str = "") -> str:
    eng = _get_engine()
    with eng.begin() as conn:
        row = conn.execute(text("SELECT value FROM meta WHERE key=:k"), {"k": key}).fetchone()
        return row[0] if row and row[0] is not None else default


def _set_meta(key: str, value: str) -> None:
    eng = _get_engine()
    with eng.begin() as conn:
        conn.execute(text("""
            INSERT INTO meta(key, value) VALUES(:k,:v)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """), {"k": key, "v": value})


def get_data_version() -> int:
    try:
        return int(_get_meta("data_version", "0"))
    except Exception:
        return 0


def bump_data_version() -> int:
    v = get_data_version() + 1
    _set_meta("data_version", str(v))
    return v


# ---- basic exec helper
def _exec(sql: str, params: Dict[str, Any] | None = None):
    eng = _get_engine()
    with eng.begin() as conn:
        return conn.execute(text(sql), params or {})


# ---- jobs helpers (fixes ats_flow imports)
def upsert_job(job_id: str, client_email: str = "", title: str = "", jd_text: str = "") -> None:
    _exec("""
        INSERT INTO jobs(job_id, title, client_email, jd_text)
        VALUES(:job_id, :title, :client_email, :jd_text)
        ON CONFLICT(job_id) DO UPDATE SET
            title=excluded.title,
            client_email=excluded.client_email,
            jd_text=excluded.jd_text;
    """, {"job_id": job_id, "title": title, "client_email": client_email, "jd_text": jd_text})


def get_job(job_id: str) -> Dict[str, Any]:
    eng = _get_engine()
    with eng.begin() as conn:
        row = conn.execute(text("""
            SELECT job_id, title, client_email, jd_text, created_at
            FROM jobs WHERE job_id=:job_id
        """), {"job_id": job_id}).fetchone()
        return {
            "job_id": (row[0] if row else job_id),
            "title": (row[1] if row else ""),
            "client_email": (row[2] if row else ""),
            "jd_text": (row[3] if row else ""),
            "created_at": (row[4] if row else None),
        }


# ---- email-based lookup for dedup ----
def find_resume_id_by_email(email: str) -> Optional[str]:
    if not (email or "").strip():
        return None
    eng = _get_engine()
    with eng.begin() as conn:
        row = conn.execute(text(
            "SELECT resume_id FROM candidates WHERE lower(email)=lower(:e) LIMIT 1;"
        ), {"e": email.strip()}).fetchone()
        return row[0] if row else None


# ---- Chroma (self-contained) ----
def get_chroma(collection_name: str = "skills", persist_dir: str | None = None):
    import chromadb
    from chromadb.config import Settings
    persist_dir = persist_dir or os.getenv("CHROMA_DIR", "./chroma")
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(
        path=persist_dir, settings=Settings(anonymized_telemetry=False)
    )
    try:
        collection = client.get_collection(collection_name)
    except Exception:
        collection = client.create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )
    return client, collection


def _chroma_delete_by_resume(collection, resume_id: str) -> None:
    try:
        # Chroma doesn’t support delete by metadata in all versions; we delete by id prefix scan:
        # Maintain ids as f"{resume_id}::{skill}"
        existing = collection.get(where={"resume_id": resume_id})
        ids = existing.get("ids", []) if isinstance(existing, dict) else []
        if ids:
            collection.delete(ids=ids)
    except Exception:
        pass


def _index_skills_in_chroma(resume_id: str, email: str, skills: Dict[str, float]) -> None:
    """
    Index each skill as a separate vector with metadata {resume_id, skill, strength}.
    (GDPR-minimization: do NOT store email in vector metadata.)
    """
    if not skills:
        return
    try:
        _, col = get_chroma()
    except Exception:
        return  # Chroma not available; retrieval will fall back to DB

    # Remove any prior vectors for this resume
    _chroma_delete_by_resume(col, resume_id)

    # Prepare embeddings
    skill_names = list(skills.keys())
    embs = embed_texts(skill_names)

    ids = [f"{resume_id}::{s}" for s in skill_names]
    # Drop email from metadata to minimize PII footprint
    metadatas = [
        {"resume_id": resume_id, "skill": s, "strength": float(skills.get(s, 0.0))}
        for s in skill_names
    ]
    try:
        col.add(ids=ids, embeddings=embs, metadatas=metadatas, documents=skill_names)
    except Exception:
        # ignore indexing failure; DB fallback still works
        pass


# ---- public APIs used by the app ----
def upsert_candidate(
    rid: str,
    name: str,
    email: str,
    education: str,
    experience: float,
    raw_text: str,
    skills: Dict[str, float],
    skills_meta: Optional[Dict[str, Dict[str, Any]]] = None,  # {"skill": {"months_since_used": int|null, "used_in_work": bool}}
) -> None:
    """
    Upsert candidate and replace skills & skill_meta. Also updates Chroma index for this resume.
    Bumps data_version to invalidate caches.
    - Protects against overwriting existing name/email with blank strings.
    """
    _exec("""
        INSERT INTO candidates(resume_id,name,email,education,experience,raw_text)
        VALUES(:rid,:name,:email,:education,:experience,:raw_text)
        ON CONFLICT(resume_id) DO UPDATE SET
            name=COALESCE(NULLIF(excluded.name,''), candidates.name),
            email=COALESCE(NULLIF(excluded.email,''), candidates.email),
            education=excluded.education,
            experience=excluded.experience,
            raw_text=excluded.raw_text;
    """, {
        "rid": rid, "name": name, "email": email, "education": education,
        "experience": experience, "raw_text": raw_text
    })
    # replace skill map
    _exec("DELETE FROM candidate_skills WHERE resume_id=:rid", {"rid": rid})
    if skills:
        eng = _get_engine()
        with eng.begin() as conn:
            for k, v in skills.items():
                conn.execute(text("""
                    INSERT INTO candidate_skills(resume_id, skill, strength)
                    VALUES(:rid, :skill, :strength)
                """), {"rid": rid, "skill": k, "strength": float(v)})

    # replace skill meta if provided (ensure table/cols exist)
    eng = _get_engine()
    with eng.begin() as conn:
        _ensure_skill_meta_schema(conn)
    _exec("DELETE FROM candidate_skill_meta WHERE resume_id=:rid", {"rid": rid})
    if skills_meta:
        eng = _get_engine()
        with eng.begin() as conn:
            for sk, meta in (skills_meta or {}).items():
                months = meta.get("months_since_used")
                used_in_work = 1 if bool(meta.get("used_in_work", False)) else 0
                # sanitize months
                if months is not None:
                    try:
                        months = int(months)
                        if months < 0 or months > 600:
                            months = None
                    except Exception:
                        months = None
                conn.execute(text("""
                    INSERT INTO candidate_skill_meta(resume_id, skill, months_since_used, used_in_work)
                    VALUES(:rid, :skill, :months, :work)
                """), {"rid": rid, "skill": sk, "months": months, "work": used_in_work})

    # Update Chroma index (email intentionally unused in metadata)
    try:
        _index_skills_in_chroma(rid, email, skills or {})
    except Exception:
        pass

    # bump data version to invalidate caches
    bump_data_version()


def fetch_candidate(resume_id: str) -> Dict[str, Any] | None:
    eng = _get_engine()
    with eng.begin() as conn:
        row = conn.execute(text("""
            SELECT resume_id, name, email, education, experience, raw_text
            FROM candidates WHERE resume_id=:rid
        """), {"rid": resume_id}).fetchone()
        if not row:
            return None
        return {
            "resume_id": row[0], "name": row[1], "email": row[2],
            "education": row[3], "experience": row[4], "raw_text": row[5]
        }


def fetch_all_candidates() -> List[Tuple[str, str, str, float]]:
    eng = _get_engine()
    with eng.begin() as conn:
        rows = conn.execute(text("""
            SELECT resume_id, name, education, experience FROM candidates
        """)).fetchall()
        return [(r[0], r[1], r[2], float(r[3] or 0.0)) for r in rows]


def fetch_all_candidate_skillmaps() -> Dict[str, Dict[str, float]]:
    eng = _get_engine()
    with eng.begin() as conn:
        rows = conn.execute(text("""
            SELECT resume_id, skill, strength FROM candidate_skills
        """)).fetchall()
        out: Dict[str, Dict[str, float]] = {}
        for rid, skill, strength in rows:
            out.setdefault(rid, {})[skill] = float(strength or 0.0)
        return out


# ---- selective loaders for the ranker ----
def load_candidate_profiles(resume_ids: List[str]) -> List[Tuple[str, str, str, float]]:
    if not resume_ids:
        return []
    eng = _get_engine()
    with eng.begin() as conn:
        placeholders = ",".join([f":r{i}" for i in range(len(resume_ids))])
        sql = text(f"""
            SELECT resume_id, name, education, experience
            FROM candidates
            WHERE resume_id IN ({placeholders})
        """)
        params = {f"r{i}": rid for i, rid in enumerate(resume_ids)}
        rows = conn.execute(sql, params).fetchall()
        # keep input order if needed; otherwise return as-is
        found = {(r[0]): (r[0], r[1], r[2], float(r[3] or 0.0)) for r in rows}
        return [found[rid] for rid in resume_ids if rid in found]


def load_skillmaps_for(resume_ids: List[str]) -> Dict[str, Dict[str, float]]:
    if not resume_ids:
        return {}
    eng = _get_engine()
    with eng.begin() as conn:
        placeholders = ",".join([f":r{i}" for i in range(len(resume_ids))])
        sql = text(f"""
            SELECT resume_id, skill, strength
            FROM candidate_skills
            WHERE resume_id IN ({placeholders})
        """)
        params = {f"r{i}": rid for i, rid in enumerate(resume_ids)}
        rows = conn.execute(sql, params).fetchall()
        out: Dict[str, Dict[str, float]] = {}
        for rid, sk, st in rows:
            out.setdefault(rid, {})[sk] = float(st or 0.0)
        return out


def load_skills_meta_for(resume_ids: List[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Returns {resume_id: {skill: {"months_since_used": int|null, "used_in_work": bool}}}
    Resilient: if table or expected columns are missing (older DB), returns {}.
    """
    if not resume_ids:
        return {}
    eng = _get_engine()
    with eng.begin() as conn:
        if not _table_exists(conn, "candidate_skill_meta"):
            return {}
        cols = set(_column_names(conn, "candidate_skill_meta"))
        required = {"resume_id", "skill", "months_since_used", "used_in_work"}
        if not required.issubset(cols):
            # old schema; skip using meta to avoid OperationalError
            return {}

        placeholders = ",".join([f":r{i}" for i in range(len(resume_ids))])
        sql = text(f"""
            SELECT resume_id, skill, months_since_used, used_in_work
            FROM candidate_skill_meta
            WHERE resume_id IN ({placeholders})
        """)
        params = {f"r{i}": rid for i, rid in enumerate(resume_ids)}
        try:
            rows = conn.execute(sql, params).fetchall()
        except Exception:
            return {}
        out: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for rid, sk, months, used in rows:
            out.setdefault(rid, {})[sk] = {
                "months_since_used": (int(months) if months is not None else None),
                "used_in_work": bool(used),
            }
        return out


def load_resume_texts(resume_ids: List[str]) -> Dict[str, str]:
    if not resume_ids:
        return {}
    eng = _get_engine()
    with eng.begin() as conn:
        placeholders = ",".join([f":r{i}" for i in range(len(resume_ids))])
        sql = text(f"""
            SELECT resume_id, raw_text FROM candidates
            WHERE resume_id IN ({placeholders})
        """)
        params = {f"r{i}": rid for i, rid in enumerate(resume_ids)}
        rows = conn.execute(sql, params).fetchall()
        return {r[0]: (r[1] or "") for r in rows}


# ---- Notes/ATS helpers ----
def add_note(job_id: str, resume_id: str, note: str) -> None:
    _exec("""
        INSERT INTO notes(job_id, resume_id, note)
        VALUES(:job_id,:rid,:note)
    """, {"job_id": job_id, "rid": resume_id, "note": note})


def mark_submission(job_id: str, resume_id: str, status: str) -> None:
    _exec("""
        INSERT INTO submissions(job_id, resume_id, status)
        VALUES(:job_id,:rid,:status)
        ON CONFLICT(job_id, resume_id) DO UPDATE SET status=excluded.status
    """, {"job_id": job_id, "rid": resume_id, "status": status})


def set_stage(job_id: str, resume_id: str, stage: str, meta: Dict[str, Any]) -> None:
    _exec("""
        INSERT INTO stages(job_id, resume_id, stage, meta)
        VALUES(:job_id,:rid,:stage,:meta)
        ON CONFLICT(job_id, resume_id) DO UPDATE SET
          stage=excluded.stage, meta=excluded.meta
    """, {"job_id": job_id, "rid": resume_id, "stage": stage, "meta": str(meta)})


# ---- Ranking cache helpers ----
def get_rank_cache(cache_key: str) -> Dict[str, Any] | None:
    eng = _get_engine()
    with eng.begin() as conn:
        row = conn.execute(text("""
            SELECT data_version, payload_json FROM rank_cache WHERE cache_key=:k
        """), {"k": cache_key}).fetchone()
        if not row:
            return None
        try:
            return {"data_version": int(row[0]), "payload": json.loads(row[1] or "[]")}
        except Exception:
            return None


def set_rank_cache(cache_key: str, payload: List[Dict[str, Any]], data_version: int) -> None:
    eng = _get_engine()
    with eng.begin() as conn:
        conn.execute(text("""
            INSERT INTO rank_cache(cache_key, data_version, payload_json)
            VALUES(:k, :v, :p)
            ON CONFLICT(cache_key) DO UPDATE SET
              data_version=excluded.data_version,
              payload_json=excluded.payload_json,
              created_at=CURRENT_TIMESTAMP
        """), {"k": cache_key, "v": int(data_version), "p": json.dumps(payload)})


# ---- Retrieval / Shortlist via Chroma ----
def _try_query_chroma(query_skills: List[str], per_skill_n: int = 200) -> List[Tuple[str, float]]:
    """
    Query Chroma for each skill and combine results into (resume_id, weighted_similarity).
    Returns a scored list; caller may post-filter and cap at K.
    """
    if not query_skills:
        return []
    try:
        _, col = get_chroma()
    except Exception:
        return []

    # embed the query skill tokens
    embs = embed_texts(query_skills) if _HAS_EMB else embed_texts(query_skills)

    # Accumulate weighted similarity per resume_id
    totals: Dict[str, float] = {}
    for i, emb in enumerate(embs):
        # query per skill; cosine distance → convert to similarity
        try:
            res = col.query(query_embeddings=[emb], n_results=per_skill_n)
        except Exception:
            continue
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        for j, rid_meta in enumerate(metas):
            rid = (rid_meta or {}).get("resume_id")
            if not rid:
                continue
            dist = float(dists[j] if j < len(dists) else 1.0)
            sim = 1.0 - dist  # cosine
            totals[rid] = totals.get(rid, 0.0) + sim

    return sorted(totals.items(), key=lambda kv: -kv[1])


def shortlist_by_jd_skills(jd_weights: Dict[str, float], k: int = 500) -> List[str]:
    """
    Retrieval-first shortlist using the Chroma skill index.
    - Embed each JD skill and query Chroma (n≈200 per skill).
    - Weight similarities by JD weight; sum per resume_id.
    - Return top-k resume_ids.
    Falls back to a deterministic DB scan if Chroma is unavailable.
    """
    jd_weights = { (s or "").strip().lower(): float(w) for s, w in (jd_weights or {}).items() if s and float(w) > 0.0 }
    if not jd_weights:
        return []

    # Try Chroma first
    try:
        _, col = get_chroma()
        # embed once for all skills
        skills = list(jd_weights.keys())
        embs = embed_texts(skills) if _HAS_EMB else embed_texts(skills)

        totals: Dict[str, float] = {}
        for skill, emb in zip(skills, embs):
            w = float(jd_weights.get(skill, 0.0))
            if w <= 0:
                continue
            try:
                res = col.query(query_embeddings=[emb], n_results=int(os.getenv("SHORTLIST_PER_SKILL", "250")))
            except Exception:
                continue
            metas = (res.get("metadatas") or [[]])[0]
            dists = (res.get("distances") or [[]])[0]
            for j, rid_meta in enumerate(metas):
                rid = (rid_meta or {}).get("resume_id")
                if not rid:
                    continue
                dist = float(dists[j] if j < len(dists) else 1.0)
                sim = 1.0 - dist  # cosine → similarity
                totals[rid] = totals.get(rid, 0.0) + (w * sim)

        ordered = sorted(totals.items(), key=lambda kv: -kv[1])
        return [rid for rid, _ in ordered[:max(1, int(k))]]
    except Exception:
        pass

    # -------- Fallback: DB-only deterministic shortlist (no Chroma) --------
    # Score = sum_{jd skill}( jd_w * candidate_strength(skill) / 5 ), then take top-k.
    eng = _get_engine()
    with eng.begin() as conn:
        rows = conn.execute(text("""
            SELECT cs.resume_id, cs.skill, cs.strength
            FROM candidate_skills cs
        """)).fetchall()
    acc: Dict[str, float] = {}
    for rid, sk, st in rows:
        w = jd_weights.get((sk or "").strip().lower(), 0.0)
        if w > 0:
            acc[rid] = acc.get(rid, 0.0) + (w * float(st or 0.0) / 5.0)
    ordered = sorted(acc.items(), key=lambda kv: -kv[1])
    return [rid for rid, _ in ordered[:max(1, int(k))]]


# --------- convenience: load everything for ranking for the shortlisted IDs
def load_for_ranking(resume_ids: List[str]) -> Tuple[
    List[Tuple[str, str, str, float]],
    Dict[str, Dict[str, float]],
    Dict[str, Dict[str, Dict[str, Any]]],
    Dict[str, str],
]:
    """
    Returns a tuple to feed ranker:
      rows_all, skillmaps, skills_meta_map, resume_texts
    """
    rows_all = load_candidate_profiles(resume_ids)
    skillmaps = load_skillmaps_for(resume_ids)
    skills_meta_map = load_skills_meta_for(resume_ids)
    resume_texts = load_resume_texts(resume_ids)
    return rows_all, skillmaps, skills_meta_map, resume_texts
