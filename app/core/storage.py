import os
import json
from typing import Dict, List, Tuple, Any, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

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

# ---- bootstrap / schema
def init_sqlite() -> None:
    """
    Creates tables if missing. Includes a small rank cache and meta KV store for invalidation.
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
        # skills (one-to-many)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS candidate_skills (
                resume_id TEXT,
                skill TEXT,
                strength REAL,
                PRIMARY KEY (resume_id, skill)
            );
        """))
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
        # rank cache (cache per JD vector + exp_weight for current data_version)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS rank_cache (
                cache_key TEXT PRIMARY KEY,  -- hash of canonicalized JD vector + exp_weight
                data_version INTEGER,
                payload_json TEXT,           -- JSON list of ranked rows (store a reasonable head)
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
        if "candidates" not in tables: return
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
    if not (email or "").strip(): return None
    eng = _get_engine()
    with eng.begin() as conn:
        row = conn.execute(text(
            "SELECT resume_id FROM candidates WHERE lower(email)=lower(:e) LIMIT 1;"
        ), {"e": email.strip()}).fetchone()
        return row[0] if row else None

# ---- public APIs used by the app ----
def upsert_candidate(
    rid: str, name: str, email: str, education: str, experience: float,
    raw_text: str, skills: Dict[str, float],
) -> None:
    _exec("""
        INSERT INTO candidates(resume_id,name,email,education,experience,raw_text)
        VALUES(:rid,:name,:email,:education,:experience,:raw_text)
        ON CONFLICT(resume_id) DO UPDATE SET
            name=excluded.name,
            email=excluded.email,
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
    # bump data version to invalidate caches
    bump_data_version()

def fetch_candidate(resume_id: str) -> Dict[str, Any] | None:
    eng = _get_engine()
    with eng.begin() as conn:
        row = conn.execute(text("""
            SELECT resume_id, name, email, education, experience, raw_text
            FROM candidates WHERE resume_id=:rid
        """), {"rid": resume_id}).fetchone()
        if not row: return None
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
        if not row: return None
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
