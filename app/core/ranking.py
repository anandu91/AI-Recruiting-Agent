# app/core/ranking.py
"""
Deterministic ranking: Skills + Experience (no education).
Experience contributes ONLY if the resume shows usage of at least one JD skill
within the last N months (default 24). No LLM at match time.

Per-skill boosts:
  - +2.0 if the skill is the candidate’s single strongest (main) skill
  - Recency bonus:
      +3.0 if used within 12 months
      +2.0 if used within 24 months
      +1.0 if the skill appears anywhere in the resume (even without dates)
      +0.0 if not mentioned at all
  - +1.0 extra if the skill appears in a WORK EXPERIENCE window
    (±600 chars around dated job ranges)
Final per-skill strength is capped at 5.0.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from functools import lru_cache
from typing import List, Dict, Any, Tuple, Optional
import calendar

# Prefer embeddings if available; otherwise our embeddings module provides
# a robust deterministic fallback (32-D hash+sin/cos) transparently.
try:
    from app.core.embeddings import embed_texts
    _HAS_EMB = True
except Exception:
    _HAS_EMB = False


# ---------- text normalization (cheap, language-agnostic) ----------

# e.g., trailing "3", "3.10", "v3", "v3.10" as a separate token
_VERSION_TAIL = re.compile(r"(?:^|\b)(?:v)?\d+(?:\.\d+){0,2}\b$")

def _key(s: str) -> str:
    s = (s or "").lower().strip()
    # keep symbols meaningful for tech: + # . / -.
    s = re.sub(r"[^\w+#./-]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _strip_trailing_version(s: str) -> str:
    parts = (s or "").split()
    if parts and _VERSION_TAIL.match(parts[-1] or ""):
        parts = parts[:-1]
    return " ".join(parts).strip()

def _base_token(raw: str) -> str:
    """
    Lightweight base tokenization:
      - lowercase + collapse whitespace
      - drop explicit trailing version tokens: "java 8" -> "java"
      - carefully drop inline jammed versions when they are clearly version-like:
        "python3" -> "python", "c++11" -> "c++"
        (but KEEP short letter+digit identifiers like "db2", "s3", "ec2")
      - collapse separators: node js / node.js -> nodejs
    """
    k = _key(raw)
    k = _strip_trailing_version(k)

    # handle jammed suffix versions like python3, java8, nodejs14, c++11
    no_space = k.replace(" ", "")
    m = re.match(r"(?i)([a-z][a-z0-9+#.]*?)(?:v)?(\d{1,2}(?:\.\d+)*)$", no_space)
    if m:
        letters, digits = m.group(1), m.group(2)
        # Heuristic guardrails:
        # - Keep short letter+digit tech tokens intact (<=4 chars total), e.g., db2, s3, ec2, gke
        # - Only strip when token is reasonably long (>4) and has enough alpha context (>=3),
        #   which strongly suggests a version suffix rather than the identity of the tech.
        if len(no_space) > 4 and len(letters) >= 3:
            k = letters

    # collapse common separators
    merged = re.sub(r"[.\s/]+", "", k)
    return merged or k


# ---------- embeddings + clustering ----------

def _cos(u: List[float], v: List[float]) -> float:
    num = sum(a * b for a, b in zip(u, v))
    du = math.sqrt(sum(a * a for a in u)) or 1.0
    dv = math.sqrt(sum(b * b for b in v)) or 1.0
    return num / (du * dv)

@lru_cache(maxsize=2048)
def _embed_cached(token: str) -> Tuple[float, ...]:
    if not _HAS_EMB:
        # rarely used: import failure path
        return (float(abs(hash(token)) % 997) / 997.0, )
    try:
        vec = embed_texts([token])[0]
        return tuple(float(x) for x in vec)
    except Exception:
        # deterministic per-token fallback if backend fails at runtime
        return (float(abs(hash(token)) % 997) / 997.0, )

def _embed_many(tokens: List[str]) -> List[Tuple[float, ...]]:
    if not _HAS_EMB:
        return [_embed_cached(t) for t in tokens]
    try:
        embs = embed_texts(tokens)
        return [tuple(float(x) for x in e) for e in embs]
    except Exception:
        # fallback gracefully per-token
        return [_embed_cached(t) for t in tokens]

def _build_clusters(tokens: List[str], sim_thresh: float = 0.90) -> Dict[str, str]:
    """
    Greedy agglomeration: group tokens whose cosine similarity >= sim_thresh.
    Representative is the shortest string (ties -> lexicographic).

    Stricter threshold + guardrails to prevent over-merge of short/digit-suffixed tokens
    (e.g., 'db2' should not cluster with 'dbt'; 's3' should not cluster with 'sql').
    """
    if not tokens:
        return {}

    # unique order-stable
    seen, uniq = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t); uniq.append(t)

    vecs = _embed_many(uniq)

    parent = {i: i for i in range(len(uniq))}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    # pairwise union (n is small in practice)
    for i in range(len(uniq)):
        vi = vecs[i]
        for j in range(i + 1, len(uniq)):
            vj = vecs[j]
            ti, tj = uniq[i], uniq[j]

            # Guardrails to avoid over-merge of short/different-style tokens:
            # 1) don't cluster very short tokens (<=4 chars) with anything else
            if len(ti) <= 4 or len(tj) <= 4:
                continue
            # 2) don't cluster tokens when one ends with a digit and the other doesn't
            #    (e.g., db2 vs dbt)
            if ti[-1].isdigit() != tj[-1].isdigit():
                continue

            if _cos(vi, vj) >= sim_thresh:
                union(i, j)

    # collect clusters
    clusters: Dict[int, List[int]] = {}
    for i in range(len(uniq)):
        r = find(i)
        clusters.setdefault(r, []).append(i)

    # choose representative per cluster
    rep_for_index: Dict[int, str] = {}
    for root, members in clusters.items():
        names = [uniq[k] for k in members]
        names_sorted = sorted(names, key=lambda s: (len(s), s))
        rep = names_sorted[0]
        for k in members:
            rep_for_index[k] = rep

    # final mapping: token -> representative
    return {tok: rep_for_index[idx] for idx, tok in enumerate(uniq)}


# ---------- canonicalization pipeline (no manual alias list) ----------

def _canon_vocab_from(
    jd_skills: List[Dict[str, Any]],
    all_skillmaps: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Build a canonicalization map for BOTH sides:
    - Normalize with base_token()
    - Cluster via embeddings
    Returns:
      (raw->canon map, base->canon map)  [raw map currently unused]
    """
    vocab: List[str] = []

    # collect from JD
    for it in jd_skills or []:
        nm = (it.get("name") or "").strip()
        if nm:
            vocab.append(_base_token(nm))

    # collect from candidate maps
    for m in (all_skillmaps or {}).values():
        for k in (m or {}).keys():
            vocab.append(_base_token(k))

    # unique tokens
    uniq, seen = [], set()
    for v in vocab:
        if v and v not in seen:
            seen.add(v); uniq.append(v)

    if not uniq:
        return {}, {}

    cluster_map = _build_clusters(uniq)  # base_token -> representative

    raw_to_canon: Dict[str, str] = {}
    base_to_canon: Dict[str, str] = {b: cluster_map.get(b, b) for b in uniq}
    return raw_to_canon, base_to_canon

def _canon_candidate_map(raw_map: Dict[str, float], base_to_canon: Dict[str, str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for raw, v in (raw_map or {}).items():
        b = _base_token(raw)
        c = base_to_canon.get(b, b)
        v = float(v or 0.0)
        if v <= 0:
            continue
        # if multiple variants map to same canonical, keep the max strength
        out[c] = max(out.get(c, 0.0), v)
    return out

def _canon_jd_list(jd_skills: List[Dict[str, Any]], base_to_canon: Dict[str, str]) -> List[Dict[str, Any]]:
    tmp: Dict[str, float] = {}
    for item in (jd_skills or []):
        nm = (item.get("name") or "").strip()
        if not nm:
            continue
        wt = float(item.get("weight", 0.0))
        if wt <= 0:
            continue
        b = _base_token(nm)
        c = base_to_canon.get(b, b)
        # if JD listed synonyms, keep the higher weight per canonical skill
        tmp[c] = max(tmp.get(c, 0.0), wt)
    return [{"name": k, "weight": v} for k, v in tmp.items()]


# ---------- scoring (pure math) ----------

def normalize_weights(skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    s = sum(float(x.get("weight", 0.0)) for x in skills) or 1.0
    return [
        {"name": x["name"], "weight": float(x.get("weight", 0.0)) / s}
        for x in skills
        if x.get("name")
    ]

def compute_weighted_score(jd_skills: List[Dict[str, Any]], cand_map: Dict[str, float]) -> float:
    """
    Compute Σ(w * strength) / Σ(w). Candidate strength is clipped to [0,5].
    Returns a value on 0..5.
    """
    if not jd_skills:
        return 0.0
    num, den = 0.0, 0.0
    for item in jd_skills:
        nm = (item.get("name") or "").strip()
        wt = float(item.get("weight", 0.0))
        if not nm or wt <= 0:
            continue
        strength = float(cand_map.get(nm, 0.0))
        num += wt * max(0.0, min(5.0, strength))
        den += wt
    return (num / den) if den > 0 else 0.0


# ---------- recency + relevance helpers (JD skill used in recent section) ----------

# Build robust month lookup
_FULL_TO_NUM = {calendar.month_name[i].lower(): i for i in range(1, 13)}
_ABBR_TO_NUM = {calendar.month_abbr[i].lower(): i for i in range(1, 13)}
_FULL_TO_NUM["sept"] = 9  # common variant

def _month_from_str(s: str) -> Optional[int]:
    s = (s or "").lower().strip().rstrip(".")
    if s in _FULL_TO_NUM:
        return _FULL_TO_NUM[s]
    # allow long names like "september"
    if s[:3] in _ABBR_TO_NUM:
        return _ABBR_TO_NUM[s[:3]]
    if s.startswith("sept"):
        return 9
    return None

def _year2(s: Optional[str]) -> Optional[int]:
    """
    Parse 2-digit years. Assumption window: 90–99 -> 1990s, 00–89 -> 2000–2089.
    """
    if not s:
        return None
    s = s.strip().strip("'")
    if re.fullmatch(r"\d{2}", s):
        yy = int(s)
        return 1900 + yy if yy >= 90 else 2000 + yy
    return None

def _parse_year_token(tok: Optional[str]) -> Optional[int]:
    if tok is None:
        return None
    if re.fullmatch(r"19[8-9]\d|20[0-5]\d", tok):
        return int(tok)
    return _year2(tok)

def _parse_month_year(token: str) -> Optional[datetime]:
    """
    Parse a single month-year-ish token safely.
    Accepts:
      - Sep 2023 / September 2023
      - 2023-09 / 09/2023 / 2023/9
      - 2023  (treated as Dec 1, 2023)
      - Jul'21 (→ 2021-07-01)
      - 07/21 (→ 2021-07-01) with 2-digit year mapping
    Returns None if it can't be parsed.
    """
    t = (token or "").strip().lower().replace("–", "-").replace("—", "-").replace("\\", "/")
    t = t.replace("’", "'")

    # "September 2023" or "Sep 2023" or "Sep'23"
    m = re.match(r"\b([a-z]{3,9})\.?\s*'?(19[8-9]\d|20[0-5]\d|\d{2})\b", t, flags=re.I)
    if m:
        mm = _month_from_str(m.group(1))
        yy = _parse_year_token(m.group(2))
        if mm and yy:
            return datetime(yy, mm, 1)

    # "2023-09" or "2023/09"
    m = re.match(r"\b(19[8-9]\d|20[0-5]\d)[-/](\d{1,2})\b", t)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return datetime(year, month, 1)

    # "09/2023" or "9-2023"
    m = re.match(r"\b(\d{1,2})[-/](19[8-9]\d|20[0-5]\d)\b", t)
    if m:
        month, year = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return datetime(year, month, 1)

    # "07/21" or "7-21"  (2-digit year)
    m = re.match(r"\b(\d{1,2})[-/](\d{2})\b", t)
    if m:
        month, yy2 = int(m.group(1)), _year2(m.group(2))
        if 1 <= month <= 12 and yy2:
            return datetime(yy2, month, 1)

    # "2023" or "1999" only
    m = re.match(r"\b(19[8-9]\d|20[0-5]\d)\b", t)
    if m:
        return datetime(int(m.group(1)), 12, 1)

    # "’21" or "'21" (assume December)
    m = re.match(r"^'(\d{2})$", t)
    if m:
        yy2 = _year2(m.group(1))
        if yy2:
            return datetime(yy2, 12, 1)

    return None

def _months_between(a: datetime, b: datetime) -> int:
    return (a.year - b.year) * 12 + (a.month - b.month)

def _recent_relevant_usage(resume_text: str, jd_canon_names: List[str], recency_months: int) -> bool:
    """
    True if a text window associated with a date range ending ≤ recency_months
    contains at least one canonical JD skill.
    Used ONLY to gate the Experience component (not the per-skill recency bonus).
    """
    if not (resume_text or "").strip():
        return False
    txt = resume_text.lower().replace("–", "-").replace("—", "-")
    now = datetime.utcnow()
    jd_tokens = [_base_token(s or "") for s in jd_canon_names if s]
    if not jd_tokens:
        return False

    monthyear = (
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+(?:19[8-9]\d|20[0-5]\d)"
        r"|\b(?:19[8-9]\d|20[0-5]\d)[-/]\d{1,2}\b"
        r"|\b\d{1,2}[-/](?:19[8-9]\d|20[0-5]\d)\b"
        r"|\b(?:19[8-9]\d|20[0-5]\d)\b"
        r"|\b'\d{2}\b"
    )
    for m in re.finditer(rf"({monthyear})\s*(?:to|-|–|—)\s*(present|current|now|{monthyear})", txt, re.I):
        end_tok = (m.group(2) or "").strip()
        end_dt = now if re.match(r"(present|current|now)$", end_tok, re.I) else _parse_month_year(end_tok)
        if not end_dt:
            continue
        if _months_between(now, end_dt) > recency_months:
            continue
        a = max(0, m.start() - 400)
        b = min(len(txt), m.end() + 800)
        window = txt[a:b]
        collapsed = re.sub(r"[.\s/]+", "", _key(window))
        if any(tok and tok in collapsed for tok in jd_tokens):
            return True

    # fallback: single month-year mentions
    for m in re.finditer(monthyear, txt, re.I):
        dt = _parse_month_year(m.group(0))
        if not dt or _months_between(now, dt) > recency_months:
            continue
        a = max(0, m.start() - 300)
        b = min(len(txt), m.end() + 600)
        window = txt[a:b]
        collapsed = re.sub(r"[.\s/]+", "", _key(window))
        if any(tok and tok in collapsed for tok in jd_tokens):
            return True

    return False

# === BOOSTS via stored metadata (fast path) ================================

def _per_skill_boosts_from_meta(
    cand_meta: Dict[str, Dict[str, Any]],
    skill_token: str
) -> Tuple[float, float]:
    """
    Compute (recency_boost, work_boost) from stored metadata:
      - months_since_used: int|null
      - used_in_work: bool
    """
    meta = cand_meta.get(skill_token) or {}
    months = meta.get("months_since_used")
    used_in_work = bool(meta.get("used_in_work", False))

    # recency mapping
    if isinstance(months, int):
        if months <= 12:
            recency = 3.0
        elif months <= 24:
            recency = 2.0
        else:
            recency = 1.0  # seen historically but older
    else:
        # unknown months: grant +1 only if the skill exists in the candidate map (presence)
        # caller ensures this is only invoked for skills in cmap
        recency = 1.0

    work = 1.0 if used_in_work else 0.0
    return recency, work

def _recent_relevant_from_meta(
    cand_meta: Dict[str, Dict[str, Any]],
    jd_canon_names: List[str],
    recency_months: int
) -> bool:
    for s in jd_canon_names:
        m = cand_meta.get(s) or {}
        months = m.get("months_since_used")
        if isinstance(months, int) and months <= recency_months:
            return True
    return False

# === BOOSTS: text-based (fallback when no meta) ============================

def _per_skill_boosts(resume_text: str, skill_token: str) -> Tuple[float, float]:
    """
    Compute (recency_boost, work_boost) for a single canonical skill by scanning text.

    RECENCY:
      +3.0 if used within 12 months
      +2.0 if used within 24 months
      +1.0 if the skill appears anywhere (even without dates)
      +0.0 if not mentioned

    WORK:
      +1.0 if the skill is mentioned within ±600 chars of a dated employment range.
    """
    if not (resume_text or "").strip() or not (skill_token or "").strip():
        return 0.0, 0.0

    txt = (resume_text or "").lower().replace("–", "-").replace("—", "-")
    now = datetime.utcnow()
    tok = _base_token(skill_token)

    # quick presence check anywhere in the doc
    collapsed_all = re.sub(r"[.\s/]+", "", _key(txt))
    appears_anywhere = bool(tok and tok in collapsed_all)

    monthyear = (
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+(?:19[8-9]\d|20[0-5]\d)"
        r"|\b(?:19[8-9]\d|20[0-5]\d)[-/]\d{1,2}\b"
        r"|\b\d{1,2}[-/](?:19[8-9]\d|20[0-5]\d)\b"
        r"|\b(?:19[8-9]\d|20[0-5]\d)\b"
        r"|\b'\d{2}\b"
    )

    recency_best: Optional[int] = None
    work_hit = False

    # Employment ranges like "Jan 2021 - Present"
    for m in re.finditer(rf"({monthyear})\s*(?:to|-|–|—)\s*(present|current|now|{monthyear})", txt, re.I):
        end_tok = (m.group(2) or "").strip()
        end_dt = now if re.match(r"(present|current|now)$", end_tok, re.I) else _parse_month_year(end_tok)
        if not end_dt:
            continue

        a = max(0, m.start() - 600)
        b = min(len(txt), m.end() + 600)
        window = txt[a:b]
        collapsed = re.sub(r"[.\s/]+", "", _key(window))

        # WORK bonus trigger
        if tok and tok in collapsed:
            work_hit = True

        # recency estimation from this window
        months = _months_between(now, end_dt)
        if recency_best is None or months < recency_best:
            recency_best = months

    # Also consider isolated month-year mentions
    for m in re.finditer(monthyear, txt, re.I):
        dt = _parse_month_year(m.group(0))
        if not dt:
            continue
        a = max(0, m.start() - 500)
        b = min(len(txt), m.end() + 500)
        window = txt[a:b]
        collapsed = re.sub(r"[.\s/]+", "", _key(window))
        if tok and tok in collapsed:
            months = _months_between(now, dt)
            if recency_best is None or months < recency_best:
                recency_best = months

    # Map recency_best → bonus (3/2/1/0)
    if recency_best is not None:
        if recency_best <= 12:
            recency_boost = 3.0
        elif recency_best <= 24:
            recency_boost = 2.0
        else:
            recency_boost = 1.0 if appears_anywhere else 0.0
    else:
        recency_boost = 1.0 if appears_anywhere else 0.0

    work_boost = 1.0 if work_hit else 0.0
    return recency_boost, work_boost


# ---------- main ranking (Skills + Experience only) ----------

def _normalize_mix_weights(mix: Dict[str, float] | None) -> Dict[str, float]:
    # fixed defaults: skills 0.85, experience 0.15
    defaults = {"skills": 0.85, "experience": 0.15}
    if not mix:
        return defaults
    m = {"skills": max(0.0, float(mix.get("skills", defaults["skills"]))),
         "experience": max(0.0, float(mix.get("experience", defaults["experience"]))),}
    s = (m["skills"] + m["experience"]) or 1.0
    m["skills"] /= s; m["experience"] /= s
    return m

# Internal constants and helper to derive primary skills from JD weights
PRIMARY_CUM_WEIGHT = 0.60   # take top skills until ≥60% cumulative weight
PRIMARY_MIN = 3
PRIMARY_MAX = 8

def _derive_primary_names(jd_norm: List[Dict[str, Any]]) -> List[str]:
    """
    Choose primary skills purely from normalized JD weights (deterministic).
    We sort by weight desc (then name), take skills until cumulative weight
    reaches PRIMARY_CUM_WEIGHT, clamped between PRIMARY_MIN and PRIMARY_MAX.
    """
    if not jd_norm:
        return []
    sorted_items = sorted(
        [{"name": x["name"], "weight": float(x.get("weight", 0.0))} for x in jd_norm],
        key=lambda d: (-d["weight"], d["name"])
    )
    out, cum = [], 0.0
    for it in sorted_items:
        if len(out) < PRIMARY_MIN or (cum < PRIMARY_CUM_WEIGHT and len(out) < PRIMARY_MAX):
            out.append(it["name"])
            cum += it["weight"]
        else:
            break
    return out

def rank_candidates_weighted(
    jd_skills: List[Dict[str, Any]],
    rows_all: List[Tuple[str, str, str, float]],   # (resume_id, name, education[str, unused], years)
    all_skillmaps: Dict[str, Dict[str, float]],
    top_k: int = 10,
    mix_weights: Dict[str, float] | None = None,
    jd_text: Optional[str] = None,                 # unused here; kept for signature stability
    recency_months: int = 24,                      # align with UI gate
    resume_texts: Optional[Dict[str, str]] = None, # resume_id -> raw_text for recency check
    skills_meta_map: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,  # fast boosts
    # Back-compat shim for old callers/tests:
    exp_weight: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Rank candidates:
      1) Canonicalize skill names on both sides (no alias list).
      2) Normalize JD weights to sum to 1.
      3) Skills score with BOOSTS:
         - Per-skill final_strength = min(5.0, raw + main(+2.0) + recency(+3/+2/+1/+0) + work(+1.0))
         - contribution_0to5 = jd_weight * final_strength
         - Skills Score (0..10) = (Σ contributions / 5.0) * 10.0
      4) Experience component (0..10) contributes ONLY if resume shows at least one JD skill
         in a date window that ends within `recency_months`. (Uses skills_meta_map if present.)
      5) Final score = skills*mix['skills'] + experience*mix['experience'].
    """
    # Back-compat: if caller passed exp_weight, derive mix_weights
    if exp_weight is not None and (mix_weights is None):
        ew = max(0.0, min(1.0, float(exp_weight)))
        mix_weights = {"skills": 1.0 - ew, "experience": ew}

    mix = _normalize_mix_weights(mix_weights)

    # 1) Build canonical mapping
    _, base_to_canon = _canon_vocab_from(jd_skills, all_skillmaps)

    # 2) Canonicalize both sides
    jd_canon = _canon_jd_list(jd_skills, base_to_canon)
    jd_norm = normalize_weights(jd_canon)

    # compute a deterministic primary list once for all candidates
    primary_names = _derive_primary_names(jd_norm)

    out: List[Dict[str, Any]] = []
    jd_names = [s["name"] for s in jd_norm]

    for rid, name, _edu_unused, exp_years in rows_all:
        cmap_raw = all_skillmaps.get(rid, {}) or {}
        cmap = _canon_candidate_map(cmap_raw, base_to_canon)

        # candidate's main skill: highest strength; ties → lexicographic first
        main_skill = None
        if cmap:
            max_strength = max(cmap.values())
            best = [k for k, v in cmap.items() if v == max_strength]
            main_skill = sorted(best)[0] if best else None

        per_skill_rows: List[Dict[str, Any]] = []
        contrib_sum = 0.0

        resume_txt = (resume_texts or {}).get(rid, "") if resume_texts is not None else ""
        cand_meta = (skills_meta_map or {}).get(rid) if skills_meta_map is not None else None

        for item in jd_norm:
            s = item["name"]
            w = float(item.get("weight", 0.0))
            raw_strength = float(cmap.get(s, 0.0))
            is_main = (s == main_skill)

            main_boost = 2.0 if is_main else 0.0

            if cand_meta is not None and s in cmap:  # fast path: use stored meta
                recency_boost, work_boost = _per_skill_boosts_from_meta(cand_meta, s)
            elif resume_texts is not None and resume_txt:
                recency_boost, work_boost = _per_skill_boosts(resume_txt, s)
            else:
                # no info → only presence yields +1 recency if present, else 0
                recency_boost, work_boost = ((1.0 if s in cmap else 0.0), 0.0)

            final_strength = min(5.0, max(0.0, raw_strength + main_boost + recency_boost + work_boost))
            contribution = w * final_strength
            contrib_sum += contribution

            per_skill_rows.append({
                "skill": s,
                "raw_strength": round(raw_strength, 3),
                "is_main": bool(is_main),
                "recency_boost": round(recency_boost, 3),
                "work_boost": round(work_boost, 3),
                "final_strength": round(final_strength, 3),
                "jd_weight": round(w, 6),
                "contribution_0to5": round(contribution, 6),
            })

        # sort rows by contribution desc for consistency
        per_skill_rows.sort(key=lambda r: (-float(r.get("contribution_0to5", 0.0)), r.get("skill","")))

        # Skills score 0..10
        skills_score_10 = ( (contrib_sum / 5.0) * 10.0 ) if jd_norm else 0.0

        # Experience gating
        exp_component_10 = 0.0
        if (exp_years or 0.0):
            if cand_meta is not None:
                recent_and_relevant = _recent_relevant_from_meta(cand_meta, jd_names, recency_months)
            elif resume_texts is not None:
                recent_and_relevant = _recent_relevant_usage(resume_txt, jd_names, recency_months)
            else:
                recent_and_relevant = False

            if recent_and_relevant:
                exp_component_10 = min(10.0, float(exp_years or 0.0))
        else:
            recent_and_relevant = False

        final_10 = mix["skills"] * skills_score_10 + mix["experience"] * exp_component_10

        matched = [s["name"] for s in jd_norm if s["name"] in cmap]
        missing = [s["name"] for s in jd_norm if s["name"] not in cmap]

        out.append({
            "resume_id": rid,
            "name": name,
            "experience": float(exp_years or 0.0),
            "score_skills_10": round(skills_score_10, 2),
            "score_exp_10": round(exp_component_10, 2),
            "score_norm_10": round(final_10, 2),
            "recent_relevant": bool(exp_component_10 > 0.0),
            "matched_skills": matched,
            "missing_skills": missing,
            "primary_names": list(primary_names),
            "per_skill_rows": per_skill_rows,
        })

    # Deterministic ordering:
    def _sort_key(row: Dict[str, Any]):
        return (-float(row.get("score_norm_10", 0.0)),
                -float(row.get("score_skills_10", 0.0)),
                -float(row.get("experience", 0.0)),
                str(row.get("name", "") or ""),
                str(row.get("resume_id", "") or ""))

    out.sort(key=_sort_key)
    return out[:max(1, int(top_k))]


# ---------- optional: exact per-candidate explanation for UI/tests ----------

def explain_candidate_score(
    jd_skills: List[Dict[str, Any]],
    resume_id: str,
    all_skillmaps: Dict[str, Dict[str, float]],
) -> Tuple[List[Dict[str, Any]], float, float]:
    """
    Returns (rows, score_0to5, score_0to10) explaining each JD skill's contribution
    for this candidate. Uses the same canonicalization + normalization as ranking().

    rows: [{"skill","weight_norm","candidate_score","contribution","matched"}]
    """
    # 1) canonicalize both sides using the full vocabulary
    _, base_to_canon = _canon_vocab_from(jd_skills, all_skillmaps)
    jd_canon = _canon_jd_list(jd_skills, base_to_canon)
    jd_norm  = normalize_weights(jd_canon)  # weights sum to 1.0

    cand_raw = all_skillmaps.get(resume_id, {}) or {}
    cand_map = _canon_candidate_map(cand_raw, base_to_canon)

    # 2) compute contributions
    rows = []
    num, den = 0.0, 0.0
    for item in jd_norm:
        s = item["name"]
        w = float(item.get("weight", 0.0))
        c = float(cand_map.get(s, 0.0))
        c_clip = max(0.0, min(5.0, c))
        contrib = w * c_clip
        rows.append({
            "skill": s,
            "weight_norm": round(w, 6),
            "candidate_score": round(c, 3),
            "contribution": round(contrib, 6),
            "matched": bool(c > 0.0),
        })
        num += contrib
        den += w

    score_0to5 = (num / den) if den > 0 else 0.0
    score_0to10 = (score_0to5 / 5.0) * 10.0
    return rows, round(score_0to5, 6), round(score_0to10, 4)
