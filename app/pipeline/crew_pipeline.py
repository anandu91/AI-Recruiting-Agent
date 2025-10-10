# app/pipeline/crew_pipeline.py
from __future__ import annotations

import os
import json
import hashlib
from typing import Dict, Any, List, Tuple, Optional

from app.agents.jd_agent import jd_extract_skills, build_jd_agent
from app.core.storage import (
    fetch_candidate,
    fetch_all_candidate_skillmaps,
    shortlist_by_jd_skills,
    load_for_ranking,
    get_rank_cache,
    set_rank_cache,
    get_data_version,
)
from app.core.ranking import rank_candidates_weighted


def _enforce_hints(
    skills: List[Dict[str, Any]],
    ui_hints: Optional[Dict[str, List[str]]]
) -> List[Dict[str, Any]]:
    """Deterministic enforcement so user hints aren’t ignored."""
    if not ui_hints:
        return skills

    primary   = {s.strip().lower() for s in (ui_hints.get("primary") or []) if s.strip()}
    secondary = {s.strip().lower() for s in (ui_hints.get("secondary") or []) if s.strip()}
    exclude   = {s.strip().lower() for s in (ui_hints.get("exclude") or []) if s.strip()}

    filtered: List[Dict[str, Any]] = []
    seen = set()
    for it in skills:
        nm = (it.get("name") or "").strip()
        if not nm:
            continue
        lo = nm.lower()
        if lo in exclude:
            continue
        w = float(it.get("weight", 0.0))
        if lo in primary:
            w = max(0.90, w)
        elif lo in secondary and w < 0.60:
            w = 0.60
        filtered.append({"name": nm, "weight": max(0.0, min(1.0, w))})
        seen.add(lo)

    # Add hinted-but-missing items at minimum weights
    for nm in primary:
        if nm not in seen and nm not in exclude:
            filtered.append({"name": nm, "weight": 0.92})
    for nm in secondary:
        if nm not in seen and nm not in exclude and nm not in primary:
            filtered.append({"name": nm, "weight": 0.70})

    # De-dup by lowercase name, keep max weight; sort deterministically
    out_map: Dict[str, float] = {}
    for it in filtered:
        lo = it["name"].lower()
        out_map[lo] = max(out_map.get(lo, 0.0), float(it["weight"]))
    out = [{"name": k, "weight": v} for k, v in out_map.items()]
    out.sort(key=lambda d: (-d["weight"], d["name"]))
    return out[:30]


def _canonicalize_skills_for_cache(skills: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
    """
    Deterministic canonical vector for cache keys:
    - lowercase trimmed names, numeric weights
    - sort by (weight desc, name asc)
    """
    vec = []
    for it in (skills or []):
        nm = (it.get("name") or "").strip().lower()
        if not nm:
            continue
        try:
            wt = float(it.get("weight", 0.0))
        except Exception:
            wt = 0.0
        if wt > 0:
            vec.append((nm, max(0.0, min(1.0, wt))))
    vec.sort(key=lambda t: (-t[1], t[0]))
    return vec[:30]


def _rank_cache_key(
    skills: List[Dict[str, Any]],
    mix_weights: Optional[Dict[str, float]],
    recency_months: int,
    data_version: int,
) -> str:
    """Stable SHA256 key of (canonical JD vector + mix + recency + data_version)."""
    canon = _canonicalize_skills_for_cache(skills)
    mix = {
        "skills": float((mix_weights or {}).get("skills", 0.85)),
        "experience": float((mix_weights or {}).get("experience", 0.15)),
    }
    payload = {
        "jd_canon": canon,
        "mix": {"skills": round(mix["skills"], 6), "experience": round(mix["experience"], 6)},
        "recency_months": int(recency_months),
        "data_version": int(data_version),
    }
    s = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class RecruitingCrew:
    """
    JD parsing (strict JSON via two-pass extractor) + retrieval + deterministic ranking
    (weighted skills + recent-relevant experience, no education), with rank caching.
    """
    def __init__(self, collection=None, openai_model: str = "gpt-4o"):
        self.model = os.getenv("DO_OPENAI_MODEL", openai_model)
        # LLM-availability flag so we can fail-soft to hints-only
        self.use_llm = bool(
            (os.getenv("DO_OPENAI_API_KEY") or "").strip()
            and (os.getenv("DO_OPENAI_API_BASE") or "").strip()
        )
        # kept for compatibility
        self.jd_agent = build_jd_agent(self.model)
        self.collection = collection

    def parse_jd(
        self,
        jd_text: str,
        ui_hints: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        # No JD text → hints only (if provided)
        if not (jd_text or "").strip():
            if ui_hints:
                prim = [s.strip() for s in (ui_hints.get("primary") or []) if s.strip()]
                sec  = [s.strip() for s in (ui_hints.get("secondary") or []) if s.strip()]
                excl = {s.strip().lower() for s in (ui_hints.get("exclude") or []) if s.strip()}
                skills = [{"name": s, "weight": 0.92} for s in prim if s.lower() not in excl] + \
                         [{"name": s, "weight": 0.70} for s in sec  if s.lower() not in excl]
                tmp = {}
                for it in skills:
                    k = it["name"].strip().lower()
                    tmp[k] = max(tmp.get(k, 0.0), it["weight"])
                out = [{"name": k, "weight": v} for k, v in tmp.items()]
                out.sort(key=lambda d: (-d["weight"], d["name"]))
                return {"skills": out[:30]}
            return {"skills": []}

        # If LLM creds are missing, fall back to hints-only behavior
        if not self.use_llm:
            return self.parse_jd("", ui_hints=ui_hints)

        # Two-pass strict JSON extractor (chunking + refine)
        try:
            data = jd_extract_skills(jd_text, max_items=30) or {"skills": []}
            skills = data.get("skills") or []
        except Exception:
            # fail-soft: hints-only if available, else empty
            return self.parse_jd("", ui_hints=ui_hints)

        # Clamp + enforce hints deterministically
        cleaned = []
        for it in skills:
            nm = (it.get("name") or "").strip()
            if not nm:
                continue
            try:
                wt = float(it.get("weight", 0.0))
            except Exception:
                wt = 0.0
            if wt > 0:
                cleaned.append({"name": nm, "weight": max(0.0, min(1.0, wt))})

        cleaned = _enforce_hints(cleaned, ui_hints)
        return {"skills": cleaned[:30]}

    # ---- Matching helpers ----
    def _rank_with_shortlist(
        self,
        jd_req: Dict[str, Any],
        *,
        top_k: int,
        mix_weights: Optional[Dict[str, float]],
        recency_months: int,
    ) -> List[Dict[str, Any]]:
        """
        Retrieval-first shortlist (Chroma) → load data → rank deterministically.
        Falls back to DB-only shortlist inside storage if Chroma is unavailable.
        """
        jd_skills = jd_req.get("skills") or []
        if not jd_skills:
            return []

        # Weighted skill → shortlist ids
        jd_weights = { (s.get("name") or "").strip().lower(): float(s.get("weight", 0.0)) for s in jd_skills if s.get("name") }
        shortlist_ids = shortlist_by_jd_skills(jd_weights, k=max(500, top_k))

        if not shortlist_ids:
            return []

        # Load only the shortlisted candidate data (profiles, skillmaps, skills_meta_map, resume_texts)
        rows_all, skillmaps, skills_meta_map, resume_texts = load_for_ranking(shortlist_ids)

        # Rank (pass skills_meta_map to avoid resume text scans when available)
        ranked = rank_candidates_weighted(
            jd_skills,
            rows_all,
            skillmaps,
            top_k=top_k,
            mix_weights=mix_weights,
            recency_months=recency_months,
            resume_texts=resume_texts,
            skills_meta_map=skills_meta_map,
        )
        return ranked

    def match(
        self,
        jd_req: Dict[str, Any],
        rows_all: List[Tuple[str, str, str, float]],
        all_skillmaps: Dict[str, Dict[str, float]],
        *,
        top_k: int = 10,
        mix_weights: Optional[Dict[str, float]] = None,
        recency_months: int = 24,  # ✅ unified to 24 months
        resume_texts: Optional[Dict[str, str]] = None,
        jd_text: str = "",
    ):
        """
        Back-compat entrypoint used by older callers/tests that pass preloaded rows/maps.
        Uses the modern ranker with optional resume_texts (for gating) but without retrieval.
        Prefer using .run() which performs retrieval + caching.
        """
        jd_skills = jd_req.get("skills") or []
        return rank_candidates_weighted(
            jd_skills, rows_all, all_skillmaps,
            top_k=top_k,
            mix_weights=mix_weights,
            jd_text=jd_text,
            recency_months=recency_months,
            resume_texts=resume_texts,
        )

    def run(
        self,
        jd_text: str,
        rows_all: List[Tuple[str, str, str, float]],
        *,
        top_k: int = 10,
        jd_req: Dict[str, Any] | None = None,
        all_skillmaps: Dict[str, Dict[str, float]] | None = None,
        mix_weights: Optional[Dict[str, float]] = None,
        recency_months: int = 24,  # ✅ unified to 24 months
        resume_texts: Optional[Dict[str, str]] = None,
        ui_hints: Optional[Dict[str, List[str]]] = None,
    ):
        """
        New path:
          1) Parse JD (or take provided jd_req)
          2) Compute rank cache key and return cached if fresh
          3) Retrieval-first shortlist via Chroma (fallback DB)
          4) Rank deterministically
          5) Write cache
        """
        # Parse JD (or accept a pre-parsed request)
        if jd_req is None:
            jd_req = self.parse_jd(jd_text or "", ui_hints=ui_hints)

        jd_skills = jd_req.get("skills") or []
        if not jd_skills:
            return jd_req, []

        # Cache lookup
        data_version = get_data_version()
        cache_key = _rank_cache_key(jd_skills, mix_weights, recency_months, data_version)
        cached = get_rank_cache(cache_key)
        if cached and int(cached.get("data_version", -1)) == data_version:
            payload = cached.get("payload") or []
            return jd_req, (payload[:top_k] if top_k else payload)

        # Retrieval-first shortlist + load and rank
        ranked = self._rank_with_shortlist(
            jd_req,
            top_k=top_k,
            mix_weights=mix_weights,
            recency_months=recency_months,
        )

        # Write cache
        try:
            set_rank_cache(cache_key, ranked, data_version)
        except Exception:
            pass

        return jd_req, ranked
