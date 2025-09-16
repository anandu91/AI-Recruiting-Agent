# app/pipeline/crew_pipeline.py
import os
from typing import Dict, Any, List, Tuple, Optional

from app.agents.jd_agent import jd_extract_skills, build_jd_agent
from app.core.storage import fetch_candidate, fetch_all_candidate_skillmaps
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


class RecruitingCrew:
    """
    JD parsing (strict JSON via two-pass extractor) + deterministic ranking
    (weighted skills + recent-relevant experience, no education).
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
        # Parse JD (or accept a pre-parsed request)
        if jd_req is None:
            jd_req = self.parse_jd(jd_text or "", ui_hints=ui_hints)

        # Skill maps fallback
        if all_skillmaps is None:
            all_skillmaps = fetch_all_candidate_skillmaps()

        # Resume texts fallback (for recent+relevant experience gating)
        if resume_texts is None:
            resume_texts = {}
            for rid, *_ in rows_all:
                c = fetch_candidate(rid) or {}
                resume_texts[rid] = c.get("raw_text", "") or ""

        ranked = self.match(
            jd_req, rows_all, all_skillmaps,
            top_k=top_k,
            mix_weights=mix_weights,
            recency_months=recency_months,
            resume_texts=resume_texts,
            jd_text=jd_text or "",
        )
        return jd_req, ranked
