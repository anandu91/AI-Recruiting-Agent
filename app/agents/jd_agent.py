# app/agents/jd_agent.py
from __future__ import annotations
import os, re, json
from textwrap import dedent
from typing import Dict, Any, List, Tuple
from crewai import Agent
from openai import OpenAI

# ========================== DO gateway helpers ==========================
def _normalize_do_base(base: str) -> str:
    base = (base or "").strip().rstrip("/")
    if not base:
        return base
    return base if base.endswith("/api/v1") else base + "/api/v1"

def _wire_do_env_for_crewai() -> None:
    do_key = (os.getenv("DO_OPENAI_API_KEY") or "").strip()
    do_base = _normalize_do_base(os.getenv("DO_OPENAI_API_BASE") or "")
    if do_key:
        os.environ["OPENAI_API_KEY"] = do_key
    if do_base:
        os.environ["OPENAI_API_BASE"] = do_base

def _get_client() -> Tuple[OpenAI, str]:
    key = (os.getenv("DO_OPENAI_API_KEY") or "").strip()
    base = _normalize_do_base(os.getenv("DO_OPENAI_API_BASE") or "")
    if not key or not base:
        raise RuntimeError("Missing DO_OPENAI_API_KEY / DO_OPENAI_API_BASE")
    # Use stronger model for cleaner extraction
    model = os.getenv("DO_OPENAI_MODEL", "gpt-4o")
    return OpenAI(api_key=key, base_url=base), model

# ============================== Prompts ================================
_SYSTEM_PROMPT = dedent("""
You are an expert recruiter. Return STRICT JSON only: {"skills":[{"name":"…","weight":0.00}]}.

Rules:
- Include ONLY concrete technologies (languages, libraries, frameworks, tools, cloud services, databases, vendor products, technical standards).
- EXCLUDE roles, duties, soft skills, data models, processes, companies, locations.
- Split combined items like “A/B”, “A or B”, “A & B” into separate items (same weight).
- Normalize names; strip versions (“Python 3.11”→“python”); lowercase is fine.
- Prefer specific AWS services (S3, Glue, EMR, Redshift, IAM) over umbrella “AWS”.
- If sections indicate priority (Required/Must-have vs Preferred/Nice-to-have), reflect it in weights (primary ≥0.90, secondary 0.60–0.89). Otherwise infer from emphasis/frequency.
- Return ≤30 unique items sorted by weight; weights in [0,1].
""").strip()

_REFINE_PROMPT = dedent("""
You are an expert recruiter. Clean and normalize the following STRICT JSON list and return STRICT JSON only: {"skills":[{"name":"…","weight":0.00}]}.

Do:
1) Split “A/B”, “A or B”, “A & B” into separate items (same weight).
2) Keep ONLY concrete technologies; remove umbrella “AWS” when specific AWS services exist (S3, Glue, EMR, Redshift, IAM).
3) Normalize names; strip versions; deduplicate; keep ≤30 items sorted by weight; weights in [0,1].
""").strip()

# ============================== Utils =================================
_JSON_RE = re.compile(r"\{[\s\S]*\}\s*$")

def _json_only(s: Any) -> Dict[str, Any]:
    if isinstance(s, dict):
        return s
    s = str(s or "")
    m = _JSON_RE.search(s)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        try:
            return json.loads(m.group(0).replace(", }", " }"))
        except Exception:
            return {}

def _dedup_and_clip(items: List[Dict[str, Any]], k: int = 30) -> List[Dict[str, Any]]:
    seen = {}
    for it in items or []:
        nm = (it.get("name") or "").strip()
        if not nm:
            continue
        wt = float(it.get("weight", 0.0))
        key = nm.lower()
        if key not in seen or wt > seen[key][1]:
            seen[key] = (nm, wt)
    sorted_items = sorted(seen.values(), key=lambda t: (-t[1], t[0]))
    return [{"name": n, "weight": max(0.0, min(1.0, float(w)))} for (n, w) in sorted_items][:k]

def _chunks(t: str, size: int = 9000, overlap: int = 700):
    t = t or ""
    if len(t) <= size:
        yield t; return
    step = max(1000, size - overlap)
    for i in range(0, len(t), step):
        yield t[i:i+size]

# --------- tiny post-filters (prompt-first, code-light) ----------
_SPLIT_SEPS = re.compile(r"\s*(?:/|,|;|\||\+|&|\bor\b|\band\b)\s*", re.I)
_AWS_SPECIFICS = {"s3","glue","emr","redshift","iam"}

def _split_combos_local(name: str) -> List[str]:
    parts = _SPLIT_SEPS.split(name or "")
    return [p.strip() for p in parts if p and p.strip()]

def _aws_umbrella_cleanup(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """If 2+ specific AWS services present, drop 'aws' to keep chips clean."""
    names = {it["name"].lower() for it in items if it.get("name")}
    if len(_AWS_SPECIFICS.intersection(names)) >= 2 and "aws" in names:
        return [it for it in items if it.get("name","").lower() != "aws"]
    return items

def _local_cleanup(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # 1) split combos the model may leave as "A/B"
    expanded: List[Dict[str, Any]] = []
    for it in items or []:
        nm = (it.get("name") or "").strip()
        if not nm:
            continue
        wt = float(it.get("weight", 0.0))
        pieces = _split_combos_local(nm)
        if len(pieces) <= 1:
            expanded.append({"name": nm, "weight": wt})
        else:
            for p in pieces:
                expanded.append({"name": p, "weight": wt})
    # 2) drop AWS umbrella if enough specifics are present
    expanded = _aws_umbrella_cleanup(expanded)
    # 3) dedup + clip
    return _dedup_and_clip(expanded, k=30)

# --------- type guard to avoid single-letter chips in UI ----------
def _coerce_skills_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure {"skills":[{"name":str,"weight":float}]} even if upstream returned
    a string or list[str]. This prevents the UI from iterating characters.
    """
    import re as _re
    items = (payload or {}).get("skills", [])
    out: List[Dict[str, Any]] = []

    if isinstance(items, str):
        for tok in _re.split(r"[,\|/;]+", items):
            name = tok.strip()
            if name:
                out.append({"name": name, "weight": 0.7})
    elif isinstance(items, list):
        if items and isinstance(items[0], str):
            for s in items:
                s = (s or "").strip()
                if s:
                    out.append({"name": s, "weight": 0.7})
        else:
            for it in items:
                if not isinstance(it, dict):
                    continue
                name = (it.get("name") or "").strip()
                if not name:
                    continue
                wt = float(it.get("weight", 0.0))
                out.append({"name": name, "weight": max(0.0, min(1.0, wt))})

    # dedup + clip
    seen = {}
    for it in out:
        k = it["name"].lower()
        w = it["weight"]
        if k not in seen or w > seen[k][1]:
            seen[k] = (it["name"], w)
    final = [{"name": n, "weight": w} for n, w in sorted(seen.values(), key=lambda t: (-t[1], t[0]))][:30]
    return {"skills": final}

# ============================ Extractor ================================
def jd_extract_skills(jd_text: str, max_items: int = 30) -> Dict[str, Any]:
    """
    Two-pass prompt JD parser (no heavy code lists):
      1) pass-1: extract concrete tech with weights (strict JSON)
      2) pass-2: refine/split/normalize (strict JSON)
      3) tiny local cleanup + type guard
    Returns: {"skills":[{"name":str,"weight":float}]}
    """
    client, model = _get_client()

    # pass-1: chunk & union with strict JSON
    all_items: List[Dict[str, Any]] = []
    for chunk in _chunks(jd_text, size=9000, overlap=700):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=800,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": chunk}
                ],
            )
            raw = resp.choices[0].message.content if resp.choices else ""
            data = _json_only(raw)
            items = data.get("skills") or []
            if isinstance(items, list):
                all_items.extend(items)
        except Exception:
            continue
        if len(all_items) >= max_items:
            break

    pre = _dedup_and_clip(all_items, k=max_items)

    # pass-2: refine (strict JSON)
    payload = json.dumps({"skills": pre}, separators=(",",":"))
    try:
        resp2 = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=600,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _REFINE_PROMPT},
                {"role": "user", "content": payload}
            ],
        )
        raw2 = resp2.choices[0].message.content if resp2.choices else ""
        data2 = _json_only(raw2)
        final_items = data2.get("skills") or pre
    except Exception:
        final_items = pre  # fail-soft

    # tiny local cleanup for umbrellas/combos
    cleaned = _local_cleanup(final_items)

    # type guard so UI always receives [{"name","weight"}...]
    return _coerce_skills_payload({"skills": cleaned})

# ============== Convenience API expected by pipeline ===================
def parse_jd_to_weighted_skills(jd_text: str) -> Dict[str, float]:
    """
    Pipeline-friendly API:
    Returns a canonical mapping {skill: weight} with skills lowercased,
    weights in [0,1], and at most 30 entries.
    """
    payload = jd_extract_skills(jd_text, max_items=30)
    items = payload.get("skills", []) if isinstance(payload, dict) else []
    out: Dict[str, float] = {}
    for it in items:
        name = (it.get("name") or "").strip().lower()
        if not name:
            continue
        try:
            w = float(it.get("weight", 0.0))
        except Exception:
            w = 0.0
        w = max(0.0, min(1.0, w))
        if w <= 0.0:
            continue
        # keep highest weight if duplicates
        if name not in out or w > out[name]:
            out[name] = w
    # clip to 30 by weight desc, name asc
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0]))[:30])

# ========================== CrewAI agent (compat) ======================
def build_jd_agent(model: str = "gpt-4o") -> Agent:
    _wire_do_env_for_crewai()
    return Agent(
        role="JD Parser",
        goal="Return prioritized, deduplicated technical skills with weights 0..1 as strict JSON.",
        backstory="Specialist at turning varied JDs into clean, weighted skill lists.",
        llm=model,
        verbose=False,
        allow_delegation=False,
        memory=False,
        system_prompt=_SYSTEM_PROMPT,
    )