# --- keep absolute imports like "from app.core.storage ..." working ---
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import re, logging, warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

from app.core.storage import (
    init_sqlite, get_chroma, upsert_candidate, fetch_candidate,
    fetch_all_candidates, fetch_all_candidate_skillmaps,
    add_note, mark_submission, set_stage, migrate_add_email,
    find_resume_id_by_email,
)
from app.core.parsing import extract_text_from_bytes, guess_education
from app.core.embeddings import embed_texts
from app.pipeline.crew_pipeline import RecruitingCrew
from app.agents.resume_agent import extract_resume_fields

# ranking helpers (ensure UI math perfectly matches engine)
from app.core.ranking import (
    normalize_weights, _canon_vocab_from, _canon_candidate_map,
    _canon_jd_list, rank_candidates_weighted, _base_token
)

# ---------------------------------------------------------------------------
# Fixed constants (no UI toggles)
# ---------------------------------------------------------------------------
RECENCY_MONTHS_FIXED = 24  # JD skills used within last 24 months  <<< changed

# ── Boot ──────────────────────────────────────────────────────────────────────
load_dotenv(find_dotenv(), override=True)
os.environ.setdefault("DATABASE_URL", "sqlite:///app.db")
st.set_page_config(page_title="Recruitment AI Agent", page_icon="🤖", layout="wide")
warnings.filterwarnings("ignore")
for n in ("chromadb","chromadb.telemetry","chromadb.api","chromadb.segment",
          "transformers","sentence_transformers","urllib3","httpx","httpcore"):
    logging.getLogger(n).setLevel(logging.WARNING)
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Theme (dark, compact)
st.markdown("""
<style>
html, body, .stApp, [data-testid="stAppViewContainer"], .main {
  background:#0b1220 !important; color:#e5e7eb !important;
}
[data-testid="stHeader"]{
  background:#0b1220 !important; border-bottom:1px solid #1f2937;
}
.block-container { padding-top: 2.2rem; padding-bottom: 1rem; }

/* metrics */
[data-testid="stMetricValue"]{ font-size:1.12rem !important; font-weight:800 !important; }
[data-testid="stMetricLabel"]{ font-size:.78rem !important; color:#9fb3c8 !important; }
div[data-testid="stMetricDelta"]{ font-size:.72rem !important; }

/* list buttons (global) */
.stButton > button{
  background:#111827; color:#cbd5e1; border:1px solid #334155;
  border-radius:999px; font-weight:600; padding:.14rem .46rem;
  margin: 2px 6px 0 0;
  line-height:1.0; font-size:.76rem;
}
.stButton > button:hover{ background:#0f172a; border-color:#3b4a63; }
.stButton.match > button{ background:#065f46; border-color:#0f8a67; color:#eafff4; }
.stButton.selected > button{ outline:2px solid #60a5fa; }

/* chips */
.chips { display:flex; flex-wrap:wrap; gap:6px; margin:.2rem 0 .1rem 0;}
.chip { background:#111827; border:1px solid #334155; color:#cbd5e1; font-size:.78rem; padding:.18rem .46rem; border-radius:999px; }
.chip.ok { background:#065f46; border-color:#0f8a67; color:#eafff4; }
.chip.miss { opacity:.78; }

.kv { color:#9ca3af; font-size:.84rem; margin:.22rem 0 .12rem 0; }
.hr { border-top:1px dashed #243142; margin:.5rem 0 .42rem 0; }
.smallnote { color:#93a3b8; font-size:.74rem; margin-top:.08rem; }

/* cards */
.hero-wrap { display:flex; align-items:center; justify-content:center; margin-top: 6px; margin-bottom: 14px; }
.hero { font-size: 1.9rem; font-weight: 800; letter-spacing: .2px; }
.cand-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }
.cand-card { background: #0f172a; border: 1px solid #1f2937; border-radius: 16px; padding: 14px 16px; box-shadow: 0 8px 24px rgba(0,0,0,.25); }
.cand-head { display:flex; align-items:center; justify-content:space-between; margin-bottom:6px; }
.cand-name { font-weight:800; font-size:1.05rem; color:#e5e7eb; }
.cand-sub { color:#9ca3af; font-size:.86rem; }

/* badges */
.badge { font-weight:800; font-size:.82rem; padding:.24rem .56rem; border-radius:999px; background:#0b5; color:#021; border:0; }
.badge.warn{ background:#f59e0b; color:#1f1300; }
.badge.danger{ background:#ef4444; color:#2a0a0a; }

/* === UI BAR LAYOUT TWEAK: start — thicker, full-width bars aligned === */
.boost-bar { height:10px;background:#0b2035;border:1px solid #1e293b;border-radius:6px;overflow:hidden; width:100%; }
.boost-fill { height:100%;background:#0ea5e9; }
.boost-outline { height:10px;background:transparent;border:2px dashed #334155;border-radius:6px; width:100%; }
/* right panel waterfall */
.wbar { height:10px;background:#1f2937;border-radius:6px;overflow:hidden; width:100%; }
.wseg { height:100%; }
.wseg-raw { background:#64748b; }
.wseg-main { background:#34d399; }
.wseg-rec { background:#60a5fa; }
.wseg-work { background:#f59e0b; }
.wseg-final { background:#0ea5e9; }
/* custom impact bar */
.impact-wrap{ height:10px;background:#1f2937;border-radius:6px;overflow:hidden; width:100%; }
.impact-fill{ height:100%; background:#0ea5e9; }

/* ===== Left-side contribution list tightening & nudge ===== */
.contrib-wrap { margin-left:-10px; }
.contrib-wrap .stButton > button { font-size:.72rem; padding:.12rem .40rem; }

/* ===== Small Donut Badges for left list ===== */
.donut-row { margin: 4px 0 8px; }
.donut-wrap{display:flex;flex-direction:column;align-items:center;gap:6px;}
.donut-sm{
  width:58px;height:58px;border-radius:50%;
  background:conic-gradient(#0ea5e9 calc(var(--p)*1%), #1f2937 0);
  display:grid;place-items:center;border:1px solid #223043;
  box-shadow:inset 0 0 0 3px #0b1220;border-radius:50%;
}
.donut-sm span{font-weight:800;font-size:.78rem;color:#e5e7eb}
.donut-label-sm{max-width:86px;text-align:center;font-size:.74rem;color:#9fb3c8;line-height:1.1}

/* compact table styling for expander */
.scoretbl { width:100%; border-collapse:collapse; }
.scoretbl th, .scoretbl td { padding:6px 8px; border-bottom:1px solid #223043; text-align:right; }
.scoretbl th:first-child, .scoretbl td:first-child { text-align:left; }
.scoretbl th { color:#9fb3c8; font-weight:700; }
</style>
""", unsafe_allow_html=True)

def hero():
    st.markdown('<div class="hero-wrap"><div class="hero">🤖 Recruitment AI Agent</div></div>', unsafe_allow_html=True)
def section(title: str):
    st.markdown(f"**{title}**")

# ── Init storage
init_sqlite()
migrate_add_email()
_client, _collection = get_chroma()

# ── Helpers
def _unique(seq: List[str]) -> List[str]:
    seen, out = set(), []
    for s in seq:
        k = s.strip()
        if k and k.lower() not in seen:
            seen.add(k.lower()); out.append(k)
    return out

def _safe_rid_from_filename(name: str) -> str:
    stem = Path(name or "").stem
    rid = re.sub(r"[^a-zA-Z0-9._-]+", "_", stem)[:80]
    return rid or "resume"

def _index_skill_names(resume_id: str, skills_map: Dict[str, float], email: str = ""):
    names = [k for k in (skills_map or {}).keys()]
    if not names: return
    try:
        _collection.delete(where={"resume_id": resume_id})
    except Exception:
        try:
            existing = _collection.get(where={"resume_id": resume_id})
            if existing and existing.get("ids"):
                _collection.delete(ids=existing["ids"])
        except Exception:
            pass
    embs = embed_texts(names)
    metas = [{"resume_id": resume_id, "skill": s, "email": email or ""} for s in names]
    ids = [f"{resume_id}::{i}" for i in range(len(names))]
    try:
        _collection.upsert(documents=names, embeddings=embs, metadatas=metas, ids=ids)
    except Exception:
        _collection.add(documents=names, embeddings=embs, metadatas=metas, ids=ids)

# DigitalOcean OpenAI helper for brief “Why this candidate?” text
def _normalize_do_base(base: str) -> str:
    base = (base or "").strip().rstrip("/")
    if not base: return base
    return base if base.endswith("/api/v1") else base + "/api/v1"
def _get_do_client():
    key = (os.getenv("DO_OPENAI_API_KEY") or "").strip()
    base = _normalize_do_base(os.getenv("DO_OPENAI_API_BASE") or "")
    if not key or not base:
        return None, ""
    model = os.getenv("DO_OPENAI_MODEL", "gpt-4o-mini")
    return OpenAI(api_key=key, base_url=base), model

@st.cache_data(show_spinner=False)
def explain_match_cached(
    resume_id: str,
    resume_text: str,
    jd_text: str,
    matched_csv: str,
    missing_csv: str,
    exp_years: float,
    model: str,
    have_do: bool,
) -> str:
    if not have_do:
        matched = matched_csv.split(", ") if matched_csv else []
        missing = missing_csv.split(", ") if missing_csv else []
        head = ", ".join(matched[:5]) or "-"
        tail = ", ".join(missing[:3]) or "-"
        return f"{exp_years:.1f} years' experience. Strong evidence of: {head}. Gaps to validate: {tail if tail else '-'}."
    client, _ = _get_do_client()
    if not client:
        matched = matched_csv.split(", ") if matched_csv else []
        missing = missing_csv.split(", ") if missing_csv else []
        head = ", ".join(matched[:5]) or "-"
        tail = ", ".join(missing[:3]) or "-"
        return f"{exp_years:.1f} years' experience. Strong evidence of: {head}. Gaps to validate: {tail if tail else '-'}."
    messages = [
        {"role": "system", "content": "You are a brief, neutral recruiter assistant. Write 2 short sentences "
                                      "explaining why a candidate is a good fit for the given JD. Mention years of "
                                      "experience and 3–5 relevant skills. If there are gaps, mention them politely. "
                                      "Plain text only, ~60–100 words."},
        {"role": "user", "content": f"JOB DESCRIPTION:\n{jd_text[:6000]}\n\n"
                                    f"CANDIDATE RESUME:\n{resume_text[:12000]}\n\n"
                                    f"Matched skills to highlight: {matched_csv or '-'}\n"
                                    f"Possible gaps: {missing_csv or '-'}\n"
                                    f"Years of experience: {exp_years:.1f}\n"
                                    "Write the explanation now."}
    ]
    try:
        resp = client.chat.completions.create(
            model=os.getenv("DO_OPENAI_MODEL", "gpt-4o-mini"),
            messages=messages,
            temperature=0.2,
            max_tokens=180,
        )
        txt = (resp.choices[0].message.content or "").strip()
        return txt or "Explanation unavailable."
    except Exception:
        return "Explanation unavailable."

# --- Job role helpers (unchanged)
_ROLE_RE = re.compile(r"(?i)\b(?:job\s*title|title|position|role)\s*[:\-]\s*([^\n,|/]+)")
_ROLE_FALLBACK = re.compile(
    r"(?i)\b(?:(sr\.|sr|senior|lead|principal|staff|jr\.|jr)\s+)?"
    r"(?:data|software|machine\s*learning|ml|devops|cloud|full[- ]stack|backend|front[- ]end|qa|test|sdet|web|android|ios|security|platform|site reliability|sre|analytics?)\s+"
    r"(?:engineer|developer|scientist|analyst|manager|architect|consultant|administrator|specialist)\b"
)
def _titlecase(s: str) -> str:
    if not s: return s
    s = re.sub(r"\s{2,}", " ", s.strip())
    out = " ".join(
        w.capitalize() if not re.match(r"(?i)^(ai|ml|qa|sre|sql|aws|gcp|ios)$", w) else w.upper()
        for w in re.split(r"\s+", s)
    )
    out = re.sub(r"(?i)\bSr\.?\b", "Senior", out)
    out = re.sub(r"(?i)\bJr\.?\b", "Junior", out)
    return out
def infer_job_role(jd_text: str) -> str:
    if not jd_text: return "the role"
    m = _ROLE_RE.search(jd_text)
    if m: return _titlecase(m.group(1))
    m2 = _ROLE_FALLBACK.search(jd_text)
    if m2: return _titlecase(m2.group(0))
    first = next((ln.strip() for ln in jd_text.splitlines() if 2 <= len(ln.split()) <= 8), "")
    return _titlecase(first) if first else "the role"

# ---------- UI chip planning ----------
def _jd_display_and_canon(jd_req: Dict[str, Any], all_skillmaps: Dict[str, Dict[str, float]]):
    jd_skills = (jd_req or {}).get("skills", [])
    _, base_to_canon = _canon_vocab_from(jd_skills, all_skillmaps)
    best = {}
    for it in jd_skills:
        nm = (it.get("name") or "").strip()
        if not nm: continue
        c = base_to_canon.get(_base_token(nm), _base_token(nm))
        wt = float(it.get("weight", 0.0))
        prev = best.get(c)
        if (prev is None) or (wt > prev[0]) or (wt == prev[0] and len(nm) < len(prev[1])):
            best[c] = (wt, nm)
    items = [{"display": v[1], "canon": c, "weight": v[0]} for c, v in best.items()]
    items.sort(key=lambda d: (-d["weight"], d["display"]))
    return items, set(x["canon"] for x in items), base_to_canon

def _score_badge_class(pct: int) -> str:
    if pct >= 80: return "badge"
    if pct >= 60: return "badge warn"
    return "badge danger"

# === EXP DISPLAY FIX: start ===
_EXP_YEARS_RE = re.compile(r"\b(\d{1,2})\s*\+?\s*(?:years?|yrs?)\b", re.I)
def _estimate_years_from_text(t: str) -> float:
    years = 0.0
    for m in _EXP_YEARS_RE.finditer(t or ""):
        try: years = max(years, float(m.group(1)))
        except Exception: pass
    return years
def _display_experience_years(stored: float, resume_text: str) -> float:
    try: s = float(stored or 0.0)
    except Exception: s = 0.0
    if s >= 2.0: return s
    est = _estimate_years_from_text(resume_text or "")
    return max(s, est)
# === EXP DISPLAY FIX: end ===

# === LABELS: start — microcopy for columns/metrics ===
LABEL_JD_PRIORITY = "JD priority"
LABEL_BASE = "Base score"
LABEL_MAIN = "Top-skill bonus"
LABEL_REC = "Recent-use bonus"
LABEL_WORK = "On-the-job bonus"
LABEL_FINAL = "Boosted score"
LABEL_IMPACT = "Impact (0..5)"
# === LABELS: end ===

# === NAME FORMAT: start — pretty display names for JD contribution list ===
_ACRONYMS = {
    "ai","ml","api","qa","sre","etl","elt","ci","cd","sql","dbt","aws","gcp","gke",
    "nlp","rest","sdk","api","ui","ux","db2"
}
def _pretty_skill(s: str) -> str:
    """
    UI-only formatter:
    - Full-uppercase known acronyms: SQL, DBT, DB2, etc.
    - 'apache' + word => 'Apache' + Word with NO space (ApacheBeam, ApacheHudi).
    - otherwise Title Case per token.
    """
    raw = (s or "").strip()
    base = _base_token(raw)
    low = base.lower()

    if low in _ACRONYMS:
        return low.upper()

    if low.startswith("apache") and len(low) > 6:
        rest = low[6:]
        if rest:
            return "Apache" + rest.capitalize()

    SPECIAL = {
        "bigquery": "BigQuery",
        "kubernetes": "Kubernetes",
        "airflow": "Airflow",
        "terraform": "Terraform",
        "docker": "Docker",
        "cloudcomposer": "Cloud Composer",
        "springboot": "Spring Boot",
        "dataproc": "Dataproc",
    }
    if low in SPECIAL:
        return SPECIAL[low]

    parts = re.split(r"[^a-z0-9]+", low)
    parts = [p for p in parts if p]
    if not parts:
        return raw
    return " ".join(p.upper() if p in _ACRONYMS else p.capitalize() for p in parts)
# === NAME FORMAT: end ===

# === Left donuts (unchanged from your last version) ===
def render_contrib_donuts_small(r: Dict[str, Any], sel_key: str) -> None:
    rows = sorted(
        r.get("per_skill_rows") or [],
        key=lambda x: (-float(x.get("contribution_0to5", 0.0)), x.get("skill", ""))
    )[:5]
    if not rows:
        st.info("No per-skill rows."); return

    st.caption("JD contributions (top 5)")
    cols = st.columns(5, gap="small")
    for i, row in enumerate(rows):
        with cols[i]:
            skill = row.get("skill","")
            label = _pretty_skill(skill)
            contrib = float(row.get("contribution_0to5", 0.0))
            jd_w = float(row.get("jd_weight", 0.0))
            pct = max(0.0, min(1.0, contrib / 5.0)) * 100.0

            if st.button(label, key=f"dsel_{r.get('resume_id','')}_{skill}"):
                st.session_state[sel_key] = skill

            st.markdown(
                f"""
                <div class="donut-wrap">
                  <div class="donut-sm" style="--p:{pct:.0f}">
                    <span>{pct:.0f}%</span>
                  </div>
                  <div class="donut-label-sm">wt {jd_w:.3f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

# === Right-side detail (updated: expander shows OTHER 4 skills, no bars) ===
def _render_skill_detail(selected_row: Dict[str, Any], per_rows: List[Dict[str, Any]], selected_skill: str) -> None:
    jd_w  = float(selected_row.get("jd_weight", 0.0))
    raw   = float(selected_row.get("raw_strength", 0.0))
    main  = 2.0 if bool(selected_row.get("is_main", False)) else 0.0
    rec   = float(selected_row.get("recency_boost", 0.0))
    work  = float(selected_row.get("work_boost", 0.0))
    final = float(selected_row.get("final_strength", 0.0))
    contrib = float(selected_row.get("contribution_0to5", 0.0))

    st.markdown(f"### {_pretty_skill(selected_row.get('skill',''))}")
    a,b,c,d,e,f = st.columns(6)
    with a: st.metric(LABEL_JD_PRIORITY, f"{jd_w:.3f}")
    with b: st.metric(LABEL_BASE, f"{raw:.2f}")
    with c: st.metric(LABEL_MAIN, f"{main:.1f}")
    with d: st.metric(LABEL_REC, f"{rec:.1f}")
    with e: st.metric(LABEL_WORK, f"{work:.1f}")
    with f: st.metric(LABEL_FINAL, f"{final:.2f}")

    st.markdown(
        f"<div style='color:#9fb3c8;margin:.1rem 0 .25rem 0;'>{LABEL_IMPACT}: {contrib:.2f}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='impact-wrap'><div class='impact-fill' style='width:{min(100, max(0, contrib/5*100)):.0f}%'></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # EXPANDER: show the OTHER top skills (exclude the currently selected one)
    others = [r for r in per_rows if r.get("skill") != selected_skill]
    # keep only top 4 others based on contribution
    others = sorted(others, key=lambda x: (-float(x.get("contribution_0to5",0.0)), x.get("skill","")))[:4]

    with st.expander("Other top skills — details", expanded=False):
        if not others:
            st.caption("No additional skills to compare.")
        else:
            # Build compact table (no bars)
            header = """
            <table class="scoretbl">
              <thead>
                <tr>
                  <th>Skill</th>
                  <th>JD priority</th>
                  <th>Base</th>
                  <th>Top-skill</th>
                  <th>Recent-use</th>
                  <th>On-the-job</th>
                  <th>Boosted</th>
                </tr>
              </thead>
              <tbody>
            """
            rows_html = []
            for r in others:
                rows_html.append(
                    f"<tr>"
                    f"<td>{_pretty_skill(r.get('skill',''))}</td>"
                    f"<td>{float(r.get('jd_weight',0.0)):.3f}</td>"
                    f"<td>{float(r.get('raw_strength',0.0)):.2f}</td>"
                    f"<td>{(2.0 if bool(r.get('is_main', False)) else 0.0):.1f}</td>"
                    f"<td>{float(r.get('recency_boost',0.0)):.1f}</td>"
                    f"<td>{float(r.get('work_boost',0.0)):.1f}</td>"
                    f"<td>{float(r.get('final_strength',0.0)):.2f}</td>"
                    f"</tr>"
                )
            footer = "</tbody></table>"
            st.markdown(header + "\n".join(rows_html) + footer, unsafe_allow_html=True)

def render_candidate_cards(
    ranked: List[Dict[str, Any]],
    email_map: Dict[str,str],
    top_n: int,
    jd_text: str,
    jd_req: Dict[str, Any] | None,
    all_skillmaps: Dict[str, Dict[str, float]] | None,
):
    st.markdown('<div class="cand-grid">', unsafe_allow_html=True)

    have_do = bool((os.getenv("DO_OPENAI_API_KEY") or "").strip() and (_normalize_do_base(os.getenv("DO_OPENAI_API_BASE") or "")))
    model = os.getenv("DO_OPENAI_MODEL", "gpt-4o-mini")

    jd_items: List[Dict[str, Any]] = []
    jd_canon_set = set()
    base_to_canon = {}
    if jd_req and all_skillmaps:
        jd_items, jd_canon_set, base_to_canon = _jd_display_and_canon(jd_req, all_skillmaps)

    safe_count = max(1, int(top_n))
    for idx in range(min(safe_count, len(ranked))):
        r = ranked[idx] or {}
        score10 = float(r.get("score_norm_10", 0.0))
        pct = int(round(score10 * 10))  # 0..100
        badge_cls = _score_badge_class(pct)
        email = email_map.get(r.get("resume_id",""), "")

        cand = fetch_candidate(r.get("resume_id","")) or {}
        full_text = cand.get("raw_text", "") or ""
        exp_stored = float(r.get("experience", 0.0))
        exp_display = _display_experience_years(exp_stored, full_text)

        matched_list = r.get("matched_skills", []) or []  # canonical names
        missing_list = r.get("missing_skills", []) or []
        matched_csv = ", ".join(matched_list) if matched_list else ""
        missing_csv = ", ".join(missing_list) if missing_list else ""

        reason = explain_match_cached(
            resume_id=r.get("resume_id",""),
            resume_text=full_text,
            jd_text=jd_text or "",
            matched_csv=matched_csv,
            missing_csv=missing_csv,
            exp_years=exp_display,
            model=model,
            have_do=have_do,
        )

        st.markdown(f"""
        <div class="cand-card">
          <div class="cand-head">
            <div>
              <div class="cand-name">#{idx+1} &nbsp; {r.get('name','')}</div>
              <div class="cand-sub">{email or '&nbsp;'}</div>
            </div>
            <div><span class="{badge_cls}">{pct}% Match</span></div>
          </div>
        """, unsafe_allow_html=True)

        # --- Skills from the JD (chips) ---
        if jd_items:
            st.markdown('<div class="kv"><b>Skills from the JD</b></div>', unsafe_allow_html=True)
            chips_html = []
            matched_set = set(matched_list)
            for it in jd_items:
                cls = "chip ok" if it["canon"] in matched_set else "chip miss"
                chips_html.append(f'<span class="{cls}">{it["display"]}</span>')
            st.markdown(f'<div class="chips">{"".join(chips_html)}</div>', unsafe_allow_html=True)

        # --- Skills from the Resume (all) ---
        if all_skillmaps:
            cmap_raw = all_skillmaps.get(r.get("resume_id",""), {}) or {}
            skills_sorted = sorted(
                ((k, float(v or 0.0)) for k,v in cmap_raw.items()),
                key=lambda kv: (-kv[1], kv[0])
            )
            if skills_sorted:
                st.markdown('<div class="kv"><b>Skills from the Resume</b></div>', unsafe_allow_html=True)
                chips = []
                for k,_ in skills_sorted:
                    canon_k = base_to_canon.get(_base_token(k), _base_token(k))
                    cls = "chip ok" if canon_k in jd_canon_set else "chip"
                    chips.append(f'<span class="{cls}">{k}</span>')
                st.markdown(f'<div class="chips">{"".join(chips)}</div>', unsafe_allow_html=True)

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.markdown("**Why this candidate?**")
        st.markdown(reason)

        # Per-skill: left list (top-5) + right detail
        per_rows = r.get("per_skill_rows") or []
        if jd_items and per_rows:
            rid = r.get("resume_id","")
            sel_key = f"sel_skill_{rid}"
            if sel_key not in st.session_state:
                top_skill = max(per_rows, key=lambda x: (float(x.get("contribution_0to5", 0.0)), x.get("skill","")))["skill"]
                st.session_state[sel_key] = top_skill

            left, right = st.columns([0.54, 0.46])
            with left:
                render_contrib_donuts_small(r, sel_key)
            with right:
                picked = next((x for x in per_rows if x.get("skill")==st.session_state.get(sel_key)), None)
                if picked:
                    _render_skill_detail(picked, per_rows, st.session_state.get(sel_key))

        # summary line (updated wording)
        st.markdown(
            f'<div class="smallnote">Experience: {exp_display:.1f} yrs &nbsp; • &nbsp; '
            f'JD skills used ≤{RECENCY_MONTHS_FIXED} months: {"Yes" if r.get("recent_relevant", False) else "No"}</div>',
            unsafe_allow_html=True
        )

        with st.expander("Full resume text"):
            st.text_area(
                label="",
                value=full_text, height=420, label_visibility="collapsed", disabled=True,
                key=f"rt_{r.get('resume_id','')}"
            )

        st.markdown("</div>", unsafe_allow_html=True)  # .cand-card

    st.markdown("</div>", unsafe_allow_html=True)  # .cand-grid

# ── UI ────────────────────────────────────────────────────────────────────────
hero()

# 1) Upload / Ingest
st.markdown("**1) Upload / Ingest Resumes**")
uploaded_files = st.file_uploader(
    "Upload PDF or DOCX resumes",
    accept_multiple_files=True,
    type=["pdf","docx"],
    help="Resumes are parsed once to extract name/email/experience/skills."
)
if uploaded_files:
    with st.status("Bulk Ingestion", expanded=True) as status:
        enriched: List[Dict[str, Any]] = []
        status.write("Parsing files and calling LLM…")
        for f in uploaded_files:
            raw = f.read()
            text = extract_text_from_bytes(raw, f.name)
            if not text.strip():
                st.warning(f"Could not parse {f.name}"); continue
            fields = extract_resume_fields(text)
            email = (fields.get("email") or "").strip()
            existing_rid = find_resume_id_by_email(email) if email else None

            resume_id = existing_rid or _safe_rid_from_filename(f.name)
            full_name = (fields.get("full_name") or resume_id).strip()[:80]
            skills_map = fields.get("skills_map") or {}
            edu_level = fields.get("education_level") or guess_education(text)
            try:
                exp_years = float(fields.get("experience_years") or 0.0)
            except Exception:
                exp_years = 0.0

            enriched.append({
                "resume_id": resume_id,
                "name": full_name,
                "email": email,
                "text": text,
                "skills_map": skills_map,
                "education": edu_level,
                "experience": exp_years
            })

        status.write("Persisting to DB & indexing skill names…")
        for e in enriched:
            upsert_candidate(
                rid=e["resume_id"], name=e["name"], email=e["email"],
                education=e["education"], experience=e["experience"],
                raw_text=e["text"], skills=e["skills_map"],
            )
            _index_skill_names(e["resume_id"], e["skills_map"], e["email"])

        st.session_state["ingest_counter"] = st.session_state.get("ingest_counter", 0) + 1
        status.update(label=f"Processed {len(enriched)} of {len(uploaded_files)} file(s)", state="complete")

st.divider()

# 2) Job Description
st.markdown("**2) Job Description**")
jd_text = st.text_area("Paste the job description…", height=220, placeholder="Paste the JD…")

# 3) Top-N and Rank
st.markdown("**3) Ranked Candidates**")
top_k = st.number_input("Top-N best candidates", min_value=1, max_value=100, value=10, step=1)

def _dedup_by_email_preserve_order(ranked_rows: List[Dict[str, Any]], desired_k: int) -> List[Dict[str, Any]]:
    email_map: Dict[str, str] = {}
    for r in ranked_rows:
        rid = r.get("resume_id", "")
        if not rid: continue
        c = fetch_candidate(rid) or {}
        email_map[rid] = (c.get("email","") or "").strip().lower()
    out, seen = [], set()
    for r in ranked_rows:
        rid = r.get("resume_id", "")
        key = email_map.get(rid) or f"RID::{rid}"
        if key in seen: continue
        seen.add(key)
        out.append(r)
        if len(out) >= desired_k:
            break
    return out

if st.button("Rank / Re-Rank Candidates"):
    with st.status("Ranking Resumes", expanded=True) as stat:
        stat.write("Collecting candidates…")
        rows_all: List[Tuple[str, str, str, float]] = fetch_all_candidates()

        stat.update(label="Loading skill maps…")
        all_maps_raw = fetch_all_candidate_skillmaps()

        stat.update(label="Loading resume texts…")
        resume_texts: Dict[str, str] = {}
        for rid, *_ in rows_all:
            c = fetch_candidate(rid) or {}
            resume_texts[rid] = (c.get("raw_text") or "")

        stat.update(label="Parsing JD…")
        crew = RecruitingCrew(_collection, openai_model=os.getenv("OPENAI_MODEL","gpt-4o-mini"))
        jd_req_parsed = crew.parse_jd(jd_text or "") or {"skills": []}
        if not jd_req_parsed.get("skills"):
            tokens = [t.strip() for t in re.split(r"[,\n/;|]+", jd_text or "") if t.strip()]
            tokens = _unique(tokens)[:12]
            jd_req_parsed = {"skills": [{"name": t, "weight": 1.0} for t in tokens]}

        stat.update(label="Ranking…")
        raw_ranked = rank_candidates_weighted(
            jd_skills=jd_req_parsed.get("skills") or [],
            rows_all=rows_all,
            all_skillmaps=all_maps_raw,
            top_k=min(len(rows_all), int(top_k) * 3),  # buffer for email dedup
            mix_weights={"skills": 0.85, "experience": 0.15},
            jd_text=jd_text,
            recency_months=RECENCY_MONTHS_FIXED,
            resume_texts=resume_texts,
        )
        ranked = _dedup_by_email_preserve_order(raw_ranked, int(top_k))

        st.session_state["ranked"] = ranked
        st.session_state["jd_req"] = jd_req_parsed
        st.session_state["jd_text"] = jd_text or ""
        st.session_state["all_maps"] = all_maps_raw

        stat.update(label="Ranking complete", state="complete")

ranked = st.session_state.get("ranked", [])
if ranked:
    role_for_heading = infer_job_role(st.session_state.get("jd_text",""))
    st.markdown(f"### Top {len(ranked)} best candidates for {role_for_heading}")

    # Map resume_id -> email for quick lookup
    email_map: Dict[str, str] = {}
    for r in ranked:
        rid = r.get("resume_id", "")
        if not rid: continue
        c = fetch_candidate(rid) or {}
        email_map[rid] = c.get("email","")

    view_mode = st.radio("View as", ["Cards", "Table"], horizontal=True, index=0)

    if view_mode == "Cards":
        render_candidate_cards(
            ranked, email_map, top_n=len(ranked),
            jd_text=st.session_state.get("jd_text","") or "",
            jd_req=st.session_state.get("jd_req", {"skills": []}),
            all_skillmaps=st.session_state.get("all_maps", {}),
        )
    else:
        df = pd.DataFrame([{
            "Resume_Id": r.get("resume_id",""),
            "Name": r.get("name",""),
            "Email": email_map.get(r.get("resume_id",""), ""),
            "Score (0-10)": r.get("score_norm_10", 0.0),
            "Skills (0-10)": r.get("score_skills_10", 0.0),
            "Experience (0-10)": r.get("score_exp_10", 0.0),
            "Relevant ≤24m": "Yes" if r.get("recent_relevant", False) else "No",
            "Experience (Years)": r.get("experience", 0.0),
            "Matched Skills": ", ".join(r.get("matched_skills", []) or []) or "-",
            "Missing Skills": ", ".join(r.get("missing_skills", []) or []) or "-",
        } for r in ranked])
        st.dataframe(df, use_container_width=True, hide_index=True)

# Candidate actions
st.markdown("##### Candidate actions")
selected_resume_id = st.selectbox("Pick a candidate", [r.get("resume_id","") for r in st.session_state.get("ranked", []) if r.get("resume_id")])
job_id = "JOB-DEFAULT"

if selected_resume_id:
    cand = fetch_candidate(selected_resume_id)
    if cand:
        exp_display_hdr = _display_experience_years(cand.get("experience", 0.0), cand.get("raw_text","") or "")
        st.markdown(f"**{cand.get('name','')}** | {cand.get('email','')} | Edu: {cand.get('education','')} | Exp: {exp_display_hdr:.1f} years")
        with st.expander("Full resume text"):
            st.text_area("", value=cand.get("raw_text","") or "", height=500, label_visibility="collapsed", disabled=True, key=f"cand_rt_{selected_resume_id}")

        note = st.text_area("Internal note (saved to DB)")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Save Note"):
                add_note(job_id, selected_resume_id, note or ""); st.success("Note saved.")
        with c2:
            if st.button("Internal Submit"):
                mark_submission(job_id, selected_resume_id, "internal_submit")
                set_stage(job_id, selected_resume_id, "internal_submit", {}); st.success("Marked as Internal Submission.")
        with c3:
            if st.button("Internal Reject"):
                mark_submission(job_id, selected_resume_id, "internal_reject")
                set_stage(job_id, selected_resume_id, "internal_reject", {"reason": note or ""}); st.success("Marked as Internal Reject.")

        st.markdown("**Upload revised resume for this candidate (re-index only this candidate)**")
        upd = st.file_uploader("Upload updated PDF/DOCX", type=["pdf","docx"], key=f"upd_{selected_resume_id}")
        if upd:
            raw = upd.read()
            new_text = extract_text_from_bytes(raw, upd.name)
            if not new_text.strip():
                st.warning("Could not parse the updated file.")
            else:
                fields2 = extract_resume_fields(new_text)
                new_name = (fields2.get("full_name") or cand.get("name", selected_resume_id)).strip()[:80]
                new_email = (fields2.get("email") or cand.get("email","")).strip()
                new_skills = fields2.get("skills_map") or {}
                new_edu = (fields2.get("education_level") or guess_education(new_text))
                try:
                    new_exp = float(fields2.get("experience_years") or 0.0)
                except Exception:
                    new_exp = cand.get("experience", 0.0)

                upsert_candidate(
                    rid=selected_resume_id, name=new_name, email=new_email,
                    education=new_edu, experience=new_exp, raw_text=new_text, skills=new_skills,
                )
                _index_skill_names(selected_resume_id, new_skills, new_email)
                st.success("Updated resume ingested and re-indexed.")
