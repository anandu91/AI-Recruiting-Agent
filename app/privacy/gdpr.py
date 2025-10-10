# app/privacy/gdpr.py
from __future__ import annotations
import os
import re
import json
import hashlib
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

# === ENV SWITCHES (no code changes needed) =========================
GDPR_MODE = (os.getenv("GDPR_MODE", "true").lower() in {"1","true","yes"})
REDACT_PHONE = True
REDACT_EMAIL = True
REDACT_NAME  = True

# retention (days) – your app can enforce this elsewhere if you want
RETENTION_DAYS = int(os.getenv("DATA_RETENTION_DAYS", "365"))

# ==================================================================

_EMAIL_RE = re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)")
# very simple “name line” heuristic (first/last etc.)
_NAME_LINE_RE = re.compile(r"(?m)^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*$")

@dataclass
class LocalPII:
    name: str = ""
    email: str = ""
    phone: str = ""

def _first(regex: re.Pattern, text: str) -> str:
    m = regex.search(text or "")
    return m.group(0).strip() if m else ""

def extract_local_pii(text: str) -> LocalPII:
    email = _first(_EMAIL_RE, text)
    phone = _first(_PHONE_RE, text)
    name = ""
    for line in (text or "").splitlines()[:25]:
        l = line.strip()
        if _NAME_LINE_RE.match(l):
            name = l
            break
    return LocalPII(name=name, email=email, phone=phone)

def _token(label: str, value: str) -> str:
    # stable token (doesn't leak raw value)
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"__{label.upper()}_{digest}__"

def sanitize_text_for_llm(text: str, pii: Optional[LocalPII]=None) -> Tuple[str, Dict[str,str]]:
    """
    Returns (safe_text, token_map). Replaces found PII with deterministic tokens.
    token_map maps token -> original value (server-side only; never send it out).
    """
    if not GDPR_MODE:
        return text or "", {}

    original = text or ""
    token_map: Dict[str,str] = {}
    pii = pii or extract_local_pii(original)
    safe = original

    # replace longest first to minimize partial overlaps
    replacements = []

    if REDACT_EMAIL and pii.email:
        tok = _token("EMAIL", pii.email)
        replacements.append((pii.email, tok))

    if REDACT_PHONE and pii.phone:
        tok = _token("PHONE", pii.phone)
        replacements.append((pii.phone, tok))

    if REDACT_NAME and pii.name:
        tok = _token("NAME", pii.name)
        replacements.append((pii.name, tok))

    # sort by length desc so we don’t double-replace substrings
    for val, tok in sorted(replacements, key=lambda x: -len(x[0])):
        safe = safe.replace(val, tok)
        token_map[tok] = val

    return safe, token_map

def rehydrate_tokens(text: str, token_map: Dict[str,str]) -> str:
    """
    Put PII back into a text blob for **internal** display if allowed.
    """
    out = text or ""
    for tok, real in token_map.items():
        out = out.replace(tok, real)
    return out

# --- subject rights helpers (data portability / erasure) ------------------

def export_subject_record(record: dict) -> str:
    """
    Build a JSON export of everything your app stores for a candidate.
    Call this when user requests 'Right of Access / Portability'.
    """
    # Ensure we never export LLM prompts/responses containing PII (we don’t store those here)
    return json.dumps(record, ensure_ascii=False, indent=2)

def minimal_log(event: str, subject_id: str, meta: Optional[dict]=None) -> None:
    """
    Very small audit log stub; replace with DB/central logging if you want.
    """
    if os.getenv("GDPR_AUDIT_LOG", "stdout") == "stdout":
        print(json.dumps({"event": event, "subject_id": subject_id, "meta": meta or {}}, ensure_ascii=False))
