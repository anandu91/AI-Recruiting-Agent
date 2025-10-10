# app/utils/parsing.py
from __future__ import annotations

import io
import os
import re
import tempfile
from datetime import datetime
from typing import Optional, List, Tuple

from docx import Document
import fitz  # PyMuPDF

try:
    import mammoth
except Exception:
    mammoth = None

try:
    import docx2txt
except Exception:
    docx2txt = None


MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12
}

DEGREE_MAP: List[Tuple[str, str]] = [
    (r"\b(ph\.?d\.?|doctorate)\b", "phd"),
    (r"\b(masters?|m\.?sc\.?|m\.?tech|m\.?eng|m\.?e\.)\b", "master"),
    (r"\b(bachelors?|b\.?sc\.?|b\.?tech|b\.?e\.?|b\.?eng)\b", "bachelor"),
    (r"\b(associate|diploma)\b", "associate"),
    (r"\b(certificate|certification)\b", "certificate"),
]


# ----------------------------- DOC/DOCX/PDF text -----------------------------
def _read_docx_best_effort(file_bytes: bytes) -> str:
    """
    Tries mammoth -> python-docx -> docx2txt in that order. Best-effort, safe fallbacks.
    """
    if mammoth:
        try:
            result = mammoth.convert_to_markdown(io.BytesIO(file_bytes))
            return (result.value or "").strip()
        except Exception:
            pass
    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".docx") as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            doc = Document(tmp.name)
            return "\n".join(p.text for p in doc.paragraphs if p.text).strip()
    except Exception:
        pass
    if docx2txt:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(file_bytes)
                tmp.flush()
                tmp_path = tmp.name
            text = docx2txt.process(tmp_path) or ""
            return text.strip()
        except Exception:
            return ""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
    return ""


def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    """
    Extracts plain text from PDF/DOCX inputs. Returns empty string on failure.
    """
    name = (filename or "").lower()
    try:
        if name.endswith(".pdf"):
            text_chunks: List[str] = []
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for page in doc:
                    # Explicit "text" extraction mode for consistency
                    text_chunks.append(page.get_text("text") or "")
            return "\n".join(text_chunks).strip()
        elif name.endswith(".docx"):
            return _read_docx_best_effort(file_bytes)
        else:
            return ""
    except Exception:
        return ""


# ----------------------------- Heuristics: education -----------------------------
def guess_education(text: str) -> str:
    """
    Very light heuristic to classify the highest education level present in free text.
    Only used as a fallback; primary pipeline uses LLM strict-JSON extractor.
    """
    low = (text or "").lower()
    for pat, label in DEGREE_MAP:
        if re.search(pat, low):
            return label
    return "none"


# ----------------------------- Date parsing helpers -----------------------------
def _year(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    try:
        y = int(s)
        # allow up to next year for “expected” entries
        if 1980 <= y <= datetime.now().year + 1:
            return y
    except Exception:
        return None
    return None


def _month(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    s = str(s).lower().strip(". ,")
    if s.isdigit():
        n = int(s)
        return n if 1 <= n <= 12 else None
    return MONTHS.get(s[:3])


def _merge_ranges(ranges: List[Tuple[datetime, datetime]]) -> List[Tuple[datetime, datetime]]:
    """
    Merge overlapping or contiguous [start, end] month-bounded ranges.
    """
    if not ranges:
        return []
    ranges = sorted(ranges, key=lambda r: (r[0], r[1]))
    merged: List[Tuple[datetime, datetime]] = []
    cur_start, cur_end = ranges[0]
    for s, e in ranges[1:]:
        if s <= (cur_end):  # overlap or touch
            if e > cur_end:
                cur_end = e
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))
    return merged


# ----------------------------- Experience estimation -----------------------------
def estimate_total_experience_years(text: str) -> float:
    """
    Estimate total professional experience in years from free-text by detecting
    dateranges like 'Jan 2019 - Present' or '2016–2021'. Overlapping ranges are merged.
    If no ranges are found, falls back to span between min/max year tokens.

    Returns float rounded to two decimals.
    """
    text = text or ""
    now = datetime.now()

    # Examples matched:
    # Jan 2019 - Present
    # 2018 to 2022
    # Mar 2016 – Nov 2019
    date_pat = re.compile(
        r"(?:(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\s+)?(19[8-9]\d|20[0-4]\d)"
        r"\s*(?:-|–|—|to|till)\s*"
        r"(?:(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\s+)?(present|current|now|19[8-9]\d|20[0-4]\d)",
        re.I,
    )

    ranges: List[Tuple[datetime, datetime]] = []
    for m in date_pat.finditer(text):
        m1, y1s, m2, y2s = m.group(1), m.group(2), m.group(3), m.group(4)
        y1 = _year(y1s)
        if not y1:
            continue

        is_present = bool(re.match(r"present|current|now", (y2s or ""), re.I))
        if is_present:
            y2 = now.year
            mm2 = now.month
        else:
            y2 = _year(y2s)
            if not y2:
                continue
            mm2 = _month(m2) or 12

        mm1 = _month(m1) or 1

        try:
            d1 = datetime(y1, mm1, 1)
            d2 = datetime(y2, mm2, 1)
        except Exception:
            continue
        if d2 >= d1:
            ranges.append((d1, d2))

    # Merge overlapping/adjacent ranges before summation
    if ranges:
        merged = _merge_ranges(ranges)
        total_days = sum((e - s).days for s, e in merged)
        return round(total_days / 365.25, 2)

    # Fallback: min/max year window if at least two plausible years are present
    year_hits: List[int] = []
    for y in re.findall(r"(19[8-9]\d|20[0-4]\d)", text):
        try:
            yy = int(y)
        except Exception:
            continue
        if 1980 <= yy <= datetime.now().year + 1:
            year_hits.append(yy)

    if len(year_hits) >= 2:
        return round(float(max(year_hits) - min(year_hits)), 2)

    return 0.0
