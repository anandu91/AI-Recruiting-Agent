# app/utils/parsing.py
from __future__ import annotations

import io
import os
import re
import tempfile
import unicodedata
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

# -------- Normalization & limits --------
_MAX_PAGES = int(os.getenv("PARSE_MAX_PDF_PAGES", "200"))

def _u(s: str) -> str:
    """Normalize Unicode, normalize dashes, collapse funky whitespace."""
    s = unicodedata.normalize("NFKC", s or "")
    s = (s
         .replace("\u2013", "-")
         .replace("\u2014", "-")
         .replace("\u2212", "-")
         .replace("\u00A0", " "))
    # collapse horizontal whitespace (preserve newlines)
    s = re.sub(r"[ \t\f\v]+", " ", s)
    # strip trailing spaces on lines
    s = re.sub(r"[ \t]+(\r?\n)", r"\1", s)
    return s

def _strip_repeating_lines(text: str, threshold: int = 3) -> str:
    """
    Remove lines that repeat across many pages (e.g., headers/footers) to reduce noise.
    Lines appearing >= threshold times are stripped.
    """
    lines = [ln.strip() for ln in (text or "").splitlines()]
    freq = {}
    for ln in lines:
        if ln:
            freq[ln] = freq.get(ln, 0) + 1
    keep = [ln for ln in lines if not ln or freq.get(ln, 0) < threshold]
    return "\n".join(keep)


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
    Adds Unicode normalization, header/footer de-dup, and PDF page cap.
    """
    name = (filename or "").lower()
    try:
        if name.endswith(".pdf"):
            text_chunks: List[str] = []
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for i, page in enumerate(doc):
                    if i >= _MAX_PAGES:
                        break
                    # Explicit "text" extraction mode for consistency
                    text_chunks.append(page.get_text("text") or "")
            txt = "\n".join(text_chunks).strip()
            txt = _u(txt)
            txt = _strip_repeating_lines(txt, threshold=3)
            return txt
        elif name.endswith(".docx"):
            txt = _read_docx_best_effort(file_bytes)
            txt = _u(txt)
            txt = _strip_repeating_lines(txt, threshold=3)
            return txt
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
    return _year(tok) or _year2(tok)

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
        if s <= cur_end:  # overlap or touch
            if e > cur_end:
                cur_end = e
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))
    return merged

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
        mm = _month(m.group(1))
        yy = _parse_year_token(m.group(2))
        if mm and yy:
            return datetime(yy, mm, 1)

    # "2023-09" or "2023/09"
    m = re.match(r"\b(20\d{2})[-/](\d{1,2})\b", t)
    if m:
        year, month = int(m.group(1)), int(m.group(2))
        if 1 <= month <= 12:
            return datetime(year, month, 1)

    # "09/2023" or "9-2023"
    m = re.match(r"\b(\d{1,2})[-/](20\d{2})\b", t)
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

    # "2023" only
    m = re.match(r"\b(20\d{2})\b", t)
    if m:
        return datetime(int(m.group(1)), 12, 1)

    # "’21" or "'21" (assume December)
    m = re.match(r"^'(\d{2})$", t)
    if m:
        yy2 = _year2(m.group(1))
        if yy2:
            return datetime(yy2, 12, 1)

    return None


# ----------------------------- Experience estimation -----------------------------
def estimate_total_experience_years(text: str) -> float:
    """
    Estimate total professional experience in years from free-text by detecting
    dateranges like 'Jan 2019 - Present' or '2016–2021'. Overlapping ranges are merged.
    If no ranges are found, falls back to span between min/max year tokens.

    Returns float rounded to two decimals.
    """
    text = _u(text or "")
    now = datetime.now()

    # Flexible patterns:
    #   - Jan 2019 - Present
    #   - Jul'21 – Nov'23
    #   - 2018 to 2022
    #   - Mar 2016 – Nov 2019
    #   - 07/21 - 02/24
    month = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*"
    yr4   = r"(19[8-9]\d|20[0-5]\d)"
    yr2   = r"\d{2}"
    # any month-year token or pure year token
    my    = rf"(?:{month}\s*'?(?:{yr4}|{yr2})|(?:{yr4}|{yr2})[-/]\d{{1,2}}|\d{{1,2}}[-/](?:{yr4}|{yr2})|{yr4}|'{yr2})"
    sep   = r"(?:-|–|—|to|till|through)"
    date_pat = re.compile(rf"({my})\s*{sep}\s*(present|current|now|{my})", re.I)

    ranges: List[Tuple[datetime, datetime]] = []
    for m in date_pat.finditer(text):
        start_tok, end_tok = m.group(1), m.group(2)

        def _parse(tok: str, is_end: bool) -> Optional[datetime]:
            if re.fullmatch(r"(?i)present|current|now", tok.strip()):
                return datetime(now.year, now.month, 1)
            dt = _parse_month_year(tok)
            if dt:
                return dt
            # last resort: bare year or 'yy
            yy = _parse_year_token(tok.strip().strip("'"))
            if yy:
                return datetime(yy, 12 if is_end else 1, 1)
            return None

        sdt = _parse(start_tok, is_end=False)
        edt = _parse(end_tok, is_end=True)

        if sdt and edt and edt >= sdt:
            ranges.append((sdt, edt))

    # Merge overlapping/adjacent ranges before summation
    if ranges:
        merged = _merge_ranges(ranges)
        total_days = sum((e - s).days for s, e in merged)
        return round(total_days / 365.25, 2)

    # Fallback: min/max year window if at least two plausible years are present (incl. 2-digit)
    year_hits: List[int] = []
    for y in re.findall(r"(19[8-9]\d|20[0-5]\d|\b'\d{2}\b|\b\d{2}\b)", text):
        yy = _parse_year_token(y.strip().strip("'"))
        if yy and 1980 <= yy <= datetime.now().year + 1:
            year_hits.append(yy)

    if len(year_hits) >= 2:
        return round(float(max(year_hits) - min(year_hits)), 2)

    return 0.0
