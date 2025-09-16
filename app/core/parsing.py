import io, re, tempfile
from datetime import datetime
from typing import Optional
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
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"sept":9,"oct":10,"nov":11,"dec":12
}
DEGREE_MAP = [
    (r"\b(ph\.?d\.?|doctorate)\b", "phd"),
    (r"\b(masters?|m\.?sc\.?|m\.?tech|m\.?eng|m\.?e\.)\b", "master"),
    (r"\b(bachelors?|b\.?sc\.?|b\.?tech|b\.?e\.?|b\.?eng)\b", "bachelor"),
    (r"\b(associate|diploma)\b", "associate"),
    (r"\b(certificate|certification)\b", "certificate"),
]

def _read_docx_best_effort(file_bytes: bytes) -> str:
    if mammoth:
        try:
            result = mammoth.convert_to_markdown(io.BytesIO(file_bytes))
            return result.value or ""
        except Exception:
            pass
    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".docx") as tmp:
            tmp.write(file_bytes); tmp.flush()
            doc = Document(tmp.name)
            return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        pass
    if docx2txt:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(file_bytes); tmp.flush()
                return docx2txt.process(tmp.name) or ""
        except Exception:
            pass
    return ""

def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    name = (filename or "").lower()
    try:
        if name.endswith(".pdf"):
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                return "\n".join([p.get_text() or "" for p in doc])
        elif name.endswith(".docx"):
            return _read_docx_best_effort(file_bytes)
        else:
            return ""
    except Exception:
        return ""

def guess_education(text: str) -> str:
    low = (text or "").lower()
    for pat, label in DEGREE_MAP:
        if re.search(pat, low):
            return label
    return "none"

def _year(s: Optional[str]) -> Optional[int]:
    if s is None: return None
    try:
        y = int(s)
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

def estimate_total_experience_years(text: str) -> float:
    text = text or ""
    now = datetime.now()
    year_hits = []
    ranges = []

    date_pat = re.compile(
        r"(?:(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\s+)?(19[8-9]\d|20[0-4]\d)"
        r"\s*(?:-|–|to|till|—)\s*"
        r"(?:(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\s+)?(present|current|now|19[8-9]\d|20[0-4]\d)",
        re.I
    )
    for m in date_pat.finditer(text):
        m1, y1s, m2, y2s = m.group(1), m.group(2), m.group(3), m.group(4)
        y1 = _year(y1s)
        y2 = now.year if re.match(r"present|current|now", (y2s or ""), re.I) else _year(y2s)
        if y1 and y2:
            mm1 = _month(m1) or 1
            mm2 = _month(m2) or 12
            try:
                d1 = datetime(y1, mm1, 1)
                d2 = datetime(y2, mm2, 1)
            except Exception:
                continue
            if d2 >= d1:
                ranges.append((d1, d2))

    if ranges:
        total_days = sum((d2 - d1).days for d1, d2 in ranges)
        return round(total_days / 365.25, 2)

    for y in re.findall(r"(19[8-9]\d|20[0-4]\d)", text):
        try:
            yy = int(y)
        except Exception:
            continue
        if 1980 <= yy <= datetime.now().year + 1:
            year_hits.append(yy)
    if len(year_hits) >= 2:
        return float(max(year_hits) - min(year_hits))
    return 0.0
