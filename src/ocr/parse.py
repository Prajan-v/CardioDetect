import re


def _num(x):
    try:
        return float(x)
    except Exception:
        return None


def _to_mgdl(val, unit):
    if val is None:
        return None
    u = unit.lower()
    if u in ["mg/dl", "mgdl", "mg%"]:
        return val
    if u in ["mmol/l", "mmol"]:
        return val * 38.67
    return val

def _in_range(v, lo, hi):
    try:
        fv = float(v)
    except Exception:
        return None
    if fv < lo or fv > hi:
        return None
    return fv


def parse_text(text: str):
    out = {}
    m = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", text, re.I)
    if m:
        out["bp_sys"] = _num(m.group(1))
        out["bp_dia"] = _num(m.group(2))
    m = re.search(r"\bchol(?:esterol)?\b[^\d]*(\d{2,4})(?:\s*(mg/dl|mmol/l|mgdl|mg%))?", text, re.I)
    if m:
        out["chol"] = _to_mgdl(_num(m.group(1)), m.group(2) or "mg/dl")
    m = re.search(r"\bhdl\b[^\d]*(\d{1,3})(?:\s*(mg/dl|mmol/l|mgdl|mg%))?", text, re.I)
    if m:
        out["hdl"] = _to_mgdl(_num(m.group(1)), m.group(2) or "mg/dl")
    m = re.search(r"\bldl\b[^\d]*(\d{1,3})(?:\s*(mg/dl|mmol/l|mgdl|mg%))?", text, re.I)
    if m:
        out["ldl"] = _to_mgdl(_num(m.group(1)), m.group(2) or "mg/dl")
    m = re.search(r"\btrig(?:lycerides?)?\b[^\d]*(\d{2,4})(?:\s*(mg/dl|mmol/l|mgdl|mg%))?", text, re.I)
    if m:
        out["triglycerides"] = _to_mgdl(_num(m.group(1)), m.group(2) or "mg/dl")
    m = re.search(r"\bglucose\b[^\d]*(\d{2,4})(?:\s*(mg/dl|mmol/l|mgdl|mg%))?", text, re.I)
    if m:
        out["glucose"] = _to_mgdl(_num(m.group(1)), m.group(2) or "mg/dl")
    m = re.search(r"\bage\b[^\d]*(\d{1,3})", text, re.I)
    if m:
        age = _in_range(m.group(1), 18, 120)
        if age is not None:
            out["age"] = age
    m = re.search(r"\b(?:hr|heart\s*rate)\b[^\d]*(\d{2,3})\s*(?:bpm)?", text, re.I)
    if m:
        hr = _in_range(m.group(1), 30, 220)
        if hr is not None:
            out["heart_rate"] = hr
    return out
