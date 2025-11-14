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
    return out
