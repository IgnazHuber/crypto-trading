# utils/formatting.py

import re

def safe_sheet_name(name, maxlen=28):
    name = re.sub(r'[\/\\\?\*\[\]\:]', '_', name)
    return name[:maxlen]

def round1(x):
    try:
        return round(float(x), 1)
    except Exception:
        return x
