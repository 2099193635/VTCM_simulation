import json
import re
import sys
from pathlib import Path


data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for b in data["blocks"]:
    if b["type"] != "p":
        continue
    t = b.get("text", "").strip()
    if re.match(r"^(Figure|Table)\b", t, re.I) or "Figure ." in t or "Table ." in t:
        print(f"[p:{b['i']}] {t}")
