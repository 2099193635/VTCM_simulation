import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def text_of(block):
    if block["type"] == "p":
        return block.get("text", "")
    return " | ".join(" ; ".join(row) for row in block.get("rows", []))


data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
blocks = data["blocks"]
start = next(
    (
        i
        for i, b in enumerate(blocks)
        if b.get("type") == "p" and b.get("text", "").strip().lower() in {"reference", "references", "bibliography"}
    ),
    None,
)

print(f"REFERENCE_START={start}")
refs = []
if start is not None:
    for b in blocks[start : start + 260]:
        txt = text_of(b).strip()
        if txt:
            print(f"[{b['type']}:{b['i']}] {txt[:1400]}")
            if b["type"] == "p" and b["i"] != 362:
                refs.append(txt)

print("\n## AUTHOR-YEAR CLUSTERS")
groups = defaultdict(list)
for r in refs:
    m = re.search(r"\b([A-Z][A-Za-z'`\-]+)[^()]{0,140}\((20\d{2})([a-z]?)\)", r)
    if not m:
        m = re.search(r"\b([A-Z][A-Za-z'`\-]+)[^,]{0,80},\s+.*?\b(20\d{2})([a-z]?)\b", r)
    if m:
        groups[(m.group(1).lower(), m.group(2))].append(r[:500])
for key, vals in sorted(groups.items()):
    if len(vals) > 1:
        print(f"{key}:")
        for v in vals:
            print(f"  - {v}")

print("\n## 2026_OR_ARXIV")
for r in refs:
    if "2026" in r or re.search(r"arxiv|preprint", r, re.I):
        print(f"- {r[:700]}")

print("\n## FORMAT_SUSPECTS")
for r in refs:
    if re.search(r"[A-Za-z],[A-Z]|，|;;|  |Ã|�|Accessed|doi:|DOI", r):
        print(f"- {r[:700]}")
