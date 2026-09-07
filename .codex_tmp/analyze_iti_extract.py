import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip()


def block_text(block):
    if block["type"] == "p":
        return block["text"]
    return " | ".join(" ; ".join(row) for row in block.get("rows", []))


def print_section(title, lines):
    print(f"\n## {title}")
    for line in lines:
        print(line)


def find_block_index(blocks, patterns):
    for idx, b in enumerate(blocks):
        t = block_text(b)
        if all(re.search(p, t, re.I) for p in patterns):
            return idx
    return None


def section_after_heading(blocks, heading_regex, max_blocks=80):
    start = None
    for idx, b in enumerate(blocks):
        if b["type"] == "p" and re.search(heading_regex, b["text"], re.I):
            start = idx
            break
    if start is None:
        return []
    out = []
    for b in blocks[start : start + max_blocks]:
        txt = block_text(b)
        if txt:
            out.append(f"[{b['type']}:{b['i']}] {txt[:800]}")
    return out


def refs_candidates(blocks):
    refs_start = None
    for idx, b in enumerate(blocks):
        txt = block_text(b)
        if re.fullmatch(r"references|bibliography", txt.strip(), flags=re.I):
            refs_start = idx
            break
    if refs_start is None:
        for idx, b in enumerate(blocks):
            if re.match(r"^\s*\[\d+\]", block_text(b)):
                refs_start = idx
                break
    refs = []
    if refs_start is not None:
        for b in blocks[refs_start : refs_start + 260]:
            txt = block_text(b)
            if txt:
                refs.append(f"[{b['type']}:{b['i']}] {txt}")
    return refs


def main():
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    blocks = data["blocks"]

    headings = []
    for h in data["headings"]:
        text = h["text"]
        if len(text) < 220:
            headings.append(f"[p:{h['i']}] {h['style']} :: {text}")
    print_section("HEADINGS", headings[:220])

    table_summaries = []
    for tbl in data["tables"]:
        rows = tbl["rows"]
        preview = " | ".join(" ; ".join(cell[:80] for cell in row[:6]) for row in rows[:3])
        table_summaries.append(f"[table:{tbl['i']}] rows={len(rows)} preview={preview[:1000]}")
    print_section("TABLES", table_summaries[:80])

    for name, regex, max_blocks in [
        ("INTRODUCTION", r"^\s*1\.?\s+Introduction\b|^\s*Introduction\b", 80),
        ("SECTION 1.1", r"^\s*1\.1\b", 60),
        ("SECTION 2", r"^\s*2\b", 160),
        ("TABLE 1 AREA", r"Table\s*1|Comparative analysis|comparison", 80),
        ("REFERENCES AREA", r"^\s*References\s*$|^\s*Bibliography\s*$", 120),
    ]:
        print_section(name, section_after_heading(blocks, regex, max_blocks))

    key_patterns = [
        "Bianchi", "Shang", "Sun", "Khajehdezfuly", "2026", "arXiv", "Peng,Jianping", "Han", "乱码", "GPR", "ultrasonic", "InSAR", "DAS", "point cloud", "LiDAR", "vibration",
    ]
    hits = []
    for b in blocks:
        txt = block_text(b)
        if any(k.lower() in txt.lower() for k in key_patterns):
            hits.append(f"[{b['type']}:{b['i']}] {txt[:1200]}")
    print_section("KEYWORD HITS", hits[:220])

    refs = refs_candidates(blocks)
    print_section("REFERENCE CANDIDATES", refs[:260])

    names_years = []
    for line in refs:
        clean = re.sub(r"^\[[^\]]+\]\s*", "", line)
        m = re.search(r"([A-Z][A-Za-z'`\-]+).*?\b(20\d{2})([a-z]?)\b", clean)
        if m:
            names_years.append((m.group(1), m.group(2), m.group(3), clean[:260]))
    groups = defaultdict(list)
    for last, year, suffix, clean in names_years:
        groups[(last.lower(), year)].append((suffix, clean))
    dupes = []
    for key, vals in groups.items():
        if len(vals) > 1:
            dupes.append(f"{key}: " + " || ".join(v[1] for v in vals[:6]))
    print_section("POSSIBLE SAME AUTHOR-YEAR CLUSTERS", dupes[:80])

    print_section("REF TOKEN SUMMARY", [
        f"figures={data['refs'].get('figures', [])[:80]}",
        f"tables={data['refs'].get('tables', [])[:80]}",
        f"bracket_refs_sample={data['refs'].get('raw_ref_like', [])[:80]}",
        f"fields_sample={data['refs'].get('fields', [])[:20]}",
    ])


if __name__ == "__main__":
    main()
