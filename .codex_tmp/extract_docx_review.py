import json
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from docx import Document


NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}


def para_style_name(paragraph):
    try:
        return paragraph.style.name or ""
    except Exception:
        return ""


def table_text(table):
    rows = []
    for row in table.rows:
        rows.append([cell.text.strip() for cell in row.cells])
    return rows


def iter_block_items(parent):
    # python-docx exposes paragraphs/tables separately; preserve enough order from XML.
    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    parent_elm = parent.element.body
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield "paragraph", Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield "table", Table(child, parent)


def collect_docx(path):
    doc = Document(path)
    blocks = []
    headings = []
    tables = []
    para_index = 0
    table_index = 0

    for kind, item in iter_block_items(doc):
        if kind == "paragraph":
            text = re.sub(r"\s+", " ", item.text).strip()
            style = para_style_name(item)
            if text:
                blocks.append({"type": "p", "i": para_index, "style": style, "text": text})
                if "Heading" in style or re.match(r"^\d+(\.\d+)*\s+\S+", text):
                    headings.append({"i": para_index, "style": style, "text": text})
            para_index += 1
        else:
            rows = table_text(item)
            tables.append({"i": table_index, "rows": rows})
            blocks.append({"type": "table", "i": table_index, "rows": rows[:5]})
            table_index += 1

    return {"blocks": blocks, "headings": headings, "tables": tables}


def collect_xml_refs(path):
    refs = {"figures": [], "tables": [], "fields": [], "hyperlinks": [], "raw_ref_like": []}
    with zipfile.ZipFile(path) as zf:
        document_xml = zf.read("word/document.xml")
    root = ET.fromstring(document_xml)
    texts = []
    for t in root.findall(".//w:t", NS):
        if t.text:
            texts.append(t.text)
    joined = " ".join(texts)

    for pattern, key in [
        (r"\b(Fig(?:ure)?\.?\s*\d+[A-Za-z]?)", "figures"),
        (r"\b(Table\s*\d+[A-Za-z]?)", "tables"),
    ]:
        refs[key] = sorted(set(re.findall(pattern, joined, flags=re.I)))

    refs["raw_ref_like"] = sorted(set(re.findall(r"\[[0-9,\-\s;]+\]", joined)))[:500]
    refs["fields"] = re.findall(r"(REF|SEQ|PAGEREF|TOC|HYPERLINK)[^<]{0,120}", document_xml.decode("utf-8", errors="ignore"))[:500]
    return refs


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: extract_docx_review.py INPUT.docx OUTPUT.json")
    path = Path(sys.argv[1])
    out = Path(sys.argv[2])
    data = collect_docx(path)
    data["refs"] = collect_xml_refs(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
