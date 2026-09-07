from __future__ import annotations

import argparse
import datetime as dt
import zipfile
from copy import deepcopy
from pathlib import Path

from lxml import etree


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}


def w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def xml_bytes(root: etree._Element) -> bytes:
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")


def enable_track(settings_root: etree._Element) -> None:
    if settings_root.find("w:trackRevisions", namespaces=NS) is None:
        settings_root.insert(0, etree.Element(w("trackRevisions")))


def next_change_id(doc_root: etree._Element) -> int:
    ids = []
    for el in doc_root.xpath(".//*[@w:id]", namespaces=NS):
        try:
            ids.append(int(el.get(w("id"))))
        except Exception:
            pass
    return max(ids, default=0) + 1


def visible_text(el: etree._Element) -> str:
    parts = []
    for node in el.xpath(".//w:t|.//w:delText", namespaces=NS):
        if node.text:
            parts.append(node.text)
    return "".join(parts)


class ChangeMaker:
    def __init__(self, doc_root: etree._Element, author: str):
        self.doc_root = doc_root
        self.author = author
        self.when = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        self.cid = next_change_id(doc_root)

    def _r_with_text(self, text: str, tag: str = "t", rpr: etree._Element | None = None) -> etree._Element:
        r = etree.Element(w("r"))
        if rpr is not None:
            r.append(deepcopy(rpr))
        t = etree.SubElement(r, w(tag))
        if text.startswith(" ") or text.endswith(" "):
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        t.text = text
        return r

    def ins(self, text: str, rpr: etree._Element | None = None) -> etree._Element:
        el = etree.Element(w("ins"))
        el.set(w("id"), str(self.cid))
        el.set(w("author"), self.author)
        el.set(w("date"), self.when)
        self.cid += 1
        el.append(self._r_with_text(text, "t", rpr))
        return el

    def delete(self, text: str, rpr: etree._Element | None = None) -> etree._Element:
        el = etree.Element(w("del"))
        el.set(w("id"), str(self.cid))
        el.set(w("author"), self.author)
        el.set(w("date"), self.when)
        self.cid += 1
        el.append(self._r_with_text(text, "delText", rpr))
        return el

    def replace_paragraph(self, old_text: str, new_text: str) -> bool:
        for p in self.doc_root.xpath(".//w:body/w:p", namespaces=NS):
            if visible_text(p).strip() == old_text.strip():
                rpr = None
                first_r = p.find("w:r", namespaces=NS)
                if first_r is not None:
                    rpr = first_r.find("w:rPr", namespaces=NS)
                for child in list(p):
                    if child.tag != w("pPr"):
                        p.remove(child)
                p.append(self.delete(old_text, rpr))
                p.append(self.ins(new_text, rpr))
                return True
        return False

    def replace_paragraph_startswith(self, prefix: str, new_text: str) -> bool:
        for p in self.doc_root.xpath(".//w:body/w:p", namespaces=NS):
            txt = visible_text(p).strip()
            if txt.startswith(prefix):
                return self.replace_paragraph(txt, new_text)
        return False

    def inserted_paragraph(self, text: str, style: str | None = None) -> etree._Element:
        p = etree.Element(w("p"))
        if style:
            ppr = etree.SubElement(p, w("pPr"))
            pstyle = etree.SubElement(ppr, w("pStyle"))
            pstyle.set(w("val"), style)
        p.append(self.ins(text))
        return p

    def replace_cell_text(self, tc: etree._Element, new_text: str, old_text: str | None = None) -> None:
        if old_text is None:
            old_text = visible_text(tc).strip()
        tc_pr = tc.find("w:tcPr", namespaces=NS)
        for child in list(tc):
            if child.tag != w("tcPr"):
                tc.remove(child)
        p = etree.SubElement(tc, w("p"))
        if old_text:
            p.append(self.delete(old_text))
        if new_text:
            p.append(self.ins(new_text))


def find_body(root: etree._Element) -> etree._Element:
    body = root.find("w:body", namespaces=NS)
    if body is None:
        raise RuntimeError("word/document.xml has no w:body")
    return body


def make_bordered_table(cm: ChangeMaker, rows: list[list[str]]) -> etree._Element:
    tbl = etree.Element(w("tbl"))
    tbl_pr = etree.SubElement(tbl, w("tblPr"))
    tbl_w = etree.SubElement(tbl_pr, w("tblW"))
    tbl_w.set(w("w"), "0")
    tbl_w.set(w("type"), "auto")
    borders = etree.SubElement(tbl_pr, w("tblBorders"))
    for side in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        b = etree.SubElement(borders, w(side))
        b.set(w("val"), "single")
        b.set(w("sz"), "4")
        b.set(w("space"), "0")
        b.set(w("color"), "BFBFBF")
    grid = etree.SubElement(tbl, w("tblGrid"))
    col_count = max(len(r) for r in rows)
    for _ in range(col_count):
        col = etree.SubElement(grid, w("gridCol"))
        col.set(w("w"), str(int(9000 / col_count)))
    for ridx, row in enumerate(rows):
        tr = etree.SubElement(tbl, w("tr"))
        for cell in row:
            tc = etree.SubElement(tr, w("tc"))
            tc_pr = etree.SubElement(tc, w("tcPr"))
            shading = etree.SubElement(tc_pr, w("shd"))
            shading.set(w("fill"), "D9EAF7" if ridx == 0 else "FFFFFF")
            p = etree.SubElement(tc, w("p"))
            p.append(cm.ins(cell))
    return tbl


def insert_coverage_matrix(root: etree._Element, cm: ChangeMaker) -> None:
    body = find_body(root)
    tables = body.findall("w:tbl", namespaces=NS)
    if not tables:
        raise RuntimeError("No tables found")
    first_table = tables[0]
    idx = body.index(first_table)
    caption = cm.inserted_paragraph(
        "Table 1b. Coverage matrix of previous railway AI/O&M reviews and the present dual-loop review",
        "Caption",
    )
    note = cm.inserted_paragraph(
        "The matrix clarifies that prior reviews mainly emphasize sensing, monitoring, assessment or AI algorithms, whereas this review explicitly connects decision optimization, physical/robotic execution, lifecycle feedback and field validation within a dual closed-loop O&M architecture."
    )
    rows = [
        ["Review", "Perception", "Assessment", "Decision optimization", "Robotic execution", "Lifecycle feedback", "Field case"],
        ["Peinado Gonzalo et al. (2022)", "Yes", "Yes", "Partial", "No", "Partial", "No"],
        ["Tang et al. (2022)", "Yes", "Partial", "Partial", "Partial", "No", "No"],
        ["Aela, Chi, et al. (2024)", "Yes", "Partial", "No", "No", "No", "No"],
        ["Aela, Cai, et al. (2024)", "Yes", "Partial", "No", "No", "No", "No"],
        ["Koohmishi et al. (2024)", "Yes", "Yes", "Partial", "No", "Partial", "No"],
        ["Rahman et al. (2024)", "Yes", "Partial", "No", "No", "No", "No"],
        ["Khajehdezfuly et al. (2025)", "Yes", "Yes", "Partial", "No", "Partial", "No"],
        ["Olivier et al. (2025)", "Yes", "Partial", "No", "No", "No", "No"],
        ["Zhang, Zhang and Dinavahi (2025)", "Partial", "Yes", "Partial", "No", "No", "No"],
        ["Zhang et al. (2026)", "Yes", "Yes", "Partial", "No", "Partial", "No"],
        ["This review", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes"],
    ]
    tbl = make_bordered_table(cm, rows)
    body.insert(idx + 1, caption)
    body.insert(idx + 2, tbl)
    body.insert(idx + 3, note)


def revise_measurement_table(root: etree._Element, cm: ChangeMaker) -> None:
    body = find_body(root)
    tables = body.findall("w:tbl", namespaces=NS)
    if len(tables) < 2:
        raise RuntimeError("Expected at least two tables")
    tbl = tables[1]
    new_rows = [
        ["Technology", "State variable measured", "Operational role", "What it enables", "What it misses / uncertainty", "Dual-loop requirement"],
        ["Eddy current testing", "Rail-surface and near-surface electromagnetic response", "Local precision inspection", "Fast screening of surface rolling-contact fatigue and conductive defects", "Sensitive to lift-off, roughness, contamination and non-conductive defects", "Use as confirmatory evidence after network alerts; report uncertainty before repair decisions"],
        ["Magnetic flux leakage", "Magnetic-field disturbance from rail surface or near-surface discontinuities", "Local precision inspection", "High-sensitivity detection of rail discontinuities at inspection speed", "Affected by rust, oil, magnetization history and defect orientation", "Fuse with visual/ultrasonic evidence before classifying repairable defect geometry"],
        ["Visual inspection", "Surface appearance of rails, fasteners, sleepers and visible components", "Network survey and local inspection", "Scalable defect localization, documentation and asset inventory", "Cannot observe subsurface damage; affected by lighting, occlusion, weather and label bias", "Convert detections into asset-level work orders with confidence scores and human/robot verification"],
        ["Ultrasonic testing", "Internal acoustic reflections from cracks, breaks and inclusions", "Local precision diagnosis", "Quantifies internal rail defects and supports safety-critical acceptance decisions", "Blind zones, coupling condition, speed constraints and incomplete cross-section coverage", "Provide defect size/depth uncertainty and trigger repair or speed restriction only with traceable thresholds"],
        ["Ground penetrating radar", "Dielectric contrast linked to ballast fouling, voids, moisture and slab/subgrade interfaces", "Corridor-scale substructure survey plus targeted diagnosis", "Reveals hidden causal parameters behind repeated geometry defects", "Multipath scattering, moisture sensitivity, blurred boundaries and interpretation dependence", "Link subsurface causes to deterioration forecasts and select tamping, cleaning, drainage or slab repair actions"],
        ["On-board vibration/acoustic sensing", "Vehicle dynamic response and wheel-rail noise induced by track irregularity and local defects", "High-frequency network census", "Low-cost continuous screening and trend monitoring using in-service trains", "Inverse mapping is ill-posed; speed, load, vehicle condition and environment confound signals", "Use uncertainty-aware inverse models to generate prioritized alerts for the second precision loop"],
        ["DAS/fibre-optic sensing", "Distributed strain/acoustic signatures along trackside fibres", "Continuous corridor monitoring", "Captures train passages, impacts, intrusion events and evolving dynamic anomalies", "Signal interpretation depends on cable coupling, soil condition and noise environment", "Transform event streams into localized, confidence-ranked maintenance triggers"],
        ["InSAR/GNSS absolute measurement", "Long-wavelength displacement, settlement and geospatial alignment", "Network-scale deformation monitoring", "Identifies slow deformation and transition-zone settlement over large areas", "Temporal decorrelation, tunnel/urban signal loss and limited sensitivity to short-wave defects", "Anchor multi-source data to stable spatial coordinates for lifecycle tracking"],
        ["LiDAR/point cloud", "3D geometry of track corridor, ballast profile, clearances and visible assets", "Survey, inventory and local reconstruction", "Supports digital twins, clearance checks and geometry-aware robotic localization", "Occlusion, registration drift, density variation and weak subsurface observability", "Provide object-level geometry for decision optimization and executable robotic maintenance"],
    ]
    existing_rows = tbl.findall("w:tr", namespaces=NS)
    while len(existing_rows) < len(new_rows):
        existing_rows.append(deepcopy(existing_rows[-1]))
        tbl.append(existing_rows[-1])
    for ridx, row_data in enumerate(new_rows):
        tr = existing_rows[ridx]
        cells = tr.findall("w:tc", namespaces=NS)
        while len(cells) < len(row_data):
            cells.append(deepcopy(cells[-1]))
            tr.append(cells[-1])
        for cidx, text in enumerate(row_data):
            old = visible_text(cells[cidx]).strip() if ridx < 8 else ""
            cm.replace_cell_text(cells[cidx], text, old)


def apply_revisions(in_docx: Path, out_docx: Path, author: str) -> None:
    replacements = {
        "Over the past decade, China's railway network has undergone rapid and sustained expansion, producing a system of unprecedented scale and operational complexity. With the completion of the “Four Vertical and Four Horizontal” backbone network and progress toward the ambitious “Eight Vertical and Eight Horizontal” blueprint, the high-speed rail network has expanded substantially in both mileage and spatial coverage (Zhang et al., 2025).":
            "Railway systems worldwide are entering a maintenance-dominated era in which high-speed railways, heavy-haul corridors and urban rail networks must deliver higher availability with ageing assets, rising traffic density and tighter safety margins. The central challenge is no longer only how to build new infrastructure, but how to control lifecycle cost and safety risk while existing networks remain in continuous service.",
        "China's high-speed rail network is the largest by a substantial margin globally. HIGH SPEED LINES IN THE WORLD 2023, published by the International Union of Railways (Accessed 2026), demonstrates that China's railway industry leads in both absolute scale and development speed. In terms of absolute magnitude, China's operational network represents the largest concentration of high-speed rail infrastructure in a single national system. As shown in Figure 1, China's operational network has reached 40,493 km—roughly an order of magnitude larger than the world's second-largest network. From a continental perspective, the scale of China's single-nation network is 3.2 times larger than the entire operational network of Europe, which is 12,495 km. Regarding development velocity, China continues to":
            "Large railway networks in Asia, Europe, North America and rapidly urbanizing regions all face the same transition from expansion to asset stewardship. China provides an important large-scale example because its high-speed network combines exceptional mileage, high traffic intensity and diverse operating environments; however, the maintenance problem addressed in this review is international: dense railway systems must detect rare defects, interpret coupled infrastructure states and intervene within short possession windows.",
        "Figure Cross-sectional comparison of HSR network scales highlighting China's dominant position":
            "Figure Cross-sectional comparison of high-speed rail network scales as background for lifecycle O&M pressure",
        "However, the sustainability of high-speed rail will be a key focus in the coming years, particularly the goal of reducing its full life-cycle costs. As operational mileage continues to increase, the scale and complexity of maintenance work grow substantially, with its economic impact becoming increasingly significant. According to a report by the International Union of Railways, high-speed rail maintenance has become a major component of total lifecycle costs, with annual maintenance costs per kilometer reaching approximately 90,000 euros. In China, due to high train operating frequencies and heavy loads, the deterioration rate of track components (such as ballast) often exceeds European standards, resulting in a shortened actual service life. This further increases the frequency and cost ":
            "The operational difficulty arises from four intrinsic features of railway infrastructure: it is a large-scale linear asset exposed to heterogeneous geology and climate; its condition is governed by strong wheel-rail-structure coupling; defects are sparse, stochastic and often hidden beneath surface geometry; and maintenance can be performed only within limited possession windows. These features make railway O&M a lifecycle control problem rather than a simple inspection problem.",
        "Within this context, and accompanied by rapid advances in artificial intelligence, the railway industry's primary challenge has undergone a fundamental shift as shown in Figure 2: the industry’s primary technical challenge has shifted from ensuring construction quality to managing in-service degradation. During the construction era, the central question was whether infrastructure met design specifications at commissioning; in the current operational era, the critical question is how quickly performance degrades and whether intervention can be timed optimally. Facing this challenge, traditional passive or fixed-cycle O&M models are proving increasingly unsustainable under the combined pressure of accumulating safety risks and rising maintenance costs. Based on a demand-oriented approach, th":
            "This shift changes the role of intelligent technologies. AI, digital sensing and robotic systems are valuable only when they close the chain from observation to engineering action: perception must feed physical-state assessment, assessment must support deterioration prediction, prediction must lead to optimized maintenance decisions, and the executed intervention must be verified by feedback measurement.",
        "The resulting literature indicates that existing research has been concentrated mainly in several relatively narrow subdomains, especially rail defect detection, track geometry monitoring, sensor-based condition assessment, and data-driven degradation prediction. However, these advances are often discussed as separate technical tasks, with limited attention to how sensing, assessment, prediction, maintenance decision-making, field intervention, and post-maintenance verification can be connected across the infrastructure lifecycle. This fragmentation motivates the present review to move beyond a catalogue of AI techniques and to examine intelligent railway infrastructure O&M as an integrated process linking data acquisition, state understanding, deterioration forecasting, maintenance planni":
            "Existing reviews have made important contributions on individual sensing technologies, specific assets, defect detection methods and AI algorithms. What remains underdeveloped is a system-level framework that follows the complete O&M chain from perception to assessment, decision optimization, physical execution and feedback verification. This review therefore positions intelligent railway infrastructure O&M as a dual closed-loop system rather than as a catalogue of separate techniques.",
        "Inherent Constraints in the Existing O&M System":
            "Systemic Bottlenecks in Existing Railway O&M Systems",
        "This section reviews railway infrastructure O&M from four main aspects: perception, assessment, decision-making, and maintenance. It summarizes the representative technologies and methods at each stage, covering condition detection, state evaluation, maintenance decision support, and maintenance implementation. These aspects provide a systematic view of the technical chain in railway infrastructure O&M.":
            "This section reframes the literature around O&M bottlenecks rather than around isolated technologies. For each sensing or decision method, the review asks four questions: what state variable it measures, what maintenance function it enables, what uncertainty or blind spot it leaves, and what the dual closed-loop system requires to convert the observation into verified intervention.",
        "Precision Measurement Technologies":
            "Precision Measurement Bottlenecks: Observability, Uncertainty and Actionability",
        "Precision measurement technologies for railway infrastructure can be categorized by their observation targets into two complementary groups: explicit surface defects detectable via direct sensing, and implicit subsurface conditions requiring indirect non-destructive methods. Table 2 provides a systematic comparison, and key technologies are summarized in the following paragraphs.":
            "Precision measurement technologies should be evaluated by their contribution to an O&M loop, not only by their physical principle. A useful method must identify the state variable being observed, indicate whether it supports network-level census or local precision diagnosis, quantify uncertainty, and clarify whether its output can be translated into a maintenance action.",
        "Table . Comparison of Precision Detection Technologies for Railway Infrastructure":
            "Table 2. Precision detection technologies evaluated against dual-loop O&M requirements",
        "General Survey":
            "Network-Level General Survey Bottlenecks",
        "In conclusion, while macroscopic census measurements provide an essential baseline for network-level health screening, their systemic limitations comprehensively demonstrate that \"top-down\" macroscopic inspections can no longer independently satisfy the rigorous demands of modern railway safety. To break the cycle of reactive maintenance and bridge the fatal gaps in spatial masking, temporal lag, physical load discrepancies, and causal blindness, it is imperative to integrate targeted, short-term, and high-frequency \"bottom-up\" precise micro-inspection technologies.":
            "In conclusion, macroscopic census measurements remain indispensable for network-level screening, but they do not by themselves close the O&M loop. Their outputs must be connected to targeted precision diagnosis, uncertainty-aware deterioration prediction and executable maintenance decisions; otherwise, the system remains trapped in delayed threshold exceedance and weakly verified intervention.",
        "Condition assessment of railway infrastructure is a critical engineering issue for ensuring operational safety and reducing maintenance costs. In recent years, this field has undergone a profound transformation from traditional manual inspections and contact-based measurements to automated sensing and data-driven intelligent analysis. This section provides a systematic review of the three dimensions of railway condition assessment: (1) traditional methods represented by track geometry measurement, ultrasonic inspection, eddy current testing, and shunt monitoring; (2) cutting-edge technologies represented by structural health monitoring, ground-penetrating radar, distributed optical fiber sensing, and 3D point cloud measurement; (3) artificial intelligence applications represented by deep l":
            "Condition assessment is the interface between raw perception and maintenance action. The key question is not whether an algorithm can classify a defect, but whether the assessment converts heterogeneous observations into decision-ready variables such as physical state, confidence level, deterioration trend, threshold-exceedance probability and repairable engineering object.",
        "As we mentioned in the Introduction part before, several systematic reviews have provided an overview of the current state of the field in recent years. Bianchi et al.(2025b) provided a comprehensive classification of monitoring technologies ranging from classical to modern methods; Khajehdezfuly, Azizipour and Kaewunruen (2025) have reviewed more than 500 studies on AI applications and systematically described the current status of AI technology adoption across various components and fields within the railway sector. Building on the aforementioned work, this section constructs a comprehensive review framework spanning three tiers—traditional, cutting-edge, and AI—with a focus on methodological dimensions, supplemented by technological maturity and application scenarios.":
            "Building on these reviews, this section emphasizes what existing methods still miss from a closed-loop O&M perspective: causal observability, spatial-temporal alignment, uncertainty quantification, lifecycle prediction, decision optimization and feedback after physical intervention.",
        "Railway operation and maintenance":
            "Decision and Execution Bottlenecks",
        "Railway operation and maintenance (O&M) is a fundamental component for ensuring the safety, reliability, efficiency, and economic performance of railway systems. It covers multiple subsystems, including tracks, bridges, tunnels, power supply, signalling and communication, rolling stock, stations, and train operation organization. Compared with general industrial maintenance, railway O&M is characterized by large-scale assets, complex service environments, strong coupling among subsystems, limited maintenance windows, and strict safety constraints (Davari et al., 2021). With the continuous expansion of high-speed railway, heavy-haul railway, and urban rail transit networks, traditional maintenance modes that rely mainly on manual inspection, periodic repair, and experience-based judgement a":
            "The main O&M bottleneck is the gap between digital diagnosis and executable field intervention. Railway decisions must specify where to intervene, when the possession window is available, which repair action is physically appropriate, how resources are allocated, and how post-maintenance measurement verifies whether the intervention changed the infrastructure state as intended.",
        "Maintenance Research: Existing Knowledge and Technical Evolution":
            "From Knowledge Clusters to Closed-Loop Gaps",
    }

    with zipfile.ZipFile(in_docx, "r") as zin:
        doc_root = etree.fromstring(zin.read("word/document.xml"))
        settings_name = "word/settings.xml"
        settings_root = (
            etree.fromstring(zin.read(settings_name))
            if settings_name in zin.namelist()
            else etree.Element(w("settings"))
        )
        enable_track(settings_root)
        cm = ChangeMaker(doc_root, author)

        missing = []
        for old, new in replacements.items():
            if not cm.replace_paragraph(old, new):
                missing.append(old[:90])

        insert_coverage_matrix(doc_root, cm)
        revise_measurement_table(doc_root, cm)

        overrides = {
            "word/document.xml": xml_bytes(doc_root),
            settings_name: xml_bytes(settings_root),
        }
        with zipfile.ZipFile(out_docx, "w", zipfile.ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                if info.filename in overrides:
                    zout.writestr(info, overrides[info.filename])
                else:
                    zout.writestr(info, zin.read(info.filename))

    if missing:
        print("Missing replacements:")
        for item in missing:
            print(" -", item)
    print(f"Wrote {out_docx}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--out", required=True)
    ap.add_argument("--author", default="Codex")
    args = ap.parse_args()
    apply_revisions(Path(args.input), Path(args.out), args.author)


if __name__ == "__main__":
    main()
