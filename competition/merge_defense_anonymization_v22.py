"""Apply Artifact Tool-authored text edits without reserializing unrelated assets."""

import argparse
import io
import json
import zipfile
from pathlib import Path

from lxml import etree


NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}
FORBIDDEN = ["乔安琪", "陈雍杰", "郭千弘", "404 Not Found", "长沙理工", "学院", "指导老师", "指导教师"]


def replace_text(container, value):
    nodes = container.findall(".//a:t", NS)
    if not nodes:
        raise ValueError("Existing text run not found")
    nodes[0].text = value
    for node in nodes[1:]:
        node.text = ""


def text(container):
    return "".join(container.xpath(".//a:t/text()", namespaces=NS))


def serialize(root):
    return etree.tostring(root, encoding="utf-8", xml_declaration=True)


def scrub_properties(payload, name):
    root = etree.fromstring(payload)
    for node in root.iter():
        tag = etree.QName(node).localname
        if tag in {"creator", "lastModifiedBy", "Company", "Manager"}:
            node.text = ""
        elif name == "docProps/app.xml" and tag in {"Slides", "Notes"}:
            node.text = "15"
    return serialize(root)


def main(source, authored, output):
    if output.exists():
        raise FileExistsError(output)
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(authored) as edited:
        if any("comment" in n.lower() or "persons/" in n for n in original.namelist()):
            raise ValueError("Review comment parts require explicit anonymization")
        replacements = {}
        root = etree.fromstring(original.read("ppt/slides/slide2.xml"))
        draft = etree.fromstring(edited.read("ppt/slides/slide2.xml"))
        old_cells = root.findall(".//a:tbl/a:tr/a:tc", NS)
        new_cells = draft.findall(".//a:tbl/a:tr/a:tc", NS)
        assert len(old_cells) == len(new_cells) == 8
        for old, new in zip(old_cells, new_cells):
            replace_text(old, text(new))
        for old, new in [("团队成员与分工", "项目实现与分工"), ("404 Not Found队", "算法、交互与答辩材料")]:
            shapes = [s for s in root.findall(".//p:sp", NS) if text(s) == old]
            assert len(shapes) == 1, old
            assert any(text(s) == new for s in draft.findall(".//p:sp", NS)), new
            replace_text(shapes[0], new)
        replacements["ppt/slides/slide2.xml"] = serialize(root)

        closing = etree.fromstring(original.read("ppt/slides/slide15.xml"))
        labels = [s for s in closing.findall(".//p:sp", NS) if text(s) == "404 Not Found队"]
        assert len(labels) == 1
        replace_text(labels[0], "AI 超表面结构色智能设计系统")
        replacements["ppt/slides/slide15.xml"] = serialize(closing)

        for number in [1, 2]:
            part = f"ppt/notesSlides/notesSlide{number}.xml"
            notes = etree.fromstring(original.read(part))
            draft_notes = etree.fromstring(edited.read(part))
            old_bodies = notes.xpath('.//p:sp[p:nvSpPr/p:nvPr/p:ph[@type="body"]]', namespaces=NS)
            new_bodies = draft_notes.xpath('.//p:sp[p:nvSpPr/p:nvPr/p:ph[@type="body"]]', namespaces=NS)
            old_body = old_bodies[0] if len(old_bodies) == 1 else None
            new_body = new_bodies[0] if len(new_bodies) == 1 else None
            if old_body is None or new_body is None:
                raise ValueError("Speaker notes body missing")
            paragraphs = new_body.findall("p:txBody/a:p", NS)
            body = old_body.find("p:txBody", NS)
            for p in body.findall("a:p", NS):
                body.remove(p)
            for p in paragraphs:
                body.append(p)
            replacements[part] = serialize(notes)

        for part in ["docProps/core.xml", "docProps/app.xml"]:
            replacements[part] = scrub_properties(original.read(part), part)

        for name in original.namelist():
            if name.endswith(".xlsx"):
                payload = original.read(name)
                with zipfile.ZipFile(io.BytesIO(payload)) as workbook:
                    # Refuse any hidden identity in embedded workbooks; don't touch data.
                    for part in workbook.namelist():
                        if part.endswith((".xml", ".rels")):
                            value = workbook.read(part).decode("utf-8", "ignore")
                            assert not any(term in value for term in FORBIDDEN), (name, part)

        for name in original.namelist():
            if name.endswith((".xml", ".rels")):
                value = replacements.get(name, original.read(name)).decode("utf-8", "ignore")
                assert not any(term in value for term in FORBIDDEN), name

        with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as result:
            for info in original.infolist():
                result.writestr(info, replacements.get(info.filename, original.read(info.filename)))
        print(json.dumps({"changed_parts": list(replacements), "all_other_parts_preserved": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ["source", "authored", "output"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.authored, args.output)
