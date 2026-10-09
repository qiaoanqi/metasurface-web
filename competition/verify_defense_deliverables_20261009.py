import argparse
import hashlib
import json
import posixpath
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from PIL import Image, ImageChops, ImageStat
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "competition" / "答辩材料_20261009"
BUILD = ROOT / ".defense-build-20261009"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest().upper()


def verify(version: str) -> dict:
    stem = f"AI超表面结构色智能设计系统_答辩修订版_{version}"
    pptx = OUTPUT / f"{stem}.pptx"
    pdf = OUTPUT / f"{stem}.pdf"
    handbook = OUTPUT / ("答辩准备手册_v1.md" if version == "v18" else "答辩准备手册_v2.md")
    validation = json.loads((BUILD / f"validation_{version}.json").read_text(encoding="utf-8"))
    assert validation["finalSha256"].upper() == sha256(pptx)
    assert validation["packageIntegrity"]["findingCount"] == 0
    assert validation["presentationLayout"]["findingCount"] == 0
    assert validation["firstPartyImport"]["passed"]
    assert validation["nativeQuantitativeCharts"]["report"]["passed"]

    with zipfile.ZipFile(pptx) as package:
        parts = package.namelist()
        slide_parts = [p for p in parts if re.fullmatch(r"ppt/slides/slide\d+\.xml", p)]
        note_parts = [p for p in parts if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml", p)]
        assert len(slide_parts) == len(note_parts) == 15
        texts = []
        for part in slide_parts + note_parts + ["docProps/core.xml"]:
            texts.append(" ".join(ElementTree.fromstring(package.read(part)).itertext()))
        combined = "\n".join(texts)
        slide_meta = json.loads((BUILD / ("slide_content.json" if version == "v18" else f"slide_content_{version}.json")).read_text(encoding="utf-8"))
        for item in slide_meta:
            notes_text = " ".join(ElementTree.fromstring(package.read(f"ppt/notesSlides/notesSlide{item['slide']}.xml")).itertext())
            assert item["script"] in notes_text, item["slide"]
        forbidden = [s for s in ["长沙理工大学", "人工智能学院", "指导教师", "指导老师"] if s in combined]
        assert not forbidden, forbidden
        for member in ["乔安琪", "陈雍杰", "郭千弘"]:
            assert member in combined
        team_slide = ElementTree.fromstring(package.read("ppt/slides/slide2.xml"))
        namespace = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
        team_tables = team_slide.findall(".//a:tbl", namespace)
        assert len(team_tables) == 1
        team_rows = [["".join(cell.itertext()) for cell in row.findall("a:tc", namespace)] for row in team_tables[0].findall("a:tr", namespace)]
        if version == "v18":
            assert team_rows == [["成员", "分工"], ["乔安琪", "项目负责人、核心算法、系统开发与主讲"], ["陈雍杰", "UI设计"], ["郭千弘", "PPT制作"]], team_rows
        else:
            with zipfile.ZipFile(BUILD / "source_v18_snapshot.pptx") as source:
                source_team = ElementTree.fromstring(source.read("ppt/slides/slide2.xml"))
                source_rows = [["".join(cell.itertext()) for cell in row.findall("a:tc", namespace)] for row in source_team.findall(".//a:tbl/a:tr", namespace)]
                assert team_rows == source_rows, (team_rows, source_rows)
                final_cover = [e.text for e in ElementTree.fromstring(package.read("ppt/slides/slide1.xml")).findall(".//a:t", namespace)]
                source_cover = [e.text for e in ElementTree.fromstring(source.read("ppt/slides/slide1.xml")).findall(".//a:t", namespace)]
                assert final_cover == source_cover
            assert "undefined" not in combined
        for stale in ["排练计时、材料核对", "演示备份、会议调试", "工作阶段", "核心技术贡献由项目负责人承担"]:
            assert stale not in combined, stale
        slide_four = ElementTree.fromstring(package.read("ppt/slides/slide4.xml"))
        slide_four_text = "".join(slide_four.itertext())
        for expected in ["五个页面", "预览", "逆设计", "图案", "映射", "光谱", "查看 D-H 颜色"]:
            assert expected in slide_four_text, expected
        map_table = slide_four.findall(".//a:tbl", namespace)
        assert len(map_table) == 1
        map_rows = map_table[0].findall("a:tr", namespace)
        assert len(map_rows) == 6
        map_evidence = json.loads((ROOT / ".state" / "offline_flow_mapping_single_20261006.json").read_text(encoding="utf-8"))
        for cell in map_evidence["payload"]["cells"]:
            table_cell = map_rows[5 - cell["hi"]].findall("a:tc", namespace)[cell["di"]]
            actual = table_cell.find("a:tcPr/a:solidFill/a:srgbClr", namespace).attrib["val"]
            expected = "".join(f"{int(v * 255 + 0.5):02X}" for v in cell["rgb"])
            assert actual.upper() == expected, (cell["hi"], cell["di"], actual, expected)
        if version != "v18":
            chart_ns = {"c": "http://schemas.openxmlformats.org/drawingml/2006/chart", "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
            series = {}
            for number in [7, 9]:
                owner = ElementTree.fromstring(package.read(f"ppt/slides/slide{number}.xml"))
                chart_id = owner.find(".//c:chart", chart_ns).attrib[f"{{{chart_ns['r']}}}id"]
                rels = ElementTree.fromstring(package.read(f"ppt/slides/_rels/slide{number}.xml.rels"))
                target = next(r.attrib["Target"] for r in rels if r.attrib["Id"] == chart_id)
                part = posixpath.normpath(posixpath.join("ppt/slides", target)).lstrip("/")
                chart = ElementTree.fromstring(package.read(part))
                series[number] = [[float(p.find("c:v", chart_ns).text) for p in chart.findall(f".//c:{axis}/c:numRef/c:numCache/c:pt", chart_ns)] for axis in ["xVal", "yVal"]]
            reference = json.loads(next(line for line in (ROOT / "competition/tio2_air_reference_records_v1.jsonl").open(encoding="utf-8") if line.strip()))
            assert series[9][0] == reference["wavelength_nm"]
            assert series[9][1] == [round(v, 6) for v in reference["R"]]
            assert series[7][0] == [round(50 + i * 300 / 7, 6) for i in range(8) for _ in range(8)]
            assert series[7][1] == [round(80 + j * 520 / 7, 6) for _ in range(8) for j in range(8)]

    reader = PdfReader(pdf)
    assert len(reader.pages) == 15
    assert all(abs(float(page.mediabox.width) - 960) < 0.01 and abs(float(page.mediabox.height) - 540) < 0.01 for page in reader.pages)
    differences = []
    for n in range(1, 16):
        source = Image.open(BUILD / f"final_slides_{version}" / f"slide-{n}.png").convert("RGB")
        rendered = Image.open(BUILD / f"pdf_render_{version}" / f"slide-{n:02}.png").convert("RGB")
        assert source.size == rendered.size == (1280, 720)
        difference = ImageStat.Stat(ImageChops.difference(source, rendered)).mean
        differences.append(max(difference))
        # PDF rasterizers apply font/image antialiasing differently from the
        # source PNG renderer. The hard checks are page count, dimensions and
        # visual inspection; this threshold catches accidental scaling rather
        # than normal cross-renderer antialiasing drift.
        assert max(difference) < 5.0, (n, difference)
        if version == "v18" and n != 4:
            previous = BUILD / "final_slides_v16" / f"slide-{n}.png"
            assert sha256(previous) == sha256(BUILD / f"final_slides_{version}" / f"slide-{n}.png"), n
        elif version != "v18" and n in [1, 2, 11, 12, 13]:
            previous = Image.open(BUILD / "source_v18_current" / f"slide-{n}.png").convert("RGB")
            assert ImageChops.difference(previous, source).getbbox() is None, n

    handbook_text = handbook.read_text(encoding="utf-8")
    assert version in handbook_text and "### 20." in handbook_text
    if version == "v18":
        assert "陈雍杰负责UI设计" in handbook_text and "郭千弘负责PPT制作" in handbook_text
    else:
        assert "陈雍杰负责离线与云端系统UI设计" in handbook_text and "郭千弘负责大纲整理和PPT制作" in handbook_text
    assert "- [x] 已准备：" in handbook_text
    return {
        "schema": "competition-defense-delivery-v1",
        "scope": "local_only",
        "status": "pass",
        "slide_count": 15,
        "speaker_note_count": 15,
        "pdf_page_count": 15,
        "suggested_speech_seconds": 560,
        "qa_count": 20,
        "version": version,
        "changed_slides": [4] if version == "v18" else [3, 4, 5, 6, 7, 8, 9, 10, 14, 15],
        "unchanged_slide_count": 14 if version == "v18" else 5,
        "preserved_user_cover_and_team_edits": version != "v18",
        "source_snapshot_sha256": sha256(BUILD / "source_v18_snapshot.pptx") if version != "v18" else None,
        "grid_chart_points_verified": 0 if version == "v18" else 64,
        "reference_spectrum_points_verified": 81,
        "mapping_preview_cells_verified": 48,
        "public_material_identity_text_findings": forbidden,
        "pdf_max_mean_channel_difference": max(differences),
        "native_powerpoint_open_verified": False,
        "pdf_fixed_layout_from_rendered_slides": True,
        "video_recorded": False,
        "cloud_updated": False,
        "github_updated": False,
        "files": [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in [pptx, pdf, handbook]],
        "source_data_sha256": sha256(ROOT / "competition" / "tio2_air_reference_records_v1.jsonl"),
        "limitations": ["No new simulation or training", "No new physical accuracy claim", "MP4 and submission receipts pending", "Official team number and QQ registration sequence pending"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v21")
    version = parser.parse_args().version
    report = verify(version)
    report_path = BUILD / f"delivery_{version}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
