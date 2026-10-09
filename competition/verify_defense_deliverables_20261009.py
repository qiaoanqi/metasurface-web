import hashlib
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from PIL import Image, ImageChops, ImageStat
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "competition" / "答辩材料_20261009"
BUILD = ROOT / ".defense-build-20261009"
STEM = "AI超表面结构色智能设计系统_答辩修订版_v16"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest().upper()


def verify() -> dict:
    pptx = OUTPUT / f"{STEM}.pptx"
    pdf = OUTPUT / f"{STEM}.pdf"
    handbook = OUTPUT / "答辩准备手册_v1.md"
    validation = json.loads((BUILD / "validation_v16.json").read_text(encoding="utf-8"))
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
        forbidden = [s for s in ["长沙理工大学", "人工智能学院", "指导教师", "指导老师"] if s in combined]
        assert not forbidden, forbidden
        for member in ["乔安琪", "陈雍杰", "郭千弘"]:
            assert member in combined
        team_slide = ElementTree.fromstring(package.read("ppt/slides/slide2.xml"))
        namespace = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
        team_tables = team_slide.findall(".//a:tbl", namespace)
        assert len(team_tables) == 1
        team_rows = [["".join(cell.itertext()) for cell in row.findall("a:tc", namespace)] for row in team_tables[0].findall("a:tr", namespace)]
        assert team_rows == [["成员", "分工"], ["乔安琪", "项目负责人、核心算法、系统开发与主讲"], ["陈雍杰", "UI设计"], ["郭千弘", "PPT制作"]], team_rows
        for stale in ["排练计时、材料核对", "演示备份、会议调试", "工作阶段", "核心技术贡献由项目负责人承担"]:
            assert stale not in combined, stale

    reader = PdfReader(pdf)
    assert len(reader.pages) == 15
    assert all(abs(float(page.mediabox.width) - 960) < 0.01 and abs(float(page.mediabox.height) - 540) < 0.01 for page in reader.pages)
    differences = []
    for n in range(1, 16):
        source = Image.open(BUILD / "final_slides_v16" / f"slide-{n}.png").convert("RGB")
        rendered = Image.open(BUILD / "pdf_render_v16" / f"slide-{n:02}.png").convert("RGB")
        assert source.size == rendered.size == (1280, 720)
        difference = ImageStat.Stat(ImageChops.difference(source, rendered)).mean
        differences.append(max(difference))
        # PDF rasterizers apply font/image antialiasing differently from the
        # source PNG renderer. The hard checks are page count, dimensions and
        # visual inspection; this threshold catches accidental scaling rather
        # than normal cross-renderer antialiasing drift.
        assert max(difference) < 5.0, (n, difference)

    handbook_text = handbook.read_text(encoding="utf-8")
    assert "v16" in handbook_text and "### 20." in handbook_text
    assert "陈雍杰负责UI设计" in handbook_text and "郭千弘负责PPT制作" in handbook_text
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
    report = verify()
    report_path = BUILD / "delivery_v16.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
