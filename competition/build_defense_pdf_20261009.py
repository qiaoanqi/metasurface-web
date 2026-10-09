from pathlib import Path

from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
SLIDES = ROOT / ".defense-build-20261009" / "final_slides_v16"
OUTPUT = ROOT / "competition" / "答辩材料_20261009" / "AI超表面结构色智能设计系统_答辩修订版_v16.pdf"
WIDTH, HEIGHT = 960, 540


def build_pdf() -> None:
    slide_paths = sorted(SLIDES.glob("slide-*.png"), key=lambda p: int(p.stem.split("-")[-1]))
    if len(slide_paths) != 15:
        raise RuntimeError(f"expected 15 rendered slides, found {len(slide_paths)}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(OUTPUT), pagesize=(WIDTH, HEIGHT), pageCompression=1)
    pdf.setTitle("AI超表面结构色智能设计系统 - 答辩修订版")
    pdf.setAuthor("404 Not Found队")
    for slide_path in slide_paths:
        pdf.drawImage(ImageReader(str(slide_path)), 0, 0, width=WIDTH, height=HEIGHT, preserveAspectRatio=True, mask="auto")
        pdf.showPage()
    pdf.save()


if __name__ == "__main__":
    build_pdf()
