from __future__ import annotations

from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "test_product_launch_brief.txt"
OUTPUT = ROOT / "examples" / "test_product_launch_brief.pdf"


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def wrapped_lines(text: str, width: int = 92) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines():
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if len(candidate) <= width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def build_pdf() -> None:
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)
    lines = wrapped_lines(SOURCE.read_text(encoding="utf-8"))

    for start in range(0, len(lines), 48):
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_reference})}
        )
        commands = ["BT", "/F1 10 Tf", "12 TL", "54 742 Td"]
        for index, line in enumerate(lines[start : start + 48]):
            if index:
                commands.append("T*")
            commands.append(f"({pdf_escape(line)}) Tj")
        commands.append("ET")
        stream = DecodedStreamObject()
        stream.set_data("\n".join(commands).encode("latin-1"))
        page[NameObject("/Contents")] = writer._add_object(stream)

    with OUTPUT.open("wb") as target:
        writer.write(target)


if __name__ == "__main__":
    build_pdf()
    print(OUTPUT)
