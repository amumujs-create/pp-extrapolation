#!/usr/bin/env python3
"""Create a 20-minute talk by selecting slides from the Final detailed deck."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Pt


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "ppt" / "PP-X_Final_Research_Detailed_v2.pptx"
OUTPUT = ROOT / "ppt" / "PP-X_Final_Research_20min.pptx"

# Narrative: why extrapolation → PP-X/PAE route → Final algorithm → results
# → fair comparison → complete inference explanation → Q&A.
KEEP = (1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 14, 24, 43, 44, 45, 46, 47, 40)


def remove_slide(prs: Presentation, zero_based_index: int) -> None:
    slide_id = prs.slides._sldIdLst[zero_based_index]
    relationship_id = slide_id.rId
    prs.part.drop_rel(relationship_id)
    del prs.slides._sldIdLst[zero_based_index]


def replace_text(slide, old: str, new: str) -> None:
    for shape in slide.shapes:
        if not hasattr(shape, "text_frame"):
            continue
        for paragraph in shape.text_frame.paragraphs:
            for run in paragraph.runs:
                if old in run.text:
                    run.text = run.text.replace(old, new)


def build() -> None:
    shutil.copy2(SOURCE, OUTPUT)
    prs = Presentation(OUTPUT)
    selected = set(KEEP)
    for index in range(len(prs.slides), 0, -1):
        if index not in selected:
            remove_slide(prs, index - 1)

    total = len(prs.slides)
    if total != len(KEEP):
        raise RuntimeError(f"expected {len(KEEP)} slides, got {total}")
    retained_ids = list(prs.slides._sldIdLst)
    by_source_index = dict(zip(sorted(KEEP), retained_ids))
    for slide_id in retained_ids:
        prs.slides._sldIdLst.remove(slide_id)
    for source_index in KEEP:
        prs.slides._sldIdLst.append(by_source_index[source_index])

    # Make the copied cover and method wording specific to this talk.
    replace_text(prs.slides[0], "전체 연구 지도", "20분 요약 발표")
    for slide in prs.slides:
        replace_text(
            slide,
            "route가 승인되지 않으면 해당 contract에 미리 적은 direct·persistence fallback 또는 abstention으로 간다.",
            "Final에서 executor가 승인되지 않으면 contract에 미리 적은 direct·persistence fallback으로 간다.",
        )

    for page, slide in enumerate(prs.slides, 1):
        found = False
        for shape in slide.shapes:
            if not hasattr(shape, "text_frame"):
                continue
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    if re.fullmatch(r"\d+/\d+", run.text.strip()):
                        run.text = f"{page}/{total}"
                        found = True
        # Appended statistics slides can store footer text without a separate
        # run. Add the same restrained footer number when none was detected.
        if page not in (1, total) and not found:
            shape = slide.shapes.add_textbox(1140 * 9525, 674 * 9525, 88 * 9525, 18 * 9525)
            paragraph = shape.text_frame.paragraphs[0]
            paragraph.alignment = PP_ALIGN.RIGHT
            run = paragraph.add_run()
            run.text = f"{page}/{total}"
            run.font.name = "Arial"
            run.font.size = Pt(9)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)

    prs.save(OUTPUT)
    print(f"Saved {OUTPUT} ({total} slides)")


if __name__ == "__main__":
    build()
