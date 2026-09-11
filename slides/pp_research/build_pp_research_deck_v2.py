#!/usr/bin/env python3
"""PP-X briefing — data charts (PNG) + native PPT text/tables only.

No boxed diagram images. Journal-style plots from numbers only.
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Pt

OUT = Path("/Users/baghyeongbae/Desktop/연구/pp-extrapolation/ppt/PP-X_Research_Detailed_v2.pptx")
ASSETS = Path("/Users/baghyeongbae/Desktop/연구/ppt/pp/_build")
FIGS = ASSETS / "figs"

W, H = 12192000, 6858000
PX = 9525


def px(n: float) -> int:
    return int(n * PX)


C = {
    "ink": RGBColor(0x22, 0x22, 0x22),
    "muted": RGBColor(0x66, 0x66, 0x66),
    "rule": RGBColor(0xE8, 0xE8, 0xE8),
    "blue": RGBColor(0x00, 0x72, 0xB2),
    "orange": RGBColor(0xE6, 0x9F, 0x00),
    "soft": RGBColor(0xF5, 0xF5, 0xF5),
    "soft_blue": RGBColor(0xE8, 0xF1, 0xF8),
    "soft_orange": RGBColor(0xFD, 0xF4, 0xE3),
    "white": RGBColor(0xFF, 0xFF, 0xFF),
    "red": RGBColor(0x9B, 0x1C, 0x1C),
}


def set_run(run, size=18, color=None, bold=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = "Arial"
    run.font.color.rgb = color or C["ink"]
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        from lxml import etree

        ea = etree.SubElement(rPr, qn("a:ea"))
    ea.set("typeface", "AppleGothic")


def add_text(slide, left, top, width, height, text, size=18, color=None, bold=False, align="left"):
    box = slide.shapes.add_textbox(px(left), px(top), px(width), px(height))
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(str(text).split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}[align]
        run = p.add_run()
        run.text = line
        set_run(run, size=size, color=color or C["ink"], bold=bold)
    return box


def rect(slide, left, top, width, height, fill=None, line=None, rounded=False):
    st = MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if rounded else MSO_AUTO_SHAPE_TYPE.RECTANGLE
    sh = slide.shapes.add_shape(st, px(left), px(top), px(width), px(height))
    sh.line.fill.background()
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is not None:
        sh.line.color.rgb = line
        sh.line.width = Pt(1.25)
    if rounded:
        try:
            sh.adjustments[0] = 0.12
        except Exception:
            pass
    return sh


def down_arrow(slide, left, top, width, height, fill):
    sh = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.DOWN_ARROW, px(left), px(top), px(width), px(height))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    return sh


def right_arrow(slide, left, top, width, height, fill):
    sh = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RIGHT_ARROW, px(left), px(top), px(width), px(height))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    return sh


def vline(slide, x, y1, y2, color):
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, px(x), px(y1), px(x), px(y2))
    line.line.color.rgb = color
    line.line.width = Pt(1.25)
    return line


def hline(slide, x1, x2, y, color):
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, px(x1), px(y), px(x2), px(y))
    line.line.color.rgb = color
    line.line.width = Pt(1.25)
    return line


def fill_shape_text(sh, text, size=14, color=None, bold=False, align="center"):
    tf = sh.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.clear()
    p.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}[align]
    run = p.add_run()
    run.text = text
    set_run(run, size=size, color=color or C["ink"], bold=bold)
    tf.auto_size = None
    sh.word_wrap = True


def pic(slide, name, left, top, width, height):
    path = FIGS / name
    if not path.exists():
        add_text(slide, left, top + 20, width, 24, f"missing: {name}", 12, C["muted"])
        return False
    slide.shapes.add_picture(str(path), px(left), px(top), px(width), px(height))
    return True


def add_table(slide, left, top, width, height, headers, rows, *, font_size=11, highlight_last=False, red_cols=None):
    red_cols = red_cols or set()
    table = slide.shapes.add_table(1 + len(rows), len(headers), px(left), px(top), px(width), px(height)).table
    for j in range(len(headers)):
        table.columns[j].width = px(width / len(headers))

    def cell(r, c, text, *, fill=None, color=None, bold=False, align="center"):
        cl = table.cell(r, c)
        cl.text = ""
        p = cl.text_frame.paragraphs[0]
        p.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER}[align]
        run = p.add_run()
        run.text = str(text)
        set_run(run, size=font_size, color=color or C["ink"], bold=bold)
        cl.vertical_anchor = MSO_ANCHOR.MIDDLE
        if fill is not None:
            cl.fill.solid()
            cl.fill.fore_color.rgb = fill
        else:
            cl.fill.background()

    for j, h in enumerate(headers):
        cell(0, j, h, fill=C["ink"], color=C["white"], bold=True)
    for i, row in enumerate(rows):
        last = highlight_last and i == len(rows) - 1
        bg = C["soft_blue"] if last else (C["soft"] if i % 2 else C["white"])
        for j, val in enumerate(row):
            col = C["blue"] if last else (C["red"] if j in red_cols else C["ink"])
            cell(i + 1, j, val, fill=bg, color=col, bold=last or j in red_cols or j == 0, align="left" if j == 0 else "center")
    return table


def head(slide, title, sub=""):
    rect(slide, 0, 0, 1280, 3, C["blue"])
    add_text(slide, 48, 22, 1180, 32, title, 22, C["ink"], True)
    if sub:
        add_text(slide, 48, 54, 1180, 20, sub, 12, C["muted"])


def foot(slide, n, total=23):
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, px(48), px(680), px(1232), px(680))
    line.line.color.rgb = C["rule"]
    line.line.width = Pt(0.75)
    add_text(slide, 48, 686, 800, 14, "SPS Lab  ·  Prior-Adaptive Extrapolation  ·  PP-X", 9, C["muted"])
    add_text(slide, 1100, 684, 100, 16, f"{n}/{total}", 10, C["muted"], True, "right")


def blank(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, 0, 0, W, H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = C["white"]
    bg.line.fill.background()
    return s


def build():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    n = 0
    TOTAL = 38

    def p():
        nonlocal n
        n += 1
        return n

    # 1 Cover
    s = blank(prs)
    for sh in list(s.shapes):
        sh._element.getparent().remove(sh._element)
    bg = s.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, 0, 0, W, H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = C["white"]
    bg.line.fill.background()
    frame = s.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, px(40), px(32), px(1200), px(656))
    frame.fill.background()
    frame.line.color.rgb = RGBColor(0x22, 0x22, 0x22)
    frame.line.width = Pt(1.25)
    logo = ASSETS / "sps_lab_logo.png"
    if logo.exists():
        s.shapes.add_picture(str(logo), px(860), px(58), px(320), px(64))
    add_text(s, 80, 220, 1120, 40, "Prior-Adaptive Extrapolation", 28, C["ink"], True, "center")
    add_text(s, 80, 268, 1120, 36, "for Robust Prediction Beyond Observed Support", 20, C["ink"], True, "center")
    add_text(s, 80, 330, 1120, 28, "관측된 support 밖에서의 강건 예측", 15, C["muted"], False, "center")
    add_text(s, 80, 390, 1120, 28, "논문 메인 모델  ·  PP-X", 16, C["blue"], True, "center")
    add_text(s, 80, 470, 1120, 24, "Smart Production Systems Lab.  ·  박사과정 박진서", 14, C["ink"], False, "center")
    add_text(s, 80, 520, 1120, 22, "2026.09.11", 13, C["muted"], False, "center")
    p()

    # 2 Problem + background
    s = blank(prs)
    head(s, "문제 · 연구 배경", "밖에서는 데이터가 답을 정하지 못한다.  정당화되는 prior에 맞춰 고른다.")
    pic(s, "nn_vs_saar_curves.png", 36, 86, 640, 400)
    add_text(s, 36, 492, 640, 36, "점선 왼쪽은 학습, 오른쪽은 그 밖. 제약 없는 NN은 열화 중에도 예측이 되살아난다.", 12, C["muted"])

    rect(s, 696, 86, 548, 132, C["soft"], C["rule"], True)
    add_text(s, 712, 94, 516, 22, "왜 외삽인가", 13, C["ink"], True)
    add_text(s, 712, 118, 516, 90, "학습 분포 안에서는 잘 맞혀도, 관측되지 않은 영역에서는 성능이 급히 떨어질 수 있다. 열화·RUL·피로균열은 실제 시점이 학습 support 밖인 경우가 많아, 보간보다 밖에서의 예측이 중요하다.", 12, C["ink"])

    rect(s, 696, 230, 548, 132, C["soft_orange"], C["orange"], True)
    add_text(s, 712, 238, 516, 22, "prior는 일정하지 않다", 13, C["orange"], True)
    add_text(s, 712, 262, 516, 90, "같은 데이터라도 prior가 바뀌면 밖으로 나가는 곡선이 달라진다. 어떤 문제는 방향·경계·추세만 알고, 어떤 문제는 식과 적용 조건까지 있다. 맞지 않는 prior를 강제하면 외삽이 나빠진다.", 12, C["ink"])

    rect(s, 696, 374, 548, 154, C["soft_blue"], C["blue"], True)
    add_text(s, 712, 382, 516, 22, "출발점", 13, C["blue"], True)
    add_text(s, 712, 408, 516, 108, "맞는 답이 하나가 아니다. 밖을 지탱하는 것은 데이터가 아니라 지금 정당화되는 prior다. prior가 많을수록 좋은 것이 아니라, 그 수준과 신뢰성에 맞춰 외삽 전략을 고른다. 다음 장에서 경로를 나눈다.", 12, C["ink"])
    foot(s, p(), TOTAL)

    # 3 Prior work — what existing extrapolation methods assume
    s = blank(prs)
    head(s, "사전 조사 — 회귀 외삽은 추가 가정 없이는 식별되지 않는다",
         "대표 방법은 서로 다른 구조·분포 가정을 둔다. PP-X만 prior를 쓰는 것은 아니다.")

    add_text(s, 48, 88, 1184, 34,
             "조사 결론  범용 회귀기가 데이터만으로 support 밖 함수를 정해 주지는 않는다.", 16, C["ink"], True)

    cards = [
        (48, "제약 없는 NN", "ReLU 네트워크", "마지막 선형 조각을\n밖으로 연장", "가정이 암묵적이라\ntail 형태를 통제하기 어렵다", C["soft"], C["ink"]),
        (350, "형상 제약", "CMNN · monotone NN", "증가/감소 방향을\n구조로 강제", "단조 방향이 맞다는\n도메인 prior가 필요하다", C["soft_blue"], C["blue"]),
        (652, "식·물리 제약", "EQL · PINN · Physics-ML", "함수식·PDE·열화 법칙을\n구조 또는 loss에 반영", "식과 적용 조건이 맞는\n도메인에서 강하다", C["soft_orange"], C["orange"]),
        (954, "통계적 외삽", "Engression · Xtrapolation\n· Progression", "noise 위치·미분 경계·\ntail dependence를 가정", "가정이 약하면 band가 넓고\n점예측은 식별되지 않는다", C["soft"], C["ink"]),
    ]
    for x, title, family, mechanism, limit, fill, line in cards:
        rect(s, x, 142, 278, 300, fill, line, True)
        add_text(s, x + 18, 158, 242, 26, title, 16, line, True, "center")
        add_text(s, x + 18, 198, 242, 22, family, 11, C["muted"], True, "center")
        hline(s, x + 24, x + 254, 236, C["rule"])
        add_text(s, x + 18, 254, 242, 58, mechanism, 13, C["ink"], False, "center")
        add_text(s, x + 18, 346, 242, 64, limit, 12, C["red"], False, "center")

    rect(s, 48, 470, 1184, 104, C["ink"], None, True)
    add_text(s, 70, 482, 1140, 28,
             "직접 인접 연구  EV(2024)·LBO(2025)는 boundary-focused validation을 이미 제안했다.", 13, C["white"], True, "center")
    add_text(s, 70, 516, 1140, 44,
             "남은 질문  이질적인 prior를 outcome-free contract로 제한하고, physical-unit 위험으로 executor와 exact fallback을 시험 전에 함께 고정할 수 있는가?", 13, C["white"], True, "center")
    add_text(s, 48, 604, 1184, 32,
             "Xu 2021 · Runje 2023 · Raissi 2019 · Shen & Meinshausen 2024 · Pfister & Bühlmann 2026 · Yu et al. 2024 · García et al. 2025",
             9, C["muted"], False, "center")
    foot(s, p(), TOTAL)

    # 4 Contributions — position the integration claim before the algorithm
    s = blank(prs)
    head(s, "연구 기여 — 새 블록 하나보다 ‘가정을 실행하는 규칙’을 제안한다",
         "개별 요소의 최초성이 아니라, 외삽 가정의 선언·학습·승인·거절을 하나의 검증 가능한 절차로 결합한다.")
    contributions = [
        (48, "C1  Typed contract", "test 결과를 보기 전에\n경계 · unit · causal X ·\n허용 prior · fallback 선언", C["soft_blue"], C["blue"]),
        (350, "C2  Prior-residual core", "동결된 저복잡도 tail 주변에서\nsource가 지지하는 nonlinear\nresidual만 제한적으로 학습", C["soft"], C["ink"]),
        (652, "C3  Unit-evidence approval", "group-disjoint validation의\n이득 · unit wins · worst risk로\nexecutor 승인 또는 거절", C["soft_orange"], C["orange"]),
        (954, "C4  Frozen execution", "승인 route 하나를 test 전에 고정\n근거가 없으면 exact fallback\n성공·실패 결과를 함께 보고", C["soft"], C["ink"]),
    ]
    for x, title, body, fill, line in contributions:
        rect(s, x, 118, 278, 314, fill, line, True)
        add_text(s, x + 18, 142, 242, 30, title, 15, line, True, "center")
        hline(s, x + 24, x + 254, 192, C["rule"])
        add_text(s, x + 18, 224, 242, 138, body, 13, C["ink"], False, "center")
    rect(s, 48, 458, 1184, 120, C["ink"], None, True)
    add_text(s, 70, 472, 1140, 34,
             "결합 방식  Contract C → 허용 후보 E(C) 제한 → prior-residual 후보 학습 → unit-risk 제약 아래 validation-best executor 선택 → frozen route / fallback",
             13, C["white"], True, "center")
    add_text(s, 70, 518, 1140, 42,
             "차별점  prior+residual·validation·abstention을 나열한 것이 아니라, 앞 단계의 출력이 다음 단계의 허용 입력을 제한하는 하나의 실행 알고리즘으로 만든다.",
             13, C["white"], False, "center")
    add_text(s, 48, 602, 1184, 30,
             "실증  9개 retrospective setting 중 동일예산 최강 비교모델 대비 8개 우세 · DS03에서는 prior 거절 성공, Engression보다 정확도 우월은 미확증",
             11, C["muted"], False, "center")
    foot(s, p(), TOTAL)

    # 5 Why the contribution matters
    s = blank(prs)
    head(s, "그래서 외삽에서 무엇이 좋아졌는가",
         "현재 retrospective 데이터 안에서 확인된 효과와 아직 주장할 수 없는 범위를 분리한다.")
    add_text(s, 48, 82, 1184, 36,
             "비교 대상  하나의 prior나 optional module을 모든 데이터에 항상 켜는 외삽 모델",
             15, C["ink"], True)

    uses = [
        (48, "① 평균 예측오차", "각 contract에서 validation이 지지한\nexecutor만 선택해, 맞는 구조의\n외삽 이득은 유지", "개발 결과\n동일예산 최강 비교군 대비 8/9 우세\npaired GM-RMSE 33.8% 감소", C["soft_blue"], C["blue"]),
        (350, "② 전역 적용 붕괴 방지", "모든 셋에 같은 bound·scale을\n항상 켜지 않고, 해당 contract에서\n지지된 executor만 최종 route에 포함", "직접 반례\nMICH fixed bound ΔR² −.291\nRWTH dual-scale ΔR² −.037", C["soft"], C["ink"]),
        (652, "③ 유닛 단위 안정성", "평균 RMSE만 보지 않고 unit wins와\nworst-unit ratio를 승인 조건에 포함해\n일부 설비에 손해가 몰리는 route 제한", "직접 근거\nHUST transport 15/16 unit 개선\nMICH dual-scale 7/8 unit 개선", C["soft_orange"], C["orange"]),
        (954, "④ 잘못된 승인 감소", "12-domain common-backbone audit에서\nunit-gain CI 조건을 추가해\n근거가 약한 prior 승인을 걸러냄", "retrospective policy audit\nfalse accept 2 → 0\n선택 정확도 .667 → .833", C["soft"], C["ink"]),
    ]
    for x, title, action, value, fill, line in uses:
        rect(s, x, 142, 278, 344, fill, line, True)
        add_text(s, x + 18, 160, 242, 30, title, 15, line, True, "center")
        add_text(s, x + 18, 216, 242, 104, action, 12, C["ink"], False, "center")
        hline(s, x + 24, x + 254, 340, C["rule"])
        add_text(s, x + 18, 366, 242, 76, value, 12, line, True, "center")

    rect(s, 48, 520, 1184, 82, C["ink"], None, True)
    add_text(s, 70, 532, 1140, 56,
             "기여 효과  현재 평가 데이터셋에서 PP-X는 validation이 지지한 구조만 실행해 전역 prior/executor의 성능 붕괴를 줄이고, 평균 오차와 unit-level risk를 함께 관리했다.",
             14, C["white"], True, "center")
    add_text(s, 48, 612, 1184, 22,
             "범위  위 수치는 retrospective evidence다. 보지 않은 미래 cohort에서 같은 실패 감소를 보장한다는 주장은 하지 않는다.",
             10, C["red"], True, "center")
    foot(s, p(), TOTAL)

    # 6 Evidence-backed failure analysis
    s = blank(prs)
    head(s, "잘 안 된 경우 — 왜 실패했는가",
         "사후 추측이 아니라 matched ablation·unit 결과·validation/test 불일치로 확인된 원인만 남긴다.")

    failures = [
        (48, 98, 570, 216, "① 제약이 실제 shift와 맞지 않음",
         "MICH  fixed bound가 residual 용량을 과도하게 제한\nR² .759(unbounded) → .468(fixed), Δ −.291\n\nRWTH  dual-scale은 8/8 unit에서 악화\nR² .878 → .842, BH q=.022",
         "교훈  좋은 제약도 전역 default로 켜면 negative transfer", C["soft_orange"], C["orange"]),
        (662, 98, 570, 216, "② validation evidence가 test로 이동하지 않음",
         "단순 validation PP/MLP 선택은 12개 중 7개만 정답\nfalse accept 4개\n\nCI 강화 정책도 retrospective 정확도 .833\n즉 validation 점수만으로 보편적 전이를 보장하지 못함",
         "교훈  unit risk와 CI가 필요하지만 미래 보장은 아님", C["soft"], C["ink"]),
        (48, 338, 570, 216, "③ prior 거절 뒤 fallback 표현력이 부족",
         "DS03  prior route 거절 자체는 test-best PP-X route와 일치\n하지만 Engression 대비 ΔR² −.0195\nunit wins 1/6 · worst ratio 1.989",
         "교훈  안전한 선택과 최고 정확도는 서로 다른 문제", C["soft_blue"], C["blue"]),
        (662, 338, 570, 216, "④ 구조를 학습할 정보가 부족",
         "Virkler late-tail은 test specimen당 2 rows\nResidual-state·temporal projection은 val에서 off 선택\n\nPrior-geometry는 val 승인 후 test RMSE +31.3%\nworst-unit ratio 5.82",
         "교훈  짧은 tail에서 복잡한 trajectory head는 과적합 위험", C["soft"], C["ink"]),
    ]
    for x, y, w, h, title, evidence, lesson, fill, line in failures:
        rect(s, x, y, w, h, fill, line, True)
        add_text(s, x + 18, y + 14, w - 36, 26, title, 14, line, True)
        add_text(s, x + 18, y + 52, w - 36, 104, evidence, 11, C["ink"])
        hline(s, x + 18, x + w - 18, y + 166, C["rule"])
        add_text(s, x + 18, y + 176, w - 36, 30, lesson, 11, line, True)

    add_text(s, 48, 586, 1184, 32,
             "정리  PP-X가 실패를 없앤 것이 아니다. 현재 증거는 ‘어떤 구조가 왜 무너졌는지 식별하고, 해로운 route를 전역 적용하지 않는 것’까지 지지한다.",
             12, C["red"], True, "center")
    foot(s, p(), TOTAL)

    # 7 Research route — full diagram
    s = blank(prs)
    head(s, "연구 루트", "그래서 후보식이 정당화되는지에 따라 경로를 나눈다")

    chips = [("관측", 200), ("경계", 510), ("도메인 지식", 820)]
    for label, x in chips:
        ch = rect(s, x, 92, 260, 40, C["soft"], C["ink"], True)
        fill_shape_text(ch, label, 14, C["ink"], True)
        vline(s, x + 130, 132, 156, C["rule"])
    hline(s, 330, 950, 156, C["rule"])
    vline(s, 640, 156, 170, C["rule"])
    down_arrow(s, 624, 168, 32, 12, C["muted"])

    dec = rect(s, 310, 184, 660, 48, C["ink"], None, True)
    fill_shape_text(dec, "후보식이 정당화되는가?", 17, C["white"], True)

    vline(s, 340, 232, 248, C["rule"])
    vline(s, 940, 232, 248, C["rule"])
    hline(s, 340, 940, 248, C["rule"])
    down_arrow(s, 324, 248, 32, 16, C["blue"])
    down_arrow(s, 924, 248, 32, 16, C["orange"])
    add_text(s, 70, 228, 120, 20, "NO", 12, C["blue"], True, "right")
    add_text(s, 1090, 228, 120, 20, "YES", 12, C["orange"], True, "left")

    rect(s, 70, 272, 540, 200, C["soft_blue"], C["blue"], True)
    rect(s, 70, 272, 8, 200, C["blue"])
    badge = rect(s, 456, 286, 130, 24, C["blue"], None, True)
    fill_shape_text(badge, "이번 발표", 10, C["white"], True)
    add_text(s, 98, 284, 340, 32, "PP-X", 24, C["blue"], True)
    add_text(s, 98, 322, 480, 20, "equation-free", 12, C["muted"])
    add_text(s, 98, 352, 490, 22, "prior-residual core", 15, C["ink"])
    add_text(s, 98, 384, 490, 22, "+ evidence-selected executor", 15, C["ink"])
    add_text(s, 98, 426, 490, 22, "논문 동결 정의  ·  Algorithm 1", 14, C["blue"], True)

    rect(s, 670, 272, 540, 200, C["soft_orange"], C["orange"], True)
    rect(s, 670, 272, 8, 200, C["orange"])
    nxt = rect(s, 1056, 286, 130, 24, C["orange"], None, True)
    fill_shape_text(nxt, "다음 논문", 10, C["white"], True)
    add_text(s, 698, 284, 340, 32, "PAE", 24, C["orange"], True)
    add_text(s, 698, 322, 480, 20, "equation-aware", 12, C["muted"])
    add_text(s, 698, 352, 490, 22, "허용된 식 + 제한 NN", 15, C["ink"])
    add_text(s, 698, 384, 490, 22, "이득 없으면 PP-X로 되돌림", 15, C["ink"])
    add_text(s, 698, 426, 490, 22, "LLM · 온톨로지 · source gate", 14, C["orange"], True)

    vline(s, 340, 472, 498, C["rule"])
    vline(s, 940, 472, 498, C["rule"])
    hline(s, 340, 940, 498, C["rule"])
    down_arrow(s, 624, 498, 32, 12, C["muted"])

    end = rect(s, 220, 518, 840, 44, C["soft"], C["ink"], True)
    fill_shape_text(end, "Assurance    ·    믿기  /  보류  /  거절    →    박사논문에서 두 경로 통합", 14, C["ink"], True)
    add_text(s, 70, 578, 1140, 24, "오늘은 왼쪽만 간다.  PAE와 Assurance는 지도에만 찍는다.", 13, C["muted"], False, "center")
    foot(s, p(), TOTAL)

    # 6 Method — frozen paper PP-X
    s = blank(prs)
    head(s, "방법 — 최종 PP-X", "작은 공통 prior-residual core + validation-approved executor + prespecified fallback")
    pic(s, "ppx_core.png", 20, 78, 760, 400)
    add_table(
        s,
        790,
        86,
        448,
        248,
        ["단계", "최종 PP-X가 하는 일"],
        [
            ["(a) contract", "경계·unit·causal X·허용 prior·fallback 선언"],
            ["(b) core", "동결 affine/quotient tail + bounded nonlinear residual"],
            ["(c) executor", "bound·history·transport 중 validation이 지지한 것만 실행"],
        ],
        font_size=11,
    )
    add_table(
        s,
        790,
        348,
        448,
        200,
        ["승인 규칙 (val/source만)", "조건"],
        [
            ["Prior admissibility", "source OOF regret·complete groups·regime coverage"],
            ["Executor gain", "validation RMSE 상대 2% 이상 개선"],
            ["Unit risk", "unit wins ≥60% · worst ratio ≤1.10"],
            ["거절", "사전 지정 direct/persistence fallback 또는 abstention"],
        ],
        font_size=11,
    )
    add_text(
        s,
        20,
        488,
        760,
        88,
        "모든 데이터셋에서 같은 optional module을 켜지 않는다. 공통점은 prior와 residual의 역할을 분리하고, contract가 허용한 executor만 validation에서 승인한다.\n"
        "route가 승인되지 않으면 해당 contract에 미리 적은 direct·persistence fallback 또는 abstention으로 간다.",
        12,
        C["ink"],
    )
    add_text(s, 20, 600, 1210, 22, "버전 번호 혼동을 피하기 위해 논문과 본 발표에서는 ‘PP-X Algorithm 1’로만 표기한다.", 12, C["muted"], True)
    foot(s, p(), TOTAL)

    # 7 Frozen paper algorithm
    s = blank(prs)
    head(s, "PP-X Algorithm 1", "outcome-free contract → validation approval → frozen execution")
    stages = [
        (48, "① Typed contract", "boundary · progression · history\nregime · support\n후보 executor 제한", C["soft_blue"], C["blue"]),
        (350, "② Prior admissibility", "complete source groups\nOOF regret ≤ 0\nmode stability ≥ .60", C["soft"], C["ink"]),
        (652, "③ Executor approval", "validation gain ≥ 2%\nunit wins ≥ 60%\nworst ratio ≤ 1.10", C["soft_orange"], C["orange"]),
        (954, "④ Frozen output", "승인 executor 1개\n또는 prespecified fallback\ntest에서 route 불변", C["soft_blue"], C["blue"]),
    ]
    for x, title, body, fill, line in stages:
        rect(s, x, 130, 278, 280, fill, line, True)
        add_text(s, x + 16, 146, 246, 28, title, 15, line, True, "center")
        add_text(s, x + 16, 210, 246, 130, body, 13, C["ink"], False, "center")
    for x in (326, 628, 930):
        add_text(s, x, 250, 24, 32, "→", 22, C["muted"], True, "center")
    rect(s, 48, 450, 1184, 86, C["ink"], None, True)
    add_text(s, 68, 468, 1144, 48, "논문 메인 = PP-X 선택 알고리즘  ·  CCMR은 trajectory risk-aware executor 사례  ·  12개 과거 route가 하나의 동일 NN이라는 뜻은 아님", 14, C["white"], True, "center")
    add_text(s, 48, 566, 1184, 48, "동결  protocols/PPX_PAPER_METHOD_V1_FROZEN_PROTOCOL.md  ·  구현  paper_ppx.py  ·  선택 함수는 test outcome 인자를 받지 않음", 12, C["muted"], True)
    foot(s, p(), TOTAL)

    # 5 How the evaluation interval is defined
    s = blank(prs)
    head(s, "외삽 구간을 어떻게 정하는가", "학습이 본 건강 범위보다 더 진행된 관측만 점수로 친다")
    pic(s, "extrapolation_cut.png", 36, 86, 720, 350)
    add_table(
        s,
        770,
        88,
        462,
        350,
        ["절차", "내용"],
        [
            ["1", "경계, 유닛, 시간, 입력, 외삽 좌표를 먼저 고정"],
            ["2", "학습·검증·시험에 같은 유닛을 넣지 않음"],
            ["3", "학습에는 본 적 있는 건강 구간만 사용"],
            ["4", "검증·시험에는 그 범위 밖 관측만 남김"],
            ["5", "정규화 통계는 학습 분할에서만 계산"],
            ["6", "모델과 executor는 검증에서만 선택"],
            ["7", "고정한 시험 분할은 한 번만 예측"],
        ],
        font_size=11,
    )
    add_text(
        s,
        36,
        452,
        1200,
        100,
        "세로선은 학습에서 관측한 최저 건강이다. 그보다 더 나빠진 구간(오른쪽)만 검증·시험 점수에 넣는다.\n같은 유닛의 미래 값, 최종 수명, 시험 궤적에서 만든 통계는 입력에 쓰지 않는다. 이 조건을 어기면 내삽 평가가 된다.",
        13,
        C["ink"],
    )
    add_text(s, 36, 562, 1200, 28, "자르는 규칙은 같다. 외삽 geometry는 셋마다 다르므로 다음 장에서 1D와 다차원을 구분해 적는다.", 12, C["muted"])
    foot(s, p(), TOTAL)

    # Quantification of the cut
    s = blank(prs)
    head(s, "외삽을 숫자로 확인하는 방법", "1D 열화좌표 기준이다. 전체 feature-space hull 밖이라고 말하지 않는다.")
    pic(s, "hull_distance.png", 20, 82, 520, 400)
    add_table(
        s,
        550,
        82,
        690,
        380,
        ["셋", "1D Hull", "거리", "성격"],
        [
            ["선우다", "100%", "4.42", "미지 셀 · 늦은 건강. 매우 명확"],
            ["아헨", "100%", "2.87", "미지 셀 · 늦은 건강. 매우 명확"],
            ["MIT 2019", "100%", "5.55", "1D 건강 support 밖"],
            ["미시간", "100%", "3.43", "미지 셀 · 늦은 건강"],
            ["NASA 실험셀", "100%", "2.01", "미지 셀 · 건강 tail"],
            ["화중 배터리", "100%", "1.61", "미지 유닛 · 충전법/말기. 1D 기준"],
            ["알루미늄 균열", "100%", "1.62", "미지 시편 · 균열 tail"],
            ["MIT 배치2", "100%", "1.90", "1D 100% · PCA2 0% · PCA3 99.2%"],
            ["항공기 엔진", "100%", "0.23", "1D 100% · PCA2 0% · PCA3 83.6% · 조건"],
        ],
        font_size=10,
    )
    add_text(
        s,
        20,
        498,
        1240,
        110,
        "오늘 그림에는 1D ordered-coordinate 기준의 엄격한 외삽만 남긴다. XJTU·FEMTO·NASA milling은 도메인/재료 transfer라 이 표에 없다.\n1D Hull-out 100%는 ‘선언한 건강·균열·조건 축’ 밖이지, 전체 특징공간 convex hull 밖이 아니다. 배치2와 엔진은 PCA 2D에서 0%다.\n엔진은 거리 0.23 SD·Target-out 0%라 먼 수명 외삽이 아니라 운전조건(regime) 외삽이다.",
        12,
        C["ink"],
    )
    foot(s, p(), TOTAL)

    # Protocol — after method, before scores
    s = blank(prs)
    head(s, "프로토콜", "점수는 이 범위 안에서만 읽는다  ·  적합도는 허가증이지 성적표가 아니다")
    add_table(
        s,
        48,
        100,
        1184,
        250,
        ["규칙", "의미", "깨지면"],
        [
            ["unit-disjoint", "train / val / test 셀이 겹치지 않음", "같은 유닛 누수"],
            ["hull-out", "선언한 1D 열화좌표에서 test가 train 밖", "내삽으로 바뀜"],
            ["val-only", "게이트·보정을 val에서만 고름", "test로 튜닝"],
            ["Pass / Weak / Fail", "운행 허가. Fail이면 예측을 옮기지 않음", "억지 점수"],
        ],
        font_size=13,
    )
    add_table(
        s,
        48,
        380,
        580,
        220,
        ["In scope", "예"],
        [
            ["1D late-tail", "Sun · RWTH · MICH · MATR2019 · NASA · Virkler · HUST"],
            ["1D + 다차원 설명 필요", "MATRb2 · N-CMAPSS (PCA2는 0%)"],
            ["executor 선택", "train/val · group-LOO only"],
        ],
        font_size=12,
    )
    add_table(
        s,
        652,
        380,
        580,
        220,
        ["오늘 표에 없음", "이유"],
        [
            ["전체 feature hull", "1D 100% ≠ 다차원 hull 밖"],
            ["도메인 / 재료 transfer", "조건·끝점·재질 이동은 다른 종류"],
            ["C-MAPSS FD002/004", "조건 hull은 강하나 이번 주표 밖"],
        ],
        font_size=12,
    )
    foot(s, p(), TOTAL)

    # 6 Main results — PP-X portfolio
    s = blank(prs)
    head(s, "주 결과 — 최종 PP-X portfolio", "1D 열화좌표 기준의 엄격한 외삽만.  retrospective development 결과  ·  PAE 없음")
    pic(s, "ppx_portfolio.png", 30, 82, 680, 390)
    add_table(
        s,
        720,
        88,
        512,
        384,
        ["셋", "Executor", "R²"],
        [
            ["화중 배터리", "transport", "0.958"],
            ["선우다 상용셀", "fixed BQ", "0.939"],
            ["항공기 엔진", "multiscale", "0.937"],
            ["알루미늄 균열", "gated residual", "0.888"],
            ["아헨 배터리", "fixed BQ", "0.878"],
            ["MIT 배치2", "decay+transport", "0.862"],
            ["미시간 배터리", "dual-scale", "0.751"],
            ["NASA 실험셀", "multiscale", "0.584"],
            ["MIT 2019", "cal. latent", "0.466"],
        ],
        font_size=10,
    )
    rect(s, 36, 488, 400, 168, C["soft_blue"], C["blue"], True)
    rect(s, 36, 488, 8, 168, C["blue"])
    add_text(s, 56, 498, 360, 22, "Li 배터리", 13, C["blue"], True)
    add_text(
        s,
        56,
        524,
        360,
        120,
        "화중·선우다·아헨·미시간 리튬셀\nMIT 수명 벤치 (2019 / 배치2)\nNASA 실험용 18650  (로켓 아님)",
        11,
        C["ink"],
    )
    rect(s, 452, 488, 380, 168, C["soft"], C["ink"], True)
    rect(s, 452, 488, 8, 168, C["ink"])
    add_text(s, 472, 498, 340, 22, "균열", 13, C["ink"], True)
    add_text(
        s,
        472,
        524,
        340,
        110,
        "알루미늄 판에 금이 감\n항공기 재료 피로실험\n배터리 아님",
        12,
        C["ink"],
    )
    rect(s, 848, 488, 396, 168, C["soft_orange"], C["orange"], True)
    rect(s, 848, 488, 8, 168, C["orange"])
    add_text(s, 868, 498, 356, 22, "엔진", 13, C["orange"], True)
    add_text(
        s,
        868,
        524,
        356,
        110,
        "항공기 엔진 시뮬레이터\n처음 보는 비행조건\n실제 비행기 데이터가 아님",
        12,
        C["ink"],
    )
    foot(s, p(), TOTAL)

    # Ablation — dual-scale is one executor
    s = blank(prs)
    head(s, "Ablation — dual-scale", "executor 하나  ·  모든 데이터에 켜는 기본값이 아니다")
    pic(s, "ablation_panel.png", 24, 82, 1232, 430)
    add_text(
        s,
        48,
        520,
        580,
        90,
        "(a) 결론  고정 bound는 Sun·RWTH에서는 높지만 MICH를 0.468에 묶는다. dual-scale을 켠 뒤에야 MICH가 0.751이 된다.",
        13,
        C["ink"],
    )
    add_text(
        s,
        660,
        520,
        572,
        90,
        "(b) 결론  평균을 조금 낮추고 최저 점수를 올린 선택이다. 데이터별 최고점을 모은 모델이 아니다.",
        13,
        C["ink"],
    )
    foot(s, p(), TOTAL)

    # Component ΔR²
    s = blank(prs)
    head(s, "Ablation — 구성요소", "켠 모델 − matched 제거 arm  ·  ΔR²")
    pic(s, "ablation_delta.png", 36, 88, 760, 400)
    add_table(
        s,
        810,
        88,
        430,
        400,
        ["기능", "대표", "ΔR²"],
        [
            ["Residual", "Sun", "+0.658"],
            ["Residual", "MICH", "+3.811"],
            ["Frozen affine", "MICH", "+0.149"],
            ["Fixed bound", "Sun", "+0.221"],
            ["Fixed bound", "MICH", "−0.291"],
            ["Dual-scale", "MICH", "+0.283"],
            ["Rate hist.", "RWTH", "+1.256"],
            ["Transport", "MATRb2", "+0.187"],
        ],
        font_size=11,
    )
    add_text(s, 36, 520, 760, 70, "결론  residual은 세 배터리에서 모두 이득이다. 고정 bound와 full history는 MICH에서 마이너스다. 그래서 모듈을 쌓지 않고, 근거 있는 executor만 켠다.", 12, C["ink"])
    foot(s, p(), TOTAL)

    # 6-arm + history
    s = blank(prs)
    head(s, "Ablation — 대조군 · history", "(a) matched 6-arm  ·  (b) causal history")
    pic(s, "ablation_arms.png", 20, 88, 630, 400)
    pic(s, "ablation_history.png", 650, 88, 600, 380)
    add_text(s, 48, 500, 1180, 80, "(a) 결론  Affine만, 또는 그냥 NN만으로는 부족하다. 동결 affine 위에 제한 residual을 올린 조합이 세 셋에서 가장 안정하다.\n(b) 결론  속도 이력은 Sun·RWTH에 필요하고, MICH에서는 단순한 margin history가 더 높다(0.715 vs 0.468). 이력을 전역 기본값으로 두지 않는다.", 13, C["ink"])
    foot(s, p(), TOTAL)

    # MATR 2x2 + gate
    s = blank(prs)
    head(s, "Ablation — transport · gate", "optional executor는 항상 켜지 않는다")
    pic(s, "ablation_matr.png", 40, 100, 500, 420)
    pic(s, "stats_gate.png", 560, 100, 680, 400)
    add_text(
        s,
        40,
        518,
        1200,
        80,
        "(a) 결론  MATRb2에서 transport가 주효과(+0.187)다. decay만으로는 +0.002다. 둘을 같이 켜야 0.862다.\n(b) 결론  gate는 성능을 만드는 장치가 아니라 거절 장치다. 13곳 중 3곳만 개선, 10곳은 유지, 악화 0. seed는 5/5일 때만 승인한다.",
        12,
        C["ink"],
    )
    foot(s, p(), TOTAL)

    # Ablation synthesis
    s = blank(prs)
    head(s, "Ablation 종합", "모듈을 쌓지 않는다.  검증에서 이득이 있는 executor만 켠다.")
    add_text(s, 48, 86, 1184, 28, "앞 네 장의 숫자를 한 규칙으로 읽는다. 최종 PP-X는 공통 core + 데이터마다 고른 executor다.", 14, C["ink"])

    rect(s, 48, 124, 380, 360, C["soft_blue"], C["blue"], True)
    rect(s, 48, 124, 8, 360, C["blue"])
    add_text(s, 68, 136, 340, 28, "항상 켠다", 18, C["blue"], True)
    add_text(s, 68, 172, 340, 22, "core", 12, C["muted"])
    add_text(
        s,
        68,
        202,
        340,
        260,
        "동결 affine prior\n+ 제한 residual\n\nAffine만, 또는 NN만으로는\n세 배터리에서 무너진다.\n\nResidual ΔR²\nSun +0.66  ·  MICH +3.81",
        14,
        C["ink"],
    )

    rect(s, 450, 124, 380, 360, C["soft_orange"], C["orange"], True)
    rect(s, 450, 124, 8, 360, C["orange"])
    add_text(s, 470, 136, 340, 28, "근거 있을 때만", 18, C["orange"], True)
    add_text(s, 470, 172, 340, 22, "executor  ·  val-only", 12, C["muted"])
    add_text(
        s,
        470,
        202,
        340,
        260,
        "고정 bound  Sun · RWTH\ndual-scale  MICH만 (+0.28)\n속도 이력  Sun · RWTH\ntransport  HUST · MATRb2\n\ngate  13곳 중 3곳만 승인\n악화 0",
        14,
        C["ink"],
    )

    rect(s, 852, 124, 380, 360, C["soft"], C["ink"], True)
    rect(s, 852, 124, 8, 360, C["ink"])
    add_text(s, 872, 136, 340, 28, "켜면 나빠진다", 18, C["ink"], True)
    add_text(s, 872, 172, 340, 22, "전역 기본값으로 두지 않음", 12, C["muted"])
    add_text(
        s,
        872,
        202,
        340,
        260,
        "고정 bound → MICH −0.29\nfull rate history → MICH −0.25\ndual-scale → Sun −0.005,\nRWTH −0.036\n\n평균을 조금 깎고\n최저점을 살리는 선택은\n데이터별 최고점 모음이 아니다.",
        14,
        C["ink"],
    )

    end = rect(s, 48, 504, 1184, 72, C["ink"], None, True)
    fill_shape_text(end, "읽는 법    core는 고정한다.   executor는 검증 증거가 있을 때만 켠다.   실패하면 safety로 되돌린다.", 15, C["white"], True)
    add_text(s, 48, 586, 1184, 24, "그래서 주표의 executor가 데이터마다 다르다.  한꺼번에 켠 공동 모델이 아니다.", 13, C["muted"])
    foot(s, p(), TOTAL)

    # Competitors — after internal ablation, before formal tests
    s = blank(prs)
    head(s, "비교", "(a) heatmap  ·  (b) PP-X vs TabPFN vs others")
    pic(s, "competitor_bars.png", 16, 72, 1248, 528)
    add_text(s, 40, 608, 1200, 40, "1 PP-X 0.81 · 2 GroupDRO 0.36 · 3 V-REx 0.35.  순위=9곳 평균 R².  강건=9곳에서 양수(9/9, 7/9, 7/9).  MICH TabPFN=동일 202행, v3 CPU, ensemble −1.86.", 13, C["ink"])
    foot(s, p(), TOTAL)

    # Stats then robustness — one evidence block
    s = blank(prs)
    head(s, "통계", "무엇을 검정했는가  ·  파랑 = p<0.05  ·  주황 = 유의 못 함")
    pic(s, "stats_wilcoxon.png", 16, 82, 568, 340)
    add_table(
        s,
        590,
        82,
        650,
        340,
        ["①", "검정", "질문", "결과"],
        [
            ["W", "Wilcoxon", "Sun·RWTH·MICH 25 unit RMSE", "Direct 17/25 p=.003"],
            ["W", "Wilcoxon", "soft / affine 대비", "24/25 · 25/25"],
            ["W", "Wilcoxon", "trainable / unbounded", "p=.071 · .578  못 함"],
            ["U", "Unit wins", "아홉 셋 물리 유닛 77개", "60/77 이김"],
            ["D", "Sign test", "아홉 셋 모두 우세인가", "9/9  양측 p=.0039"],
            ["B", "Bound audit", "residual이 이론 bound를 넘나", "0 / 17,645"],
        ],
        font_size=10,
    )
    rect(s, 28, 432, 300, 200, C["soft_blue"], C["blue"], True)
    add_text(s, 40, 440, 276, 20, "W  Wilcoxon", 12, C["blue"], True)
    add_text(s, 40, 464, 276, 155, "쌍을 이룬 unit RMSE.\nH0: 중앙 차이 = 0.\n파랑만 ‘이겼다’고 말함.\n주황은 이긴 칸이 있어도 유의 아님.", 11, C["ink"])
    rect(s, 340, 432, 300, 200, C["soft"], C["ink"], True)
    add_text(s, 352, 440, 276, 20, "S  Seed binomial", 12, C["ink"], True)
    add_text(s, 352, 464, 276, 155, "5 seed가 affine을 골랐는가.\nH0: 확률 ≤ 0.5.\n5/5만 α=0.05 통과.\n물리 반복이 아님.", 11, C["ink"])
    rect(s, 652, 432, 300, 200, C["soft_orange"], C["orange"], True)
    add_text(s, 664, 440, 276, 20, "D  Domain sign", 12, C["blue"], True)
    add_text(s, 664, 464, 276, 155, "1D 엄격한 외삽 9곳.\n9/9 양측 exact p=.0039.\n계층 bootstrap 95% CI\nlog-RMSE [0.17, 0.67].", 11, C["ink"])
    rect(s, 964, 432, 276, 200, C["soft"], C["ink"], True)
    add_text(s, 976, 440, 252, 20, "B  Bound audit", 12, C["ink"], True)
    add_text(s, 976, 464, 252, 155, "가설검정이 아니라 제약 감사.\n|ŷ−affine| ≤ margin·B\n17,645점 위반 0.", 11, C["ink"])
    foot(s, p(), TOTAL)

    s = blank(prs)
    head(s, "안정성", "유닛 단위에서 개선이 한쪽으로 몰리지 않았는지 확인한다")
    pic(s, "robustness_panel.png", 40, 80, 1200, 380)
    add_text(
        s,
        48,
        468,
        580,
        120,
        "(a) 결론  최종 executor 교정 후 유닛 log-RMSE 비는 평균 0.41, 계층 bootstrap 95% CI [0.17, 0.67]. 기하평균 RMSE 감소 33.8%. BH 보정 개별 유의는 선우다·배치2다.",
        13,
        C["ink"],
    )
    add_text(
        s,
        660,
        468,
        572,
        120,
        "(b) 결론  MICH 시험 8유닛의 개별 R²가 모두 양수다(0.50–0.96). 합친 점수 0.751은 한 유닛에 몰린 값이 아니므로 대표값으로 읽어도 된다. 이 그림은 검정이 아니라 분포 확인이다.",
        13,
        C["ink"],
    )
    add_text(s, 48, 598, 1184, 28, "정리  유닛 안에서는 근거가 있다. 데이터 종류가 세 개뿐이라 분야 전체 유의는 말하지 않는다.", 13, C["muted"])
    foot(s, p(), TOTAL)

    # Policy attack audit
    s = blank(prs)
    head(s, "정책 검증 — validation만으로는 부족하다", "12-domain common-backbone retrospective audit  ·  oracle은 비배포 상한")
    pic(s, "ppx_policy_audit.png", 24, 82, 760, 400)
    add_table(
        s,
        808,
        90,
        420,
        330,
        ["정책", "정확", "FA / FR"],
        [
            ["Always direct", "6/12", "0 / 6"],
            ["Always PP", "6/12", "6 / 0"],
            ["Val RMSE only", "7/12", "4 / 1"],
            ["Frozen PP-X", "8/12", "2 / 2"],
            ["Test oracle*", "12/12", "0 / 0"],
        ],
        font_size=12,
    )
    add_text(s, 808, 438, 420, 48, "* test oracle는 선택에 쓸 수 없는 성능 상한", 11, C["red"], True)
    add_text(
        s,
        48,
        510,
        1184,
        90,
        "결론  validation RMSE gate만으로는 오탐 4개다. unit-risk를 넣어도 prior contract를 고정 승인하면 오탐 2개가 남는다.\n따라서 contract가 admissible prior를 먼저 제한해야 한다. 이후 DS03 prospective에서 unsupported prior 거절이 실제 test-best PP-X route였다.",
        13,
        C["ink"],
    )
    foot(s, p(), TOTAL)

    # Fully equal candidate-budget comparison
    s = blank(prs)
    head(s, "공정 비교 — 9 settings × 8 baselines", "각 모델 validation 후보 30개  ·  search seed 42  ·  refit seeds 42–46")
    add_table(
        s,
        36,
        86,
        1208,
        380,
        ["Setting", "PP-X", "최강 30-candidate baseline", "Baseline", "ΔR²"],
        [
            ["HUST", "0.958", "GroupDRO", "0.955", "+.003"],
            ["Virkler", "0.888", "FT-Transformer", "0.890", "−.002"],
            ["NASA", "0.584", "Engression", "0.583", "+.000"],
            ["Sunwoda", "0.939", "linear-tail RBF", "0.838", "+.102"],
            ["RWTH", "0.878", "linear-tail RBF", "0.732", "+.146"],
            ["MICH", "0.751", "monotone NN", "−0.686", "+1.437"],
            ["MATR2019", "0.466", "FT-Transformer", "0.342", "+.123"],
            ["MATR-b2", "0.862", "plain MLP", "0.813", "+.049"],
            ["N-CMAPSS", "0.937", "Engression", "0.932", "+.005"],
        ],
        font_size=10,
    )
    rect(s, 36, 496, 1208, 92, C["soft_blue"], C["blue"], True)
    add_text(s, 52, 512, 1176, 54, "결론  PP-X 8/9 우세  ·  exact dataset sign test p=.0391  ·  3,360 training jobs\n예외  Virkler FT가 +.002  ·  NASA 사실상 동률  ·  PP-X는 typed executor라 하나의 공통 hyperparameter grid로 재개발하지 않음", 13, C["ink"], True, "center")
    foot(s, p(), TOTAL)

    # First prospective evidence
    s = blank(prs)
    head(s, "Prospective — N-CMAPSS DS03", "protocol → raw hash → selection artifact를 test Y 공개 전에 각각 GitHub 동결")
    add_table(
        s,
        42,
        90,
        570,
        260,
        ["Validation route", "MSE", "판정"],
        [
            ["direct fallback", "37.639", "선택"],
            ["basic prior-residual", "85.774", "거절"],
            ["multiscale prior-residual", "95.858", "거절"],
        ],
        font_size=13,
    )
    add_table(
        s,
        644,
        90,
        590,
        260,
        ["Prospective test", "R²", "판정"],
        [
            ["Engression 30c", "0.901", "최고"],
            ["FT-Transformer 30c", "0.899", "2위"],
            ["PP-X selected fallback", "0.882", "양수·비우월"],
            ["basic / multiscale PP", "0.832 / 0.869", "선택보다 열세"],
        ],
        font_size=12,
    )
    rect(s, 42, 392, 570, 160, C["soft_blue"], C["blue"], True)
    add_text(s, 58, 410, 538, 28, "PASS  미래 route 선택", 15, C["blue"], True, "center")
    add_text(s, 58, 456, 538, 72, "gate가 prior 두 개를 거절했고\n실제 test-best PP-X route 선택", 13, C["ink"], False, "center")
    rect(s, 644, 392, 590, 160, C["soft_orange"], C["orange"], True)
    add_text(s, 660, 410, 558, 28, "FAIL  predictive superiority", 15, C["orange"], True, "center")
    add_text(s, 660, 456, 558, 72, "Engression 대비 ΔR² −.0195\nunit wins 1/6 · worst ratio 1.989", 13, C["ink"], False, "center")
    add_text(s, 42, 584, 1192, 34, "정직한 결론  prospective route-selection success, predictive superiority not confirmed  ·  DS03는 within-family 검증", 13, C["muted"], True)
    foot(s, p(), TOTAL)

    # 10 Failures — tables only
    s = blank(prs)
    head(s, "실패 · 경계", "오늘 표는 1D 엄격한 외삽만.  같은 100%라도 geometry는 다르다.")
    add_table(
        s,
        48,
        110,
        1184,
        230,
        ["셋", "R²", "경계", "읽는 법"],
        [
            ["MICH (base)", "-1.522", "관계 이동 · 보정 꺼짐", "dual-scale 켠 뒤 0.751"],
            ["MATR2019", "0.466", "1D 건강 tail은 맞음", "주표에 남기되 약점"],
            ["N-CMAPSS", "0.937", "1D 100% · PCA2 0%", "조건 외삽. 먼 RUL tail 아님"],
            ["MATRb2", "0.862", "1D 100% · PCA2 0% · PCA3 99.2%", "다차원 geometry를 같이 적음"],
        ],
        font_size=12,
        red_cols={1},
    )
    add_table(
        s,
        48,
        380,
        580,
        220,
        ["Claim", "Detail"],
        [
            ["Model", "최종 PP-X = 공통 core + validation-approved executor"],
            ["Scope", "unit-disjoint · hull-out · val-only"],
            ["Not default", "dual-scale · transport · full history"],
            ["Venue", "분야 Q1–Q2"],
        ],
        font_size=12,
    )
    add_table(
        s,
        660,
        380,
        572,
        220,
        ["Not claimed", "Detail"],
        [
            ["Zn / Na", "사후 개발 — untouched 확증 아님"],
            ["도메인 transfer", "오늘 외삽 정의에 넣지 않음"],
            ["PAE 주표", "오늘 점수와 섞지 않음"],
            ["Universal SOTA", "모든 OOD 1등 아님"],
        ],
        font_size=12,
    )
    foot(s, p(), TOTAL)

    # Locked cohorts already scored — no new download
    s = blank(prs)
    head(s, "끝난 고호트 — PP-X", "잘된 셋만 PP-X.  MATR 두 곳은 기존 PP-X artifact  ·  Misata는 방금 같은 split으로 재실행")
    add_table(
        s,
        48,
        92,
        1184,
        360,
        ["고호트", "PP-X 경로", "PP-X R²", "판정"],
        [
            ["MATR 2019-01-24", "validation-calibrated latent", "0.466", "양수. 봉인 latent 0.257에서 개선. 주표와 동일"],
            ["MATR batch2", "regime transport", "0.862", "양수. 봉인 0.471/0.523에서 개선. 주표와 동일"],
            ["Misata", "core · val이 unbounded 선택", "0.829", "양수. MLP 0.854. 우월 확증은 여전히 실패"],
        ],
        font_size=13,
        red_cols={2},
    )
    add_text(
        s,
        48,
        470,
        1184,
        90,
        "Misata 새 실행  seed 42–46 ensemble 0.829 (개별 0.816–0.829). executor는 validation에서 bounded가 2%를 못 넘겨 unbounded. 기존 locked MLP와 같은 1298행.\nOxford·MATWI·팬·XJTU는 실패 고호트라 PP-X로 다시 돌리지 않았다.",
        13,
        C["ink"],
    )
    foot(s, p(), TOTAL)

    # CCMR evidence as one PP-X executor — after locked PP-X cohorts
    s = blank(prs)
    head(s, "PP-X 구조 검증 — CCMR v2.2 executor", "trajectory-domain bank + small-cohort route  ·  논문 메인 명칭은 PP-X")
    rect(s, 48, 88, 380, 200, C["soft_blue"], C["blue"], True)
    add_text(s, 68, 104, 340, 24, "PP-X trajectory executor", 14, C["blue"], True)
    add_text(s, 68, 140, 340, 120, "CCMR v2.2\npredictor v2.0 bank 유지\nval unit 3–9 → small_crossfit\n그 외 → stable / cautious / fallback", 14, C["ink"])
    rect(s, 448, 88, 380, 200, C["soft"], C["ink"], True)
    add_text(s, 468, 104, 340, 24, "개발 5고호트", 14, C["ink"], True)
    add_text(s, 468, 140, 340, 120, "false accept 0\nmax raw regret 0%\nSIT LFP만 엄격 개선\ngeo RMSE ratio 0.9986", 14, C["ink"])
    rect(s, 848, 88, 380, 200, C["soft_orange"], C["orange"], True)
    add_text(s, 868, 104, 340, 24, "거절 후보", 14, C["orange"], True)
    add_text(s, 868, 140, 340, 120, "CCMR v2.3 AC-CRPE\n안전(false accept 0)\n그러나 GM RMSE\nv2.2 대비 ≈0.0002% 악화", 14, C["ink"])
    add_text(
        s,
        48,
        310,
        1184,
        90,
        "PP-X 증거  소표본 route와 exact fallback이 선택적 prior 사용의 안전성을 지지. MultiStage에서 안정성 우위.\n주장 불가  PP-X가 보편적 SOTA / Alloy 정확도에서 Engression을 이겼다 / Alloy를 보고 재튜닝했다.",
        14,
        C["ink"],
    )
    add_text(
        s,
        48,
        420,
        1184,
        70,
        "Alloy A · MultiStage는 봉인 holdout이다. 점수를 보고 구조를 다시 맞추지 않는다. 아래 ablation·벤치마크는 retrospective evidence다.",
        13,
        C["red"],
        True,
    )
    add_text(s, 48, 510, 1184, 70, "근거  CCMR_V22_SMALL_COHORT_RESULTS_KO.md  ·  CCMR_V23_AC_CRPE_RESULTS_KO.md  ·  results/ccmr_model_ablation_stats_benchmark_v1/", 12, C["muted"])
    foot(s, p(), TOTAL)

    s = blank(prs)
    head(s, "Ablation — CCMR v2.0 → v2.2", "개발 5고호트  ·  paired RMSE gain + bootstrap CI")
    pic(s, "ccmr_v22_dev_improvement.png", 28, 82, 760, 390)
    add_table(
        s,
        808,
        92,
        420,
        330,
        ["통계", "값"],
        [
            ["GM RMSE ratio", "0.9986"],
            ["mean RMSE gain", "+0.14%"],
            ["strict better", "1 / 5"],
            ["non-worse", "5 / 5"],
            ["false accept", "0"],
            ["max regret", "0%"],
            ["sign-flip p", "1.0"],
            ["boot CI95", "[0, 0.42]%"],
        ],
        font_size=12,
    )
    add_text(
        s,
        48,
        500,
        1184,
        90,
        "SIT LFP에서만 pooled improvement 0→0.7% (small_crossfit_bank). 나머지 4곳은 non-worse.\n paired sign-flip은 표본 5로 유의하지 않다. 승격 근거는 엄격 악화 없음 + false accept 0이다.",
        13,
        C["ink"],
    )
    foot(s, p(), TOTAL)

    s = blank(prs)
    head(s, "Ablation — dynamics bank experts", "단일 expert는 약하다. bank + selection이 프로토콜이다.")
    pic(s, "ccmr_expert_ablation.png", 40, 90, 1180, 400)
    add_text(
        s,
        48,
        520,
        1184,
        80,
        "단일 dynamics expert의 mean pooled gain은 음수인 구간이 많다. 그래서 PP-X의 CCMR executor는 expert를 전역 기본값으로 두지 않고,\nvalidation에서 고른 bank / small-cohort route / exact fallback으로 위험을 가른다.",
        13,
        C["ink"],
    )
    foot(s, p(), TOTAL)

    s = blank(prs)
    head(s, "벤치마크 — 두 성공 holdout", "경쟁군 + CCMR v2.2 overlay  ·  Alloy/MultiStage는 구조 재튜닝에 쓰지 않음")
    pic(s, "ccmr_holdout_benchmark.png", 20, 78, 1240, 430)
    add_text(
        s,
        48,
        520,
        1184,
        80,
        "Alloy  정확도 1위는 Engression (R²≈0.991). CCMR v2.2는 R²≈0.771, regret 0.\nMultiStage  CCMR v2.2가 R²≈0.989·regret 0. Engression은 R²≈0.979지만 max regret ≈205%.",
        13,
        C["ink"],
    )
    foot(s, p(), TOTAL)

    s = blank(prs)
    head(s, "주장 경계 — 정확도 vs 안정성", "한 고호트 승리가 전역 SOTA가 아니다")
    pic(s, "ccmr_claim_summary.png", 60, 90, 1160, 380)
    add_text(
        s,
        48,
        500,
        1184,
        90,
        "Alloy에서는 Engression이 정확도에서 앞선다. MultiStage에서는 Engression의 유닛 regret이 폭발하고 CCMR executor가 안전하다.\nPP-X의 핵심은 “모든 holdout 1등”이 아니라, 근거 없는 prior를 거절하고 외삽 위험을 제한하는 것이다.",
        13,
        C["ink"],
    )
    foot(s, p(), TOTAL)

    s = blank(prs)
    head(s, "거절 — CCMR v2.3 AC-CRPE", "강한 no-prior portfolio + contract residual  ·  엄격 RMSE 개선 실패")
    pic(s, "ccmr_v23_rejection_path.png", 28, 82, 760, 390)
    add_table(
        s,
        808,
        92,
        420,
        330,
        ["최종 v10", "값"],
        [
            ["GM vs v2.2", "1.000002"],
            ["Δ GM RMSE", "−0.0002%"],
            ["non-worse", "4 / 5"],
            ["false accept", "0"],
            ["max regret", "0"],
            ["fallback", "exact OK"],
            ["promoted", "NO"],
            ["holdout replay", "미실행"],
        ],
        font_size=12,
        red_cols={1},
    )
    add_text(
        s,
        48,
        500,
        1184,
        90,
        "강한 Engression anchor는 development에서 regret을 키웠고, nested v2.2 safety로 수렴했다.\n최종은 안전하지만 v2.2 대비 엄격 평균 RMSE 개선이 없어 거절. Alloy를 보고 재튜닝하지 않았다.",
        13,
        C["ink"],
    )
    foot(s, p(), TOTAL)

    # Close — forward only, not a repeat of slide 2
    s = blank(prs)
    head(s, "다음", "지도는 루트 장에 있다.  여기서는 앞으로만 말한다.")

    rect(s, 48, 110, 380, 360, C["soft_blue"], C["blue"], True)
    rect(s, 48, 110, 8, 360, C["blue"])
    now = rect(s, 286, 126, 118, 24, C["blue"], None, True)
    fill_shape_text(now, "지금", 11, C["white"], True)
    add_text(s, 72, 126, 200, 32, "PP-X", 22, C["blue"], True)
    add_text(s, 72, 170, 330, 22, "식 없는 경로", 13, C["muted"])
    add_text(s, 72, 210, 330, 80, "contract-conditioned prior\nvalidation-approved executor\n근거 없으면 fallback", 14, C["ink"])
    add_text(s, 72, 320, 330, 60, "CCMR v2.2 구조 증거\nv2.3 거절 기록", 14, C["ink"])
    add_text(s, 72, 400, 330, 40, "만능 SOTA 아님", 13, C["blue"], True)

    rect(s, 450, 110, 380, 360, C["soft_orange"], C["orange"], True)
    rect(s, 450, 110, 8, 360, C["orange"])
    nxt = rect(s, 688, 126, 118, 24, C["orange"], None, True)
    fill_shape_text(nxt, "다음", 11, C["white"], True)
    add_text(s, 474, 126, 200, 32, "PAE", 22, C["orange"], True)
    add_text(s, 474, 170, 330, 22, "식 있는 경로", 13, C["muted"])
    add_text(s, 474, 210, 330, 90, "LLM·온톨로지·source gate\n허용된 식만 실행\n이득 없으면 PP-X", 14, C["ink"])
    add_text(s, 474, 400, 330, 40, "다음 논문", 13, C["orange"], True)

    rect(s, 852, 110, 380, 360, C["soft"], C["ink"], True)
    rect(s, 852, 110, 8, 360, C["ink"])
    later = rect(s, 1090, 126, 118, 24, C["ink"], None, True)
    fill_shape_text(later, "박사", 11, C["white"], True)
    add_text(s, 876, 126, 220, 32, "Assurance", 20, C["ink"], True)
    add_text(s, 876, 170, 330, 22, "두 경로 통합", 13, C["muted"])
    add_text(s, 876, 210, 330, 80, "믿기 / 보류 / 거절\n허가증으로 운행\nFail이면 옮기지 않음", 14, C["ink"])
    add_text(s, 876, 400, 330, 40, "이후", 13, C["ink"], True)

    add_text(
        s,
        48,
        500,
        1184,
        70,
        "밖을 지탱하는 것은 가정이다.  식이 없으면 PP-X.  executor는 근거 있을 때만 켠다.",
        15,
        C["ink"],
        True,
        "center",
    )
    foot(s, p(), TOTAL)

    # PAE concept + gates
    s = blank(prs)
    head(s, "후속 — PAE 컨셉", "출처가 고정된 식을 컴파일하고, 적용 가능한지 검증한 뒤에만 실행한다.")
    add_text(s, 48, 84, 1184, 22, "PAE는 식 생성기가 아니다. 문헌 식 카드가 의미적으로 실행 가능하고, 검증에서 전이될 때만 켠다.", 13, C["ink"])

    inn = rect(s, 48, 116, 360, 56, C["soft"], C["ink"], True)
    fill_shape_text(inn, "문제 서술  +  데이터 스키마", 13, C["ink"], True)
    right_arrow(s, 420, 132, 28, 22, C["muted"])
    rag = rect(s, 460, 116, 360, 56, C["soft"], C["ink"], True)
    fill_shape_text(rag, "RAG   문헌 식 카드 · 인용만 검색", 13, C["ink"], True)
    add_text(s, 840, 124, 392, 40, "실행 권한 없음. 후보만 줄인다.", 12, C["muted"])

    down_arrow(s, 624, 178, 28, 14, C["muted"])

    g1 = rect(s, 48, 198, 360, 118, C["soft_blue"], C["blue"], True)
    rect(s, 48, 198, 8, 118, C["blue"])
    add_text(s, 68, 206, 320, 22, "LLM 게이트", 15, C["blue"], True)
    add_text(s, 68, 232, 320, 72, "카드 ID와 역할 결합만 제안.\n식·상수·인용·고장경계를\n만들지 못한다.", 12, C["ink"])
    right_arrow(s, 420, 240, 28, 22, C["muted"])

    g2 = rect(s, 460, 198, 360, 118, C["soft_orange"], C["orange"], True)
    rect(s, 460, 198, 8, 118, C["orange"])
    add_text(s, 480, 206, 320, 22, "온톨로지 · 식 판별", 15, C["orange"], True)
    add_text(s, 480, 232, 320, 72, "target · 메커니즘 · 역할 · 단위\n경계 · 출처 · learnable slot.\n하나라도 깨지면 실행 불가.", 12, C["ink"])
    right_arrow(s, 832, 240, 28, 22, C["muted"])

    g3 = rect(s, 872, 198, 360, 118, C["soft"], C["ink"], True)
    rect(s, 872, 198, 8, 118, C["ink"])
    add_text(s, 892, 206, 320, 22, "Source 게이트", 15, C["ink"], True)
    add_text(s, 892, 232, 320, 72, "검증 holdout에서 식 경로가\nPP-X보다 나을 때만 통과.\n안전 증명이 아니다.", 12, C["ink"])

    down_arrow(s, 624, 322, 28, 14, C["muted"])
    dec = rect(s, 310, 340, 660, 40, C["ink"], None, True)
    fill_shape_text(dec, "식이 컴파일되고, 검증에서 이기는가?", 15, C["white"], True)

    vline(s, 420, 380, 396, C["rule"])
    vline(s, 640, 380, 396, C["rule"])
    vline(s, 860, 380, 396, C["rule"])
    hline(s, 420, 860, 396, C["rule"])
    down_arrow(s, 404, 396, 28, 14, C["orange"])
    down_arrow(s, 624, 396, 28, 14, C["blue"])
    down_arrow(s, 844, 396, 28, 14, C["muted"])
    add_text(s, 300, 394, 90, 16, "YES", 11, C["orange"], True, "right")
    add_text(s, 880, 394, 90, 16, "NO", 11, C["blue"], True, "left")

    o1 = rect(s, 48, 418, 360, 100, C["soft_orange"], C["orange"], True)
    add_text(s, 64, 426, 328, 22, "PAE 실행", 14, C["orange"], True)
    add_text(s, 64, 452, 328, 56, "허용 계수만 학습.\n현재 상태 → 경계 적분. RUL=0.", 12, C["ink"])
    o2 = rect(s, 460, 418, 360, 100, C["soft_blue"], C["blue"], True)
    add_text(s, 476, 426, 328, 22, "PP-X로 되돌림", 14, C["blue"], True)
    add_text(s, 476, 452, 328, 56, "식 계약이 안 닫히거나\n검증에서 이득이 없을 때.", 12, C["ink"])
    o3 = rect(s, 872, 418, 360, 100, C["soft"], C["ink"], True)
    add_text(s, 888, 426, 328, 22, "거절", 14, C["ink"], True)
    add_text(s, 888, 452, 328, 56, "식도 없고 PP-X도 불안정하면\n예측하지 않는다.", 12, C["ink"])

    add_text(s, 48, 532, 1184, 28, "검사 항목   상태 · 고장 메커니즘 · 필수 변수 · 단위 · 경계 · 출처.   데이터셋 이름으로 경로를 고르지 않는다.", 12, C["ink"])
    add_text(s, 48, 564, 1184, 24, "자유 LLM 식 생성은 성공이 아니라 계약 위반이다. 다음 논문 경로이며 오늘 주표와 숫자를 섞지 않는다.", 12, C["muted"])
    foot(s, p(), TOTAL)

    # PAE feasibility
    s = blank(prs)
    head(s, "후속 — PAE 피저빌리티", "게이트가 왜 필요한지, 식이 맞을 때와 경계만 있을 때를 갈라 본다.")
    add_table(
        s,
        48,
        96,
        1184,
        240,
        ["자료", "넣은 식", "게이트 읽기", "PAE", "PP-X"],
        [
            ["알루미늄 균열", "Paris + 49.8 mm 파단", "식·경계·역할이 닫힘 → 실행", "0.969", "0.888"],
            ["미시간 배터리", "80% EOL 경계만", "메커니즘 식 없음 → 되돌려야 함", "0.635", "0.826"],
            ["NASA 실험셀", "용량 속도 quotient", "셀마다 regime이 달라 계약 약함", "0.492", "0.741"],
        ],
        font_size=13,
    )
    add_table(
        s,
        48,
        360,
        1184,
        180,
        ["이미 본 것", "아직 아닌 것"],
        [
            ["식이 맞으면 PAE가 PP-X를 이김 (Virkler +0.081)", "RAG·LLM·온톨로지를 주표에 넣지 않음"],
            ["경계만 알면 PAE를 켜면 진다 (MICH · NASA)", "source gate의 광범위 일반화"],
            ["동일-split, 오늘 포트폴리오와 분리", "LLM이 물리법칙을 발견한다는 주장"],
        ],
        font_size=13,
    )
    add_text(s, 48, 556, 1184, 36, "결론  식이 맞으면 PAE, 경계만 있으면 PP-X. 미시간 0.826은 raw-cycle 맞비교(주표 MICH 0.751).", 13, C["ink"])
    add_text(s, 48, 592, 1184, 24, "게이트는 성능을 올리는 부품이 아니라, 틀린 식을 실행하지 못하게 막는 장치다.", 12, C["muted"])
    foot(s, p(), TOTAL)

    # Close
    s = blank(prs)
    for sh in list(s.shapes):
        sh._element.getparent().remove(sh._element)
    bg = s.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, 0, 0, W, H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = C["white"]
    bg.line.fill.background()
    frame = s.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, px(40), px(32), px(1200), px(656))
    frame.fill.background()
    frame.line.color.rgb = RGBColor(0x22, 0x22, 0x22)
    frame.line.width = Pt(1.25)
    add_text(s, 100, 200, 1080, 48, "감사합니다", 34, C["ink"], True, "center")
    add_text(s, 100, 270, 1080, 36, "Q & A", 22, C["blue"], True, "center")
    add_text(
        s,
        140,
        360,
        1000,
        80,
        "오늘: PP-X  ·  contract-conditioned prior + validation-approved executor + fallback\n구조 증거: CCMR v2.2  ·  한계: 정확도≠안정성, prospective 확증은 남음",
        15,
        C["ink"],
        False,
        "center",
    )
    add_text(s, 100, 500, 1080, 28, "질문 받겠습니다.", 16, C["muted"], False, "center")
    add_text(s, 100, 560, 1080, 22, "SPS Lab  ·  박사과정 박진서", 13, C["muted"], False, "center")
    p()

    # Appendix — backup after Q&A
    s = blank(prs)
    head(s, "보조 — 이게 무슨 데이터인가", "백업. 본 발표는 Q&A에서 끝낸다.")
    add_table(
        s,
        28,
        88,
        1224,
        540,
        ["표기", "물건", "한 줄"],
        [
            ["HUST", "화중과대 리튬 배터리", "충전 방법(프로토콜)이 다른 셀. 중국 랩."],
            ["Sunwoda", "선우다 상용 리튬셀", "공장 셀. 학습에 안 넣은 다른 셀로 시험."],
            ["RWTH", "아헨공대 리튬 배터리", "독일 랩 셀. 역시 처음 보는 셀로 시험."],
            ["MICH", "미시간대 리튬 배터리", "건강↔수명 관계가 학습 때와 달라진 셀."],
            ["MATR19", "MIT 2019 수명 벤치", "노트북/그리드용 셀. 고전 벤치마크."],
            ["MATRb2", "MIT 배치 2", "같은 랩, 다른 생산 로트. 시험 9셀."],
            ["NASA", "NASA PCoE 실험용 18650", "우주선/로켓이 아님. 용량 70%까지 남은 횟수."],
            ["Virkler", "알루미늄 피로균열", "판에 금이 얼마나 남았나. 배터리 아님."],
            ["N-CMAPSS", "항공기 엔진 시뮬", "NASA가 만든 디지털 터보팬. 실제 비행 기록 아님."],
        ],
        font_size=12,
    )
    add_text(s, 28, 638, 1224, 24, "NASA가 두 개다.  위는 배터리 셀, 아래 엔진은 시뮬레이터.", 12, C["muted"])
    foot(s, p(), TOTAL)

    s = blank(prs)
    head(s, "보조 — 누구로 나누고, 뭘 보고, 뭘 맞추나", "미지 = 학습 때 이름조차 안 본 셀/시편/엔진")
    add_table(
        s,
        20,
        86,
        1240,
        520,
        ["셋", "학습", "검증 / 시험", "보고 (X)", "맞추는 것 (Y)"],
        [
            ["화중 배터리", "충전법 1–6, 아직 건강한 구간", "충전법 7–8 / 9–10, 용량이 더 떨어진 구간", "지금까지의 용량·떨어지는 속도", "수명까지 남은 사이클"],
            ["선우다 · 아헨 · 미시간", "다른 셀의 건강한 구간", "처음 보는 셀의 말기 구간", "지금 건강, 최근 속도, 고장까지 여유", "수명까지 남은 사이클"],
            ["MIT 2019 / 배치2", "일부 셀의 건강한 구간 (배치2는 30셀)", "나머지 셀의 말기 (배치2는 9+9셀)", "지금까지의 용량 이력", "수명까지 남은 사이클"],
            ["NASA 실험셀", "일부 셀, 용량이 아직 높은 구간", "다른 셀, 용량이 학습 최저보다 낮은 구간", "용량, 속도, 몇 번째 사이클인가", "용량 70% 될 때까지 남은 횟수"],
            ["알루미늄 균열", "다른 시편의 짧은 금", "처음 보는 시편의 긴 금", "금 길이, 자라는 속도, 하중", "쪼개질 때까지 남은 반복"],
            ["항공기 엔진", "본 적 있는 비행조건의 엔진", "처음 보는 비행조건의 엔진", "센서(조건으로 나눈 값), 운전모드", "엔진이 버틸 남은 시간"],
        ],
        font_size=11,
    )
    add_text(s, 20, 618, 1240, 28, "같은 셀을 시간만 잘라 뒤를 맞추지 않는다.  시험 셀의 미래·최종 수명은 X에 넣지 않는다.", 12, C["muted"])
    foot(s, p(), TOTAL)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        prs.save(str(OUT))
        print(f"Saved {OUT} ({n} slides)")
    except PermissionError:
        alt = OUT.with_name("PP_Research_Detailed_v2_ccmr.pptx")
        prs.save(str(alt))
        print(f"Primary locked; saved {alt} ({n} slides)")


if __name__ == "__main__":
    build()
