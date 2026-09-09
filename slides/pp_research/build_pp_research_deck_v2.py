#!/usr/bin/env python3
"""PP research briefing v2 — cleaner narrative + embedded figures."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Pt

OUT = Path("/Users/baghyeongbae/Desktop/연구/ppt/pp/PP_Research_Detailed_v2.pptx")
ASSETS = Path("/Users/baghyeongbae/Desktop/연구/ppt/pp/_build")
FIGS = ASSETS / "figs"

W, H = 12192000, 6858000
PX = 9525


def px(n: float) -> int:
    return int(n * PX)


C = {
    "ink": RGBColor(0x10, 0x2A, 0x43),
    "navy": RGBColor(0x16, 0x3A, 0x5F),
    "crimson": RGBColor(0x8B, 0x00, 0x29),
    "crimson2": RGBColor(0xB2, 0x0D, 0x3C),
    "blue": RGBColor(0x2E, 0x6F, 0x95),
    "orange": RGBColor(0xF4, 0xA2, 0x61),
    "teal": RGBColor(0x2A, 0x9D, 0x8F),
    "red": RGBColor(0xC4, 0x45, 0x36),
    "green": RGBColor(0x3A, 0x7D, 0x44),
    "pale": RGBColor(0xF5, 0xF8, 0xFA),
    "wash": RGBColor(0xEA, 0xF2, 0xF8),
    "grey": RGBColor(0x62, 0x7D, 0x98),
    "line": RGBColor(0xC8, 0xD4, 0xE3),
    "white": RGBColor(0xFF, 0xFF, 0xFF),
    "pink": RGBColor(0xF5, 0xDD, 0xE6),
    "rose": RGBColor(0xE8, 0xBC, 0xCB),
}


def set_run(run, size=18, color=None, bold=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = "AppleGothic"
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
    lines = str(text).split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}[align]
        run = p.add_run()
        run.text = line
        set_run(run, size=size, color=color or C["ink"], bold=bold)
    return box


def rect(slide, left, top, width, height, fill=None, line=None, round_corners=False):
    st = MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if round_corners else MSO_AUTO_SHAPE_TYPE.RECTANGLE
    sh = slide.shapes.add_shape(st, px(left), px(top), px(width), px(height))
    sh.line.fill.background()
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is not None:
        sh.line.color.rgb = line
        sh.line.width = Pt(1)
    if round_corners:
        sh.adjustments[0] = 0.08
    return sh


def arrow(slide, left, top, width, height, fill):
    sh = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RIGHT_ARROW, px(left), px(top), px(width), px(height))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    return sh


def card(slide, left, top, width, height, head, body, accent):
    rect(slide, left, top, width, height, C["white"], C["line"], True)
    rect(slide, left, top, width, 7, accent, None, True)
    add_text(slide, left + 16, top + 18, width - 32, 26, head, 16, C["ink"], True)
    add_text(slide, left + 16, top + 50, width - 32, height - 62, body, 13, C["grey"])


def pic(slide, name, left, top, width, height):
    path = FIGS / name
    if path.exists():
        slide.shapes.add_picture(str(path), px(left), px(top), px(width), px(height))
        return True
    return False


def topbar(slide, sec, title, sub=""):
    rect(slide, 0, 0, 1280, 8, C["navy"])
    rect(slide, 70, 48, 8, 52, C["orange"])
    add_text(slide, 92, 45, 70, 24, sec, 14, C["navy"], True)
    add_text(slide, 92, 68, 1080, 34, title, 26, C["ink"], True)
    if sub:
        add_text(slide, 92, 106, 1080, 22, sub, 12, C["grey"])


def foot(slide, n):
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, px(72), px(674), px(1208), px(674))
    line.line.color.rgb = C["line"]
    line.line.width = Pt(1)
    add_text(slide, 72, 682, 780, 16, "Smart Production Systems Lab.  /  PP Extrapolation Research", 9, C["grey"])
    add_text(slide, 1140, 680, 60, 18, f"{n:02d}", 11, C["grey"], True, "right")


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

    def p():
        nonlocal n
        n += 1
        return n

    # ===== 1 Cover =====
    s = blank(prs)
    for sh in list(s.shapes):
        sh._element.getparent().remove(sh._element)
    rect(s, 0, 0, 1280, 720, C["crimson"])
    hero = ASSETS / "domain_hero.png"
    if hero.exists():
        s.shapes.add_picture(str(hero), px(560), 0, px(720), px(720))
    rect(s, 0, 0, 710, 720, C["crimson"])
    rect(s, 650, 0, 160, 720, C["crimson2"])
    rect(s, 72, 78, 88, 5, C["white"])
    add_text(s, 72, 42, 360, 24, "KOREA UNIVERSITY", 14, C["white"], True)
    add_text(s, 72, 155, 560, 110, "PP EXTRAPOLATION\nRESEARCH BRIEFING", 38, C["white"], True)
    add_text(s, 76, 310, 520, 70, "SAAR 주결과 · Zn/교차도메인 업데이트 ·\n주장 경계 · 다음에 할 일", 18, C["pink"])
    add_text(s, 76, 420, 500, 28, "남은수명 · 배터리 · 베어링 · 공구 상태예측", 14, C["rose"])
    add_text(s, 76, 560, 400, 50, "Smart Production Systems Lab.\nPP 연구 발표자료 (v2 갱신)", 14, C["white"])
    add_text(s, 76, 650, 200, 20, "2026-09", 12, C["rose"])
    p()

    # ===== 2 Agenda =====
    s = blank(prs)
    topbar(s, "00", "발표 구성", "본문 A–D  ·  상세 근거와 원문은 Appendix")
    items = [
        ("A", "문제 설정", "외삽 바운더리와\n연구 범위", C["orange"]),
        ("B", "방법론", "SAAR 구조와\n공용 prior", C["teal"]),
        ("C", "실증·업데이트", "3배터리 + Zn·\n교차도메인 현황", C["blue"]),
        ("D", "전망", "논문 타깃·PAE\n경계·부록", C["navy"]),
    ]
    for i, (letter, title, body, col) in enumerate(items):
        x = 85 + i * 290
        rect(s, x, 200, 265, 240, C["white"], C["line"], True)
        rect(s, x, 200, 265, 64, col)
        add_text(s, x + 20, 214, 40, 28, letter, 22, C["white"], True)
        add_text(s, x + 60, 218, 180, 30, title, 18, C["white"], True)
        add_text(s, x + 22, 300, 220, 90, body, 16, C["ink"])
        if i < 3:
            arrow(s, x + 268, 300, 18, 22, C["line"])
    rect(s, 85, 490, 1110, 95, C["wash"], None, True)
    add_text(
        s,
        115,
        520,
        1050,
        40,
        "요지  ·  논문 주모델은 SAAR. Zn·베어링은 확장·한계 보고. PAE 결과와는 아직 안 겹침.",
        16,
        C["ink"],
        True,
    )
    foot(s, p())

    # ===== 3 Motivation visual =====
    s = blank(prs)
    topbar(s, "A", "왜 이 연구를 하는가", "학습 안에서는 잘 맞아도, 밖에서는 가정이 예측을 가른다")
    pic(s, "motivation_futures.png", 60, 150, 780, 300)
    card(s, 870, 170, 340, 120, "문제", "일반 신경망은\n학습 범위 밖에서\n예측이 흔들리기 쉽다", C["red"])
    card(s, 870, 310, 340, 120, "함정", "물리식을 세게 넣으면\n그 식이 틀릴 때\n오히려 더 위험하다", C["orange"])
    card(s, 870, 450, 340, 120, "출발점", "지금 분명히 아는\n약한 가정만 쓰고\n모르면 예측을 멈춘다", C["teal"])
    add_text(s, 70, 470, 760, 50, "문헌에서 가져온 한 문장: 밖을 지탱하는 것은 데이터가 아니라 가정이다.", 14, C["crimson"], True)
    add_text(s, 70, 530, 760, 50, "출처: 외삽 문헌조사 발표", 12, C["grey"])
    foot(s, p())

    # ===== 4 Lit gap =====
    s = blank(prs)
    topbar(s, "A", "연구의 위치", "선행 연구와 겹치는 지점, 그리고 본 연구가 다루는 공백")
    card(s, 80, 165, 360, 250, "이미 축적된 흐름", "물리–신경망 결합\n단조성·방향성 제약\n불확실성 기반 예측 보류", C["grey"])
    card(s, 470, 165, 360, 250, "남는 질문", "지배 방정식은 모르더라도\n수명 경계·최근 이력 같은\n약한 가정은 어떻게 쓸 것인가", C["orange"])
    card(s, 860, 165, 360, 250, "본 연구의 답", "약한 가정을 명시적으로 고정하고\n신경망이 이를 훼손하지 않게 하며\n근거가 부족하면 예측을 유보한다", C["teal"])
    rect(s, 80, 450, 1140, 110, C["wash"], None, True)
    add_text(s, 110, 475, 1080, 28, "요약", 14, C["navy"], True)
    add_text(s, 110, 510, 1080, 35, "새로운 모형의 발명이라기보다, 학습 지지 밖 남은수명 예측에서 약한 가정을 안전하게 운용하는 절차를 정립한 연구다.", 15, C["ink"], True)
    foot(s, p())

    # ===== 4b PP research boundary =====
    s = blank(prs)
    topbar(s, "A", "PP는 어떤 외삽 연구인가", "연구 바운더리 — 하는 것 / 안 하는 것")
    card(
        s,
        70,
        155,
        370,
        320,
        "하는 것 (In scope)",
        "연속 열화의 남은수명(RUL)\n개체(unit) 분리 평가\n사전 좌표의 train hull 밖\n식 없는 구조 prior 실행\n(SAAR / 최종 PP)",
        C["teal"],
    )
    card(
        s,
        460,
        155,
        370,
        320,
        "안 하는 것 (Out)",
        "임의 tabular 만능 외삽\n도메인 물리식 발견/주입\n같은 개체 단순 미래예측만\n테스트 보고 prior 재선택\n모든 OOD 1등 주장",
        C["red"],
    )
    card(
        s,
        850,
        155,
        360,
        320,
        "한 줄 정의",
        "unit-disjoint\n+\nout-of-support\nRUL extrapolation\nunder equation-free\nstructural priors",
        C["navy"],
    )
    add_text(
        s,
        90,
        500,
        1100,
        70,
        "외삽 좌표 예: health/capacity 끝단 · 균열 길이 · 부하(TRA)·만기 조건.\n"
        "일반 ‘시간상 미래’나 분류 OOD와 문제 정의를 구분한다.",
        14,
        C["ink"],
        True,
    )
    foot(s, p())

    # ===== 4c Extrapolation scope rules =====
    s = blank(prs)
    topbar(s, "A", "외삽 범위를 어떻게 잡았나", "스플릿·평가에 넣은 규칙")
    rules = [
        ("① 계약 먼저", "경계·unit·시간축\n인과 입력·외삽 좌표를\n데이터 이름보다 먼저 고정", C["orange"]),
        ("② unit 분리", "같은 셀/시편/엔진이\ntrain·val·test에\n동시에 들어가지 않음", C["teal"]),
        ("③ train = support 안", "학습 unit의\n관측 범위 내부 행만 사용\n스케일도 train만", C["blue"]),
        ("④ 평가는 hull 밖", "val/test는 사전 좌표가\ntrain convex hull 밖인\n행만 남겨 점수", C["navy"]),
        ("⑤ val-only 선택", "설정·epoch·모듈 승인은\n검증만 · test 라벨로\n구조 다시 고르지 않음", C["crimson"]),
        ("⑥ 보고", "주: pooled R²\n보조: unit·seed·경계위반\n성공과 실패를 같이", C["green"]),
    ]
    for i, (h, b, col) in enumerate(rules):
        x = 70 + (i % 3) * 390
        y = 155 + (i // 3) * 230
        rect(s, x, y, 370, 210, C["white"], C["line"], True)
        rect(s, x, y, 370, 42, col)
        add_text(s, x + 14, y + 8, 340, 28, h, 15, C["white"], True)
        add_text(s, x + 14, y + 60, 340, 130, b, 14, C["ink"])
    foot(s, p())

    # ===== 5 Research Q + ladder =====
    s = blank(prs)
    topbar(s, "A", "질문 세 가지", "식을 넣을까 · 식 없이 어떻게 · 믿어도 될까")
    pic(s, "prior_ladder.png", 40, 145, 760, 330)
    qs = [
        ("Q1", "어떤 식을 넣을까?", "PAE"),
        ("Q2", "식 없이 어떻게?", "PP"),
        ("Q3", "믿어도 되나?", "보증"),
    ]
    for i, (q, body, tag) in enumerate(qs):
        y = 160 + i * 120
        rect(s, 840, y, 370, 100, C["white"], [C["orange"], C["teal"], C["navy"]][i], True)
        add_text(s, 860, y + 16, 80, 24, q, 14, C["grey"], True)
        add_text(s, 860, y + 42, 230, 40, body, 16, C["ink"], True)
        add_text(s, 1095, y + 45, 95, 28, tag, 15, [C["orange"], C["teal"], C["navy"]][i], True)
    add_text(
        s,
        90,
        500,
        1100,
        45,
        "지금 논문은 Q2(PP). PP는 prior-free가 아니라 equation-free. 일반 구조 bias(safeguard)는 있다.",
        15,
        C["teal"],
        True,
    )
    foot(s, p())

    # ===== 6 PP/PAE flow =====
    s = blank(prs)
    topbar(s, "B", "PP와 PAE의 분기", "후보식이 있으면 PAE(식+NN) · 없으면 PP")
    pic(s, "pp_pae_flow.png", 50, 145, 1180, 380)
    add_text(
        s,
        90,
        545,
        1100,
        40,
        "PAE에서도 식만으로 끝내지 않는다. 허용된 식 위에 부족분은 NN이 제한적으로 학습한다.",
        14,
        C["ink"],
        True,
    )
    foot(s, p())

    # ===== 6b PP prior selection =====
    s = blank(prs)
    topbar(s, "B", "PP는 prior를 어떻게 켜나", "전부 한꺼번에가 아니라, 계약상 최소 bundle만")
    card(
        s,
        80,
        155,
        380,
        250,
        "원칙",
        "데이터셋 이름으로\n고르지 않는다.\n외삽 타입·관측 가능\n정보로 사전에 고정된\neligibility rule",
        C["teal"],
    )
    card(
        s,
        480,
        155,
        380,
        250,
        "예: 배터리",
        "경계·방향·이력\nbounded correction\nrate는 causal 이력이\n있을 때만",
        C["orange"],
    )
    card(
        s,
        880,
        155,
        320,
        250,
        "예: tabular",
        "affine tail\n연속성·기울기 제한\n거리·거절",
        C["navy"],
    )
    add_text(
        s,
        90,
        430,
        1100,
        90,
        "규칙 예: 고장경계 관측 → boundary 허용 / causal history → history·rate 허용 /\n"
        "정적 tabular만 → affine·연속성만 / 방향이 외부 정당화 → 부호 제약 / 안전 연장 불가 → 거절\n"
        "성능을 보고 사람이 고르면 사후 튜닝. PP의 선택은 기계적·사전 고정이어야 PAE와 안 겹친다.",
        13,
        C["ink"],
        True,
    )
    add_text(
        s,
        90,
        545,
        1100,
        40,
        "흐름: 문제 계약 → 적용 가능한 weak-prior bundle → 안전한 실행 (equation-free safeguard)",
        14,
        C["crimson"],
        True,
    )
    foot(s, p())

    # ===== 7 Core idea: why NN fails OOS =====
    s = blank(prs)
    topbar(s, "B", "문제와 핵심 아이디어", "PP: support-aware residual extrapolation")
    pic(s, "pp_core_idea.png", 40, 140, 1200, 430)
    add_text(
        s,
        90,
        575,
        1100,
        30,
        "순서: 왜 일반 NN이 외삽에서 위험한가 → PP가 무엇을 제한하는가",
        13,
        C["grey"],
        True,
    )
    foot(s, p())

    # ===== 8 Architecture =====
    s = blank(prs)
    topbar(s, "B", "모형 구조", "affine 기본 경로 + 제한된 residual")
    pic(s, "pp_architecture.png", 40, 135, 1200, 470)
    foot(s, p())

    # ===== 8b Equation (one formula) =====
    s = blank(prs)
    topbar(s, "B", "수식은 하나면 충분", "각 항이 NN 자유도를 어떻게 줄이는가")
    pic(s, "pp_equation_panel.png", 40, 145, 1200, 400)
    add_text(
        s,
        90,
        560,
        1100,
        40,
        "ŷ = ŷ_affine + w(d) · B tanh(r_θ(x)/B)   ·   global path + support-adaptive bounded correction",
        13,
        C["ink"],
        True,
    )
    foot(s, p())

    # ===== 9 Shared vs dataset-specific =====
    s = blank(prs)
    topbar(s, "B", "공용 prior / 데이터별 학습", "배터리 열화식을 미리 넣는 모델이 아니다")
    pic(s, "pp_shared_vs_data.png", 40, 140, 1200, 380)
    add_text(
        s,
        90,
        540,
        1100,
        55,
        "발표 멘트: 어떤 데이터셋에도 동일한 외삽 구조를 적용하고, 각 데이터셋에서는 추세와 residual만 새로 학습한다.\n"
        "prior는 데이터셋 고유식이 아니라, 관측 밖에서 복잡한 NN 보정을 덜 신뢰한다는 공용 외삽 prior다.",
        13,
        C["crimson"],
        True,
    )
    foot(s, p())

    # ===== 9b Support adaptive (detail) =====
    s = blank(prs)
    topbar(s, "B", "학습에서 멀어질수록 보정을 줄인다", "w(d): 가까운 곳은 자세히, 먼 곳은 덜 신뢰")
    pic(s, "support_adaptive.png", 50, 145, 720, 380)
    card(s, 800, 170, 400, 120, "왜 필요한가", "보정을 안 막으면\n밖에서 기본 경향을\n덮어쓸 수 있다", C["red"])
    card(s, 800, 310, 400, 120, "어떻게 하나", "학습 범위에서 멀수록\n보정 허용폭을\n부드럽게 바꾼다", C["teal"])
    card(s, 800, 450, 400, 120, "그래서", "보정이 무한히\n커지지 않고\n추세 중심으로 수축", C["navy"])
    foot(s, p())

    # ===== 10 Protocol =====
    s = blank(prs)
    topbar(s, "B", "어떻게 검증하나", "랜덤으로 나누지 않고, 진짜 바깥 구간을 먼저 잡는다")
    pic(s, "protocol_flow.png", 60, 160, 1160, 320)
    add_text(s, 90, 520, 1100, 50, "원칙: 검증 구간만으로 모델 고르기 · 테스트 정답은 보지 않기 · 개발 결과와 확인 실험은 따로 쓰기", 15, C["crimson"], True)
    foot(s, p())

    # ===== 11 Main results chart =====
    s = blank(prs)
    topbar(s, "C", "핵심 실험 결과", "같은 구조로 Sunwoda · RWTH · MICH")
    pic(s, "main_ablation_bars.png", 40, 140, 780, 400)
    metric = [
        ("평균", "0.842", "최고 0.934", C["teal"]),
        ("최고", "0.934", "Sunwoda", C["blue"]),
        ("최저", "0.751", "MICH (회복)", C["orange"]),
    ]
    for i, (name, val, sub, col) in enumerate(metric):
        y = 155 + i * 115
        rect(s, 860, y, 340, 100, C["pale"], None, True)
        rect(s, 860, y, 8, 100, col)
        add_text(s, 885, y + 12, 300, 22, name, 13, C["ink"], True)
        add_text(s, 885, y + 38, 300, 32, val, 26, col, True)
        add_text(s, 885, y + 72, 300, 20, sub, 12, C["grey"])
    add_text(
        s,
        90,
        555,
        1100,
        45,
        "논문 주표(개발 3데이터). 고정 경계형 최저 0.468 → SAAR 최저 0.751. 확증 cohort는 별도.",
        13,
        C["crimson"],
        True,
    )
    foot(s, p())

    # ===== 11a Status update (Sep 2026) =====
    s = blank(prs)
    topbar(s, "C", "최신 업데이트 (2026-09)", "무엇이 바뀌었고, 무엇을 아직 주장하지 않나")
    rows = [
        ("유지", "SAAR / dual-scale BQ", "Sunwoda·RWTH·MICH 0.934 / 0.842 / 0.751\n논문 1편의 주 executor", C["teal"]),
        ("확장", "Na / Zn-ion", "BQ만으로는 Zn untouched 실패 →\nRBF-regime·refit으로 개발 점수 회복", C["blue"]),
        ("교차", "XJTU · FEMTO · Milling", "구조 복구·입력 감사 단계\n통합 우월성 주장 전 아님", C["orange"]),
        ("경계", "PAE / 저널", "결과 표는 PP만 · PAE 라우팅 실험 없음\n분야 Q1–Q2가 현실 본선", C["navy"]),
    ]
    for i, (tag, title, body, col) in enumerate(rows):
        x = 70 + (i % 2) * 590
        y = 155 + (i // 2) * 230
        rect(s, x, y, 560, 205, C["white"], C["line"], True)
        rect(s, x, y, 560, 48, col)
        add_text(s, x + 20, y + 12, 80, 24, tag, 14, C["white"], True)
        add_text(s, x + 110, y + 12, 420, 24, title, 15, C["white"], True)
        add_text(s, x + 24, y + 70, 510, 110, body, 15, C["ink"])
    foot(s, p())

    # ===== 11a2 Zn-ion path =====
    s = blank(prs)
    topbar(s, "C", "Zn-ion에서 무엇이 달라졌나", "실패 → 경로 분기 → 개발 개선 (확증 아님)")
    steps = [
        ("1", "BQ-PP", "untouched Zn\n실패 (≈ −0.38)", C["red"]),
        ("2", "RBF-regime", "장수명 memory\n+ BQ 혼합", C["orange"]),
        ("3", "α / refit", "강한 shrinkage +\nfull-dev refit", C["blue"]),
        ("4", "개발 점수", "Zn ≈ 0.91\nNa ≈ 0.82", C["teal"]),
    ]
    for i, (sn, sh, sb, scol) in enumerate(steps):
        x = 70 + i * 300
        rect(s, x, 180, 275, 220, C["white"], C["line"], True)
        rect(s, x, 180, 275, 52, scol)
        add_text(s, x + 18, 192, 40, 28, sn, 18, C["white"], True)
        add_text(s, x + 55, 196, 200, 28, sh, 16, C["white"], True)
        add_text(s, x + 20, 260, 235, 110, sb, 16, C["ink"])
        if i < 3:
            arrow(s, x + 278, 270, 18, 22, C["line"])
    rect(s, 70, 440, 1140, 140, C["wash"], None, True)
    add_text(
        s,
        95,
        460,
        1090,
        100,
        "주의  ·  0.91/0.82는 이미 본 test에 대한 사후 개발 결과. 새 untouched 확증으로 쓰지 않음.\n"
        "개선의 큰 축은 새 NN 구조만이 아니라 ridge shrinkage와 validation을 포함한 standard refit 계약.\n"
        "RBF+BQ 혼합은 margin=0 경계를 항상 보장하지 않음 → 다음 후보: quotient 안 결합(Dynamic Boundary-Scale).",
        14,
        C["ink"],
    )
    foot(s, p())

    # ===== 11a3 Why numbers moved =====
    s = blank(prs)
    topbar(s, "C", "점수가 오른 이유 — 정직하게", "구조 혁신 vs 학습 계약·수축")
    cards = [
        ("구조로 설명되는 것", "SAAR dual-scale로\nMICH 최저점 회복\n(고정 bound 0.468→0.751)\n\nZn에서 lifetime 경로\n추가로 BQ 단독 실패 보완", C["teal"]),
        ("계약·조율로 오른 것", "α=1000 affine shrinkage\nfull-development refit\n(같은 epoch로 val prefix 포함)\n\n같은 refit을 MLP에도\n줘야 공정 비교", C["orange"]),
        ("아직 분리 안 된 것", "RBF memory 정보량 vs\nNN residual 기여\n\n공정 NN 대조·nested\nunit CV·표본 수", C["red"]),
    ]
    for i, (h, b, a) in enumerate(cards):
        card(s, 70 + i * 390, 175, 370, 340, h, b, a)
    add_text(
        s,
        90,
        545,
        1100,
        40,
        "발표/논문 멘트: ‘모델이 통째로 한 단계 진화’보다 ‘어디서 구조가, 어디서 계약이 이득인지’를 분리한다.",
        14,
        C["crimson"],
        True,
    )
    foot(s, p())

    # ===== 11a4 Cross-domain =====
    s = blank(prs)
    topbar(s, "C", "교차 도메인 현황", "배터리 밖 — 복구·감사 중, 통합 성공 아님")
    headers = ["도메인", "상태", "핵심 수치/사실", "주장"]
    rows = [
        ["XJTU", "개발 개선", "progress+scale transport ≈ 0.257\n(기존 Ridge −0.84)", "post-test 개발. 독립 확증 아님"],
        ["FEMTO", "판정 보류/실패", "잘못된 진동 채널 발견\n교정 후 PP 구조들 채택 실패", "우월 실패가 아니라 설계 미성숙"],
        ["Milling", "감사", "일부 양수 점수는 수식 보정\nNN 경로 꺼진 경우 있음", "NN 기여와 보정 기여 분리 필요"],
        ["정보 한계", "이론", "비슷한 prefix + 다른 Z면\n어떤 f(H)도 한계", "안 보이는 lifetime을 복원하지 않음"],
    ]
    xs = [60, 220, 420, 860]
    ws = [150, 180, 420, 340]
    for i, h in enumerate(headers):
        add_text(s, xs[i], 150, ws[i], 24, h, 13, C["navy"], True)
    for ri, row in enumerate(rows):
        y = 185 + ri * 85
        bg = C["wash"] if ri % 2 == 0 else C["white"]
        rect(s, 50, y - 8, 1180, 75, bg, C["line"], True)
        for i, val in enumerate(row):
            add_text(s, xs[i], y + 5, ws[i], 60, val, 13, C["ink"])
    foot(s, p())

    # ===== 11b Ablation overview =====
    s = blank(prs)
    topbar(s, "C", "Ablation — 무엇이 성능을 만드나", "구성요소 on/off (개발 3데이터 + 대표 설정)")
    headers = ["기능", "제거하면", "켠 효과(요지)", "해석"]
    rows = [
        ["NN residual", "affine only", "MICH −3.3→0.47 등", "직선만으로는 부족"],
        ["Affine 동결", "trainable 경계NN", "세 데이터 모두 ↑", "NN이 tail 덮는 것 억제"],
        ["Residual bound", "unbounded", "Sun/RWTH ↑, MICH ↓", "조건부 — dual-scale로 보완"],
        ["Dual-scale gate", "고정 bound", "MICH 0.47→0.75", "이질 regime에서 용량 확대"],
        ["Rate history", "margin만", "Sun/RWTH 큰 ↑", "MICH는 단순 history가 유리한 반례"],
        ["Transport/gate", "always-on", "HUST·MATRb2 ↑", "증거 없으면 모듈 거절"],
    ]
    xs = [70, 280, 520, 820]
    ws = [200, 220, 280, 360]
    for i, h in enumerate(headers):
        add_text(s, xs[i], 150, ws[i], 24, h, 12, C["navy"], True)
    for ri, row in enumerate(rows):
        y = 185 + ri * 55
        bg = C["wash"] if ri % 2 == 0 else C["white"]
        rect(s, 60, y - 8, 1160, 50, bg, C["line"], True)
        for i, val in enumerate(row):
            add_text(s, xs[i], y, ws[i], 32, val, 12, C["ink"], i == 0)
    add_text(
        s,
        90,
        540,
        1100,
        55,
        "주장: ‘모든 부품이 모든 데이터에서 항상 이긴다’가 아니라,\n"
        "frozen affine + residual이 핵심이고, bound·history·transport는 regime에 따라 승인한다.",
        13,
        C["crimson"],
        True,
    )
    foot(s, p())

    # ===== 11b2 Gate on/off =====
    s = blank(prs)
    topbar(s, "C", "게이트 효과는 어떻게 확인하나", "넣었다 뺐다(on/off) = 표준 ablation")
    card(
        s,
        70,
        155,
        380,
        280,
        "하는 방식",
        "같은 split·같은 모델에서\n① 게이트 끔(always-on)\n② 게이트 켬(val 증거 있을 때만)\n차이를 비교",
        C["teal"],
    )
    card(
        s,
        470,
        155,
        380,
        280,
        "우리 결과",
        "13개 설정 감사:\n개선 3 · 유지 10 · 악화 0\n항상 켜면 Virkler 등 악화\n(−5.37→−6.56)",
        C["orange"],
    )
    card(
        s,
        870,
        155,
        340,
        280,
        "해석",
        "게이트는 평균 R²를\n직접 올리는 부품보다\n틀린 모듈을 막아\n기본 PP를 지키는 역할",
        C["navy"],
    )
    add_text(
        s,
        90,
        460,
        1100,
        90,
        "보통 맞음: 효과 주장은 on/off(또는 matched arm)로 확인한다.\n"
        "다만 ‘모든 shift를 잡는다’까지는 못 씀 — MICH history는 val이 full을 골랐는데 test는 단순 history가 더 나은 반례.\n"
        "원문: FINAL_PP_COMPONENT_ABLATION_RESULTS_KO.md (Evidence gate ablation)",
        13,
        C["ink"],
        True,
    )
    foot(s, p())

    # ===== 11b3 Why condition fit =====
    s = blank(prs)
    topbar(s, "C", "조건 적합도가 왜 필요한가", "밖의 점수를 미리 알 수는 없다")
    card(
        s,
        70,
        160,
        370,
        300,
        "할 수 없는 것",
        "테스트 정답을 보기 전에\n‘이번엔 R²가 몇이다’를\n맞춰 맞히는 일",
        C["red"],
    )
    card(
        s,
        460,
        160,
        370,
        300,
        "할 수 있는 것",
        "관측만으로 검사:\n이 문제에 SAAR를\n써도 되는가?\n안 되면 숫자를 내지 않기",
        C["teal"],
    )
    card(
        s,
        850,
        160,
        360,
        300,
        "최종 출력",
        "예측 숫자\n+\n적용 가능 증명\n+\n거리·불확실성",
        C["navy"],
    )
    add_text(
        s,
        90,
        490,
        1100,
        80,
        "한 줄: 적합도는 ‘잘 맞힐 보증서’가 아니라 ‘이 가정으로 말해도 되는가’의 허가증이다.\n"
        "허가 없이 점수만 내면, 외삽에서 틀린 확신을 주는 것과 같다.",
        14,
        C["crimson"],
        True,
    )
    foot(s, p())

    # ===== 11b4 Applicability checks (clearer) =====
    s = blank(prs)
    topbar(s, "C", "조건 적합도 6가지 검사", "쉬운 말로 — 무엇을 / 왜 보나")
    checks = [
        ("① 문제 타입", "무엇을", "안 본 개체의\n학습 밖 남은수명인가?", "왜", "랜덤 미래예측과\n문제가 다르기 때문", C["teal"]),
        ("② 관측 계약", "무엇을", "경계·이력·방향이\n실제로 보이는가?", "왜", "없는 prior를 켜면\n가정이 거짓이 됨", C["orange"]),
        ("③ Val 기술", "무엇을", "검증 끝단에서\n이미 도움이 되나?", "왜", "test 보기 전에\n실력 증거를 남김", C["blue"]),
        ("④ 거리·안정", "무엇을", "너무 멀거나\n재학습이 흔들리나?", "왜", "멀수록 가정 의존↑\n불안정하면 위험", C["navy"]),
        ("⑤ 경로 호환", "무엇을", "검증→시험 방향이\n비슷한가?", "왜", "반대면 보정을\n옮기면 안 됨", C["crimson"]),
        ("⑥ 메커니즘", "무엇을", "재료·고장·센서\n의미가 같은가?", "왜", "다르면 모델이\n다른 현상을 봄", C["green"]),
    ]
    for i, (title, a, av, b, bv, col) in enumerate(checks):
        x = 55 + (i % 3) * 400
        y = 145 + (i // 3) * 235
        rect(s, x, y, 385, 220, C["white"], C["line"], True)
        rect(s, x, y, 385, 38, col)
        add_text(s, x + 12, y + 7, 360, 26, title, 14, C["white"], True)
        add_text(s, x + 12, y + 50, 70, 24, a, 12, col, True)
        add_text(s, x + 80, y + 48, 290, 70, av, 13, C["ink"])
        add_text(s, x + 12, y + 130, 70, 24, b, 12, C["grey"], True)
        add_text(s, x + 80, y + 128, 290, 70, bv, 13, C["grey"])
    foot(s, p())

    # ===== 11b5 Fit decision clearer =====
    s = blank(prs)
    topbar(s, "C", "적합도 판정 — 세 갈래", "Pass / Weak / Fail")
    steps = [
        ("Pass", "써도 된다", "6항이 대체로 통과\nval에서 실력도 확인", "→ SAAR 예측을 보고\n거리·불확실성과 함께 제시", C["teal"]),
        ("Weak", "조심해서", "문제는 맞지만\n거리 큼·증거 얇음", "→ 보정 줄이거나\n직선(identity)만 유지", C["orange"]),
        ("Fail", "쓰지 않는다", "메커니즘이 다르거나\nval에서 이미 실패", "→ ABSTAIN\n숫자를 내지 않음", C["red"]),
    ]
    for i, (tag, sub, mid, out, col) in enumerate(steps):
        x = 70 + i * 390
        rect(s, x, 150, 370, 340, C["white"], col, True)
        rect(s, x, 150, 370, 70, col)
        add_text(s, x + 15, 158, 340, 28, tag, 20, C["white"], True, "center")
        add_text(s, x + 15, 188, 340, 24, sub, 13, C["white"], True, "center")
        add_text(s, x + 20, 245, 330, 90, mid, 14, C["ink"], False, "center")
        add_text(s, x + 20, 360, 330, 100, out, 14, col, True, "center")
    add_text(
        s,
        90,
        520,
        1100,
        70,
        "사례: milling 재료 1→2 → Fail(정답 없이 거절).  ·  plain MICH → coverage만으로 Pass 하면 안 됨(관계 이동).\n"
        "Sunwoda는 잘 나와도, 게이트를 엄하게 두면 Weak로 보수 운용 가능. 적합도 = 정확도 보증 아님.",
        13,
        C["ink"],
        True,
    )
    foot(s, p())

    # ===== 11b6 Everyday analogy =====
    s = blank(prs)
    topbar(s, "C", "비유로 정리", "발표 멘트용")
    card(
        s,
        70,
        160,
        560,
        320,
        "운전 비유",
        "SAAR는 ‘고속도로용 차선 유지’ 장치.\n비포장·반대차선이면 장치를 켜지 않고\n멈추거나 천천히 간다.\n\n적합도 검사 = 지금이 고속도로인지 확인.",
        C["teal"],
    )
    card(
        s,
        660,
        160,
        550,
        320,
        "논문에서 말할 문장",
        "We do not claim accuracy before seeing labels.\nWe claim a test-label-free applicability check:\nuse SAAR only when the declared contract,\nvalidation skill, and shift compatibility hold;\notherwise abstain.",
        C["navy"],
    )
    add_text(
        s,
        90,
        510,
        1100,
        60,
        "연결: 게이트 on/off 실험은 이 허가증을 ‘항상 킴’ vs ‘검사 후 킴’으로 비교한 것.\n"
        "원문: APPLICABILITY_NOVELTY_LIMITS_KO.md",
        13,
        C["grey"],
        True,
    )
    foot(s, p())

    # ===== 11c Ablation numbers =====
    s = blank(prs)
    topbar(s, "C", "Ablation 수치 (matched)", "Sunwoda / RWTH / MICH ensemble R²")
    headers = ["Arm", "Sunwoda", "RWTH", "MICH"]
    rows = [
        ["Direct NN (구조 없음)", "−1.35", "0.63", "0.68"],
        ["Affine only", "0.28", "0.66", "−3.34"],
        ["Trainable hard-boundary", "0.90", "0.86", "0.32"],
        ["Frozen + unbounded", "0.72", "0.79", "0.76"],
        ["BQ bounded (고정)", "0.94", "0.88", "0.47"],
        ["최종 dual-scale SAAR", "0.93", "0.84", "0.75"],
    ]
    xs = [100, 420, 650, 880]
    for i, h in enumerate(headers):
        add_text(s, xs[i], 155, 200, 24, h, 13, C["navy"], True)
    for ri, row in enumerate(rows):
        y = 195 + ri * 52
        bg = C["wash"] if ri == 5 else (C["white"] if ri % 2 else C["pale"])
        rect(s, 80, y - 8, 1120, 48, bg, C["teal"] if ri == 5 else C["line"], True)
        for i, val in enumerate(row):
            add_text(s, xs[i], y, 220, 30, val, 13, C["teal"] if ri == 5 else C["ink"], ri == 5 or i == 0)
    add_text(
        s,
        90,
        530,
        1100,
        60,
        "고정 bound는 MICH에서 unbounded보다 낮음(−0.29) → dual-scale이 worst-domain을 회복.\n"
        "원문: FINAL_PP_COMPONENT_ABLATION_RESULTS_KO.md · UNIFIED_SUPPORT_GATED_PP_RESULTS_KO.md",
        12,
        C["grey"],
        True,
    )
    foot(s, p())

    # ===== 12 MICH units =====
    s = blank(prs)
    topbar(s, "C", "MICH를 배터리 하나씩 보면", "전체 점수만 보지 않고, 유닛마다도 양의 성능을 확인")
    pic(s, "mich_units.png", 50, 145, 900, 420)
    card(s, 980, 200, 230, 280, "포인트", "8개 전부 양수\n\n가장 어려웠던\nunit 31도\n음수에서 회복\n\n관계가 바뀐\n대표 사례", C["orange"])
    foot(s, p())

    # ===== 13 Competitors =====
    s = blank(prs)
    topbar(s, "C", "다른 방법과 비교", "잘 된 8곳 · 알고리즘별 R² (TabPFN 전체 반영)")
    pic(s, "competitor_bars.png", 40, 130, 1200, 420)
    add_text(s, 90, 555, 1100, 50, "TabPFN 8곳 모두 반영. Sunwoda·RWTH는 음수, NCMAPSS는 PP와 동률권(0.934). train 상한·일부 단일 seed 보조 비교.", 12, C["grey"], True)
    foot(s, p())

    # ===== 13b Competitors bars by algo =====
    s = blank(prs)
    topbar(s, "C", "주요 알고리즘 막대 비교", "PP / V-REx / GroupDRO / LinRBF / Engression / TabPFN")
    pic(s, "competitor_algo_bars.png", 40, 140, 1200, 420)
    add_text(s, 90, 575, 1100, 30, "원문: ALL_DATASET_EXTRAPOLATION_COMPETITORS · FAIR_PFN_COMPARISON · TABPFN_EXTERNAL_BATTERIES_RESULTS", 11, C["grey"], True)
    foot(s, p())

    # ===== 13c PP vs TabPFN =====
    s = blank(prs)
    topbar(s, "C", "PP와 TabPFN 직접 비교", "동일 고정 test 기준 · TabPFN은 보조 조건(행 상한 등)")
    pic(s, "pp_vs_tabpfn.png", 40, 140, 1200, 420)
    add_text(s, 90, 570, 1100, 35, "MATRb2에서는 TabPFN(0.618)이 기본 PP보다 높을 수 있어, 이 분할만으로 PP 우위를 주장하지 않는다.", 12, C["crimson"], True)
    foot(s, p())

    # ===== 14 Experiment map =====
    s = blank(prs)
    topbar(s, "C", "어디서 되고, 어디서 안 되나", "성공과 실패를 한 지도에 같이 둔다")
    pic(s, "experiment_map.png", 40, 140, 820, 430)
    card(s, 900, 170, 310, 130, "잘 되는 곳", "연속 열화·경계 prior\n검증≈테스트 방향\nSAAR 3배터리", C["teal"])
    card(s, 900, 320, 310, 130, "조심할 곳", "Zn 장수명·교차조건\n개발 점수는 있으나\n확증·공정대조 남음", C["orange"])
    card(s, 900, 470, 310, 110, "아직 안 되는 곳", "FEMTO 현 설계\n정보 한계(Z 미관측)\n→ 거절·재설계", C["red"])
    foot(s, p())

    # ===== 15 Bootstrap =====
    s = blank(prs)
    topbar(s, "C", "통계로 본 개선", "일부 긴 시계열만 좋아진 것이 아님")
    pic(s, "bootstrap_ci.png", 50, 150, 900, 400)
    card(s, 980, 200, 230, 280, "읽는 법", "세 데이터 모두\n신뢰구간이 0 아래\n\n초기화 5번 모두\n같은 방향\n\n다만 도메인 수는\n아직 적음", C["teal"])
    foot(s, p())

    # ===== 16 What is significant =====
    s = blank(prs)
    topbar(s, "C", "어디까지 말해도 될까", "과장하지 않고, 솔직하게")
    cards = [
        ("꽤 확실한 것", "SAAR로 3배터리 최저점 회복\n잘 된 cohort에서 경쟁 우위\n경계 위반 0건(해당 설정)\nPAE와 결과 표 분리", C["teal"]),
        ("조심할 것", "핵심 표는 개발 결과\nZn 0.9대·XJTU 0.26은\n사후/소표본 성격\nrefit·정보량 맞춰 비교", C["orange"]),
        ("아직 약한 것", "보편 pre-gate 실패\nFEMTO 미성숙\n통합 executor 미검증\nICML급 증거는 부족", C["red"]),
    ]
    for i, (h, b, a) in enumerate(cards):
        card(s, 80 + i * 390, 200, 360, 280, h, b, a)
    add_text(s, 90, 530, 1100, 50, "말할 때: ‘어디서나 1등’이 아니라 ‘연속 열화·경계 prior가 맞고 검증이 비슷할 때 SAAR가 안전장치로 작동한다’.", 14, C["ink"], True)
    foot(s, p())

    # ===== 16b Journal target =====
    s = blank(prs)
    topbar(s, "D", "논문으로 어디를 노리나", "지금 증거 강도 기준")
    tiers = [
        ("현실 본선", "RESS · MSSP ·\nIEEE Reliability/TII", "분야 Q1–Q2\n외삽·RUL 스토리 적합", C["teal"]),
        ("도전", "Applied Energy 등\n배터리 특화", "청구항을 Li 연속열화로\n좁힐 때만", C["blue"]),
        ("아직 이름만", "ICML / NeurIPS\nNature 계열", "새 학습 원리·대규모\n확증이 더 필요", C["grey"]),
    ]
    for i, (h, b, c, col) in enumerate(tiers):
        card(s, 80 + i * 390, 180, 370, 280, h, f"{b}\n\n{c}", col)
    add_text(
        s,
        90,
        500,
        1100,
        80,
        "전략  ·  PP 1편은 SAAR+3배터리(+정직한 한계). Zn·베어링은 같은 편의 주 claim에 넣지 않거나\n"
        "개발/한계 절로만. ‘세계 탑’과 ‘분야 상위(Q1)’는 다른 줄.",
        14,
        C["ink"],
        True,
    )
    foot(s, p())

    # ===== 17 Claim boundary =====
    s = blank(prs)
    topbar(s, "D", "지금 / 다음 / 박사과정", "식 유무로 나눈 두 독립 논문")
    rows = [
        (
            "PP",
            "equation-free\n외삽 안전장치",
            "계약 → 고정 rule로 weak bundle\n기계적 활성화 · 수축·거절",
            C["teal"],
        ),
                (
            "PAE",
            "식 뼈대 +\nNN 부족분",
            "후보식 검증 → 제한 보정 학습\n이득 없으면 끔 · PP/거절",
            C["orange"],
        ),
        (
            "박사논문",
            "식 있음→PAE\n없음→PP",
            "교차 도메인 · 선택적 위험 ·\n미개봉 cohort 확증",
            C["navy"],
        ),
    ]
    for i, (a, b, c, col) in enumerate(rows):
        y = 185 + i * 115
        rect(s, 90, y, 180, 90, col, None, True)
        add_text(s, 105, y + 30, 150, 30, a, 18, C["white"], True, "center")
        add_text(s, 300, y + 18, 400, 55, b, 17, C["ink"], True)
        add_text(s, 720, y + 18, 460, 55, c, 14, C["grey"])
    rect(s, 90, 545, 1100, 55, None, C["crimson"], True)
    add_text(
        s,
        110,
        560,
        1060,
        28,
        "지금: PP(SAAR)  |  Zn·교차는 한계/확장  |  다음: PAE  |  prior-free라고 쓰지 말 것",
        14,
        C["crimson"],
        True,
        "center",
    )
    foot(s, p())

    # ===== 17b PAE equation + NN =====
    s = blank(prs)
    topbar(s, "D", "PAE 안에서 식과 NN", "식이 있어도 부족분은 신경망이 제한적으로 메운다")
    pic(s, "pae_equation_nn.png", 40, 140, 1200, 400)
    add_text(
        s,
        90,
        555,
        1100,
        40,
        "식 = 방향·경계·관계의 뼈대  ·  NN = 미지 파라미터·잔차·노이즈  ·  식을 덮어쓰지 않게 보정 제한",
        14,
        C["crimson"],
        True,
    )
    foot(s, p())

    # ===== 17c PP vs PAE contrast =====
    s = blank(prs)
    topbar(s, "D", "한 장 대비", "입력이 다르면 연구도 갈린다")
    pic(s, "pp_vs_pae_split.png", 40, 145, 1200, 420)
    add_text(
        s,
        90,
        575,
        1100,
        30,
        "명칭  ·  PAE: equation prior + bounded NN   ·   PP: equation-free extrapolation safeguard",
        13,
        C["grey"],
        True,
    )
    foot(s, p())

    # ===== 18 Takeaway =====
    s = blank(prs)
    topbar(s, "D", "한 장 요약", "오늘 가져갈 말")
    bullets = [
        ("왜", "학습 밖에서는 가정이 필요하다. 식만 넣으면 답이 아니다."),
        ("PP", "주모델 SAAR: equation-free 외삽 안전장치 (prior-free 아님)."),
        ("증거", "3배터리 개발 표가 본선. Zn·XJTU 숫자는 개발/한계로 분리."),
        ("PAE", "결과 표와 안 겹침. 식 컴파일·라우팅은 후속."),
        ("다음", "scale-aware / boundary-consistent quotient · 미개봉 cohort."),
    ]
    cols = [C["orange"], C["teal"], C["blue"], C["navy"], C["crimson"]]
    for i, ((h, b), col) in enumerate(zip(bullets, cols)):
        y = 170 + i * 85
        rect(s, 90, y, 160, 60, col, None, True)
        add_text(s, 105, y + 16, 130, 28, h, 14, C["white"], True, "center")
        add_text(s, 280, y + 14, 900, 35, b, 16, C["ink"])
    foot(s, p())

    # ===== 19 Discussion =====
    s = blank(prs)
    rect(s, 0, 0, 1280, 10, C["crimson"])
    rect(s, 72, 100, 85, 6, C["crimson"])
    add_text(s, 72, 130, 600, 50, "같이 정할 것", 40, C["ink"], True)
    add_text(s, 76, 200, 500, 28, "발표·논문 다듬을 때", 18, C["crimson"], True)
    qs = [
        "주표를 SAAR 3배터리만 둘지, Zn 개발 숫자를 한계 절로만 둘지",
        "XJTU 0.257·FEMTO 실패를 본문 한계로 둘지 Appendix로 뺄지",
        "투고 본선을 RESS/MSSP로 고정할지, 배터리 특화로 좁힐지",
        "미개봉 cohort·공정 refit MLP 대조를 언제 동결할지",
    ]
    for i, q in enumerate(qs):
        y = 260 + i * 70
        rect(s, 80, y, 34, 34, [C["crimson"], C["teal"], C["blue"], C["orange"]][i], None, True)
        add_text(s, 88, y + 6, 18, 20, str(i + 1), 12, C["white"], True, "center")
        add_text(s, 140, y + 4, 1000, 35, q, 16, C["ink"])
    add_text(s, 76, 580, 900, 40, "자료: pp-extrapolation · SAAR_NAMING · PP_INFORMATION_LIMITS · 외삽 문헌조사", 12, C["grey"])
    p()

    # =====================================================================
    # APPENDIX — paper-style study notes + material map
    # =====================================================================

    # A0 divider
    s = blank(prs)
    for sh in list(s.shapes):
        sh._element.getparent().remove(sh._element)
    rect(s, 0, 0, 1280, 720, C["navy"])
    add_text(s, 90, 220, 1000, 50, "APPENDIX", 18, C["orange"], True)
    add_text(s, 90, 280, 1100, 80, "공부용 상세 노트", 42, C["white"], True)
    add_text(s, 90, 390, 1000, 80, "논문처럼 문제·방법·결과·한계를 정리하고\n원문 파일 경로까지 연결한다", 20, RGBColor(0xC8, 0xD4, 0xE3))
    add_text(s, 90, 520, 900, 40, "본문(01–19)은 발표용 · 여기서부터는 복습·원고용", 15, C["orange"])
    p()

    # A1 map
    s = blank(prs)
    topbar(s, "Apx", "자료 지도 — 어디까지 연결되나", "발표 → 실험 원문 → 문헌조사까지 한줄로")
    rows = [
        ("발표 본문", "ppt/pp/PP_Research_Detailed_v2.pptx", "스토리·그래프·한 줄 메시지"),
        ("실험 저장소", "pp-extrapolation/", "프로토콜·결과 md·코드·json"),
        ("문헌 공부", "ppt/외삽자료/ (literec_study)", "50분 발표·통합 ppt·paper_pdfs"),
        ("프레임 초안", "pp_pae_total/ · v7 ppt", "PP–PAE 큰 그림"),
    ]
    for i, (a, b, c) in enumerate(rows):
        y = 170 + i * 90
        rect(s, 90, y, 220, 70, C["navy"], None, True)
        add_text(s, 105, y + 20, 190, 30, a, 15, C["white"], True)
        add_text(s, 340, y + 10, 520, 50, b, 14, C["ink"], True)
        add_text(s, 880, y + 18, 320, 40, c, 13, C["grey"])
    add_text(s, 90, 560, 1100, 40, "공부 순서 추천: 외삽 문헌조사 통합 ppt → 본문 A~D → 아래 부록 표 → 해당 md 원문", 14, C["teal"], True)
    foot(s, p())

    # A2 abstract
    s = blank(prs)
    topbar(s, "Apx", "논문식 요약 (Abstract 초안)", "PP가 말하는 한 단락")
    rect(s, 90, 160, 1100, 420, C["wash"], None, True)
    add_text(
        s,
        120,
        185,
        1040,
        360,
        "남은수명(RUL) 예측기는 학습 구간 안에서는 잘 맞아도, 관측된 열화 범위 밖에서는\n"
        "행동이 통제되지 않는 경우가 많다. 본 연구(SAAR)는 안정적인 기본 경향(고정 affine)과\n"
        "support-aware dual-scale residual을 분리해 equation-free 외삽 안전장치로 둔다.\n"
        "경계가 관측되면 Boundary-Quotient로 고장점 RUL=0을 강제한다. 주 개발 표는\n"
        "Sunwoda·RWTH·MICH이며, Zn-ion·베어링 확장은 한계·개발 결과로 분리 보고한다.\n"
        "PAE(식 컴파일)와는 결과 표를 공유하지 않는다.",
        16,
        C["ink"],
    )
    add_text(s, 120, 560, 1040, 30, "원문: PP_FIRST_PUBLICATION_CLAIMS_KO · UNIFIED_DUAL_SCALE · PP_INFORMATION_LIMITS", 12, C["grey"])
    foot(s, p())

    # A3 problem formulation
    s = blank(prs)
    topbar(s, "Apx", "문제 정의 (Methods)", "무엇을 외삽이라고 부르는가")
    cards = [
        ("1. 개체 분리", "같은 배터리·시편이\n학습과 테스트에\n동시에 들어가지 않음", C["orange"]),
        ("2. 범위 밖만 평가", "학습이 본 열화 좌표의\n바깥(끝단)만 테스트\n안쪽은 점수에 안 넣음", C["teal"]),
        ("3. 정답 미사용 선택", "설정·보정은\n검증 구간만으로\n테스트 정답은 안 봄", C["blue"]),
        ("4. 보고 지표", "주: 전체 설명력 R²\n보조: 유닛별·재학습\n경계 위반 횟수", C["navy"]),
    ]
    for i, (h, b, a) in enumerate(cards):
        card(s, 70 + i * 295, 175, 280, 280, h, b, a)
    add_text(s, 90, 500, 1100, 55, "원문: MODEL_AND_SPLIT_KO.md · ADDITIONAL_REAL_DATASETS_PROTOCOL.md · DISTANCE_SHELL_PROTOCOL.md", 13, C["grey"])
    add_text(s, 90, 555, 1100, 35, "핵심: 일반 ‘미래 예측’과 다르게, 처음 보는 개체의 학습 밖 구간만 본다.", 15, C["ink"], True)
    foot(s, p())

    # A3b dataset catalog
    s = blank(prs)
    topbar(s, "Apx", "데이터셋 한눈에", "도메인 · 무엇을 추론하나 · 외삽 타입")
    rows = [
        ("HUST", "배터리", "RUL", "안 본 프로토콜의\n깊은 미래/끝단", "잘 됨 · 0.96"),
        ("Virkler", "균열", "잔여 수명", "안 본 시편의\n균열 성장 끝단", "잘 됨 · 0.89"),
        ("NASA bat.", "배터리", "RUL", "건강도 끝단\n(LOBO 집계)", "잘 됨 · 0.58"),
        ("Sunwoda", "배터리", "RUL", "안 본 셀\n늦은 health", "잘 됨 · 0.87"),
        ("RWTH", "배터리", "RUL", "안 본 셀\n늦은 health", "잘 됨 · 0.74"),
        ("MICH", "배터리", "RUL", "안 본 셀\n새 regime", "통합 후 0.75"),
        ("MATR19/b2", "배터리", "RUL", "안 본 셀\n먼/가까운 꼬리", "보정 후 양수"),
        ("N-CMAPSS", "엔진", "RUL", "안 본 엔진\n× 고부하 × 만기", "잘 됨 · 0.94"),
    ]
    headers = ["데이터", "도메인", "추론", "스플릿·외삽", "결과 요지"]
    xs = [70, 220, 380, 520, 900]
    ws = [140, 150, 130, 360, 280]
    for i, h in enumerate(headers):
        add_text(s, xs[i], 145, ws[i], 22, h, 11, C["navy"], True)
    for ri, row in enumerate(rows):
        y = 175 + ri * 48
        bg = C["wash"] if ri % 2 == 0 else C["white"]
        rect(s, 60, y - 6, 1160, 44, bg, C["line"], True)
        for i, val in enumerate(row):
            col = C["teal"] if i == 4 else C["ink"]
            add_text(s, xs[i], y, ws[i], 32, val, 11, col, i == 0)
    add_text(s, 90, 575, 1100, 30, "실패 축(별도): XJTU · FEMTO · NASA milling — 아래 ‘언제 되나’ 슬라이드 참고", 12, C["crimson"], True)
    foot(s, p())

    # A3b2 what each dataset is
    s = blank(prs)
    topbar(s, "Apx", "각 데이터셋이 무엇인가", "출처·측정·왜 RUL 문제인가")
    infos = [
        ("HUST", "리튬배터리 수명\n서로 다른 충전 프로토콜\n용량 궤적 → 남은 cycle", C["teal"]),
        ("Virkler", "금속 피로 균열\n시편별 균열 성장\n임계 길이까지 잔여", C["orange"]),
        ("NASA bat.", "공개 배터리 aging\n셀별 용량 fade\n건강도 끝단 RUL", C["blue"]),
        ("Sunwoda", "상용 셀 cohort\nearly/late health\n안 본 셀 늦은 구간", C["navy"]),
        ("RWTH", "실험실 배터리\n온도·조건 다양\n안 본 셀 끝단", C["crimson"]),
        ("MICH", "배터리 새 regime\n관계 이동이 큼\n통합 SAAR 회복 사례", C["green"]),
        ("MATR", "대규모 셀 데이터\nbatch/연도 분할\n먼·가까운 health 꼬리", C["orange"]),
        ("N-CMAPSS", "항공기 엔진 시뮬\n부하·수명 교차\n안 본 엔진×고부하", C["teal"]),
    ]
    for i, (name, body, col) in enumerate(infos):
        x = 70 + (i % 4) * 300
        y = 155 + (i // 4) * 230
        rect(s, x, y, 285, 210, C["white"], C["line"], True)
        rect(s, x, y, 285, 40, col)
        add_text(s, x + 12, y + 8, 260, 28, name, 15, C["white"], True)
        add_text(s, x + 12, y + 55, 260, 140, body, 13, C["ink"])
    foot(s, p())

    # A3b3 how each was split
    s = blank(prs)
    topbar(s, "Apx", "데이터마다 어떻게 쪼갰나", "공통 규칙 + 데이터별 외삽 좌표")
    splits = [
        ("HUST", "Train: 본 프로토콜 unit\nTest: 안 본 프로토콜\n좌표: 깊은 미래 health\nhull-out 100%", C["teal"]),
        ("Virkler", "Train/Test: 시편 분리\n좌표: 균열 길이 끝단\n성장 방향 유지 구간", C["orange"]),
        ("NASA", "Leave-one-battery-out\n좌표: 용량/health 끝단\nfold 예측 후 pooled", C["blue"]),
        ("Sunwoda", "Early-life unit → train\nLate-health 안 본 셀 → test\n좌표: health (멀~4.4SD)", C["navy"]),
        ("RWTH", "Unit 분리 + late health\nval은 pseudo-tail\n좌표: health (~2.9SD)", C["crimson"]),
        ("MICH", "Unit 분리 + late\nmargin 변동이 train 밖\n관계 이동 스트레스", C["green"]),
        ("MATR19/b2", "Unseen cell tail\n2019=먼 꼬리 / b2=가까움\n출력 transport 검토", C["orange"]),
        ("N-CMAPSS", "Unseen engine\n× high TRA × late life\n짧은 조건 hull 밖", C["teal"]),
    ]
    for i, (name, body, col) in enumerate(splits):
        x = 70 + (i % 4) * 300
        y = 150 + (i // 4) * 235
        rect(s, x, y, 285, 220, C["white"], C["line"], True)
        rect(s, x, y, 8, 220, col)
        add_text(s, x + 20, y + 12, 250, 28, name, 14, C["ink"], True)
        add_text(s, x + 20, y + 50, 250, 155, body, 12, C["grey"])
    foot(s, p())

    # A3c common split figure
    s = blank(prs)
    topbar(s, "Apx", "어떻게 나누나 (공통 규칙)", "데이터셋 이름이 아니라 계약·hull로 고정")
    pic(s, "dataset_split.png", 40, 145, 1200, 400)
    add_text(
        s,
        90,
        560,
        1100,
        40,
        "원문: MODEL_AND_SPLIT_KO.md · ADDITIONAL_REAL_DATASETS_PROTOCOL.md",
        12,
        C["grey"],
        True,
    )
    foot(s, p())

    # A3d what we infer + when it works
    s = blank(prs)
    topbar(s, "Apx", "무엇을 추론하고, 언제 잘 되나", "타깃 · 입력 · 성공/실패 지도")
    pic(s, "dataset_landscape.png", 40, 140, 1200, 430)
    add_text(
        s,
        90,
        580,
        1100,
        30,
        "말할 때: ‘어디서나 1등’이 아니라 ‘경계·방향이 맞고 val/test 이동이 비슷할 때’.",
        13,
        C["crimson"],
        True,
    )
    foot(s, p())

    # A3e per-dataset detail positive
    s = blank(prs)
    topbar(s, "Apx", "잘 된 설정 — 스플릿·입력·해석", "양의 8곳 (요약)")
    details = [
        ("HUST", "스플릿: 안 본 프로토콜 unit · hull-out 100% · 거리~1.6", "추론: 용량·이력 → RUL (깊은 끝단)", "왜: 경계·방향 안정 · PP 0.958"),
        ("Virkler", "스플릿: 안 본 시편 · 균열 끝단 · hull-out 100%", "추론: 균열 좌표·성장 → 잔여 수명", "왜: 성장 방향과 맞을 때 · 0.888"),
        ("NASA bat.", "스플릿: 건강도 끝단 · LOBO 집계 · 거리~2.0", "추론: 용량 궤적 → RUL", "왜: 양수이나 격차 작음 · 0.584"),
        ("Sunwoda", "스플릿: early unit → late-health cell · 거리~4.4", "추론: health/rate window → RUL", "왜: 먼 꼬리에서도 양수 · 0.865"),
        ("RWTH", "스플릿: unit 분리 + 늦은 health · 거리~2.9", "추론: 동일 입력 계약 → RUL", "왜: 통합·보정 후 · 0.743"),
        ("N-CMAPSS", "스플릿: 안 본 엔진 × 고부하 × late · 짧은 거리", "추론: 조건·센서 요약 → cycle RUL", "왜: 짧은 조건 외삽 · 0.937"),
    ]
    for i, (name, split, infer, why) in enumerate(details):
        x = 70 + (i % 3) * 390
        y = 155 + (i // 3) * 230
        rect(s, x, y, 370, 210, C["white"], C["line"], True)
        rect(s, x, y, 370, 40, C["teal"] if i < 3 else C["blue"])
        add_text(s, x + 14, y + 8, 340, 28, name, 16, C["white"], True)
        add_text(s, x + 14, y + 55, 340, 45, split, 12, C["ink"])
        add_text(s, x + 14, y + 110, 340, 40, infer, 12, C["grey"])
        add_text(s, x + 14, y + 160, 340, 35, why, 13, C["teal"], True)
    foot(s, p())

    # A3f MATR / MICH / failures detail
    s = blank(prs)
    topbar(s, "Apx", "특수·실패 설정 — 왜 그런가", "같은 규칙으로도 안 되는 경우")
    cards = [
        ("MATR2019 / batch2", "안 본 셀의 health 꼬리\n2019는 먼 꼬리(~5.6 SD)\nbatch2는 상대적으로 가까움", "기본 PP 약함 →\n출력 보정 후 0.466\nbatch2는 양수이나\nNN/TabPFN과 경합", C["orange"]),
        ("MICH (통합)", "안 본 셀 + 새 regime\nmargin 변동이 train 밖", "기본 PP −1.52 실패\n경계·dual-scale로\n0.751 · 8/8 unit +", C["teal"]),
        ("XJTU / FEMTO", "베어링·미접촉 cohort\n끝점 희소 / 방향 불일치", "전 방법 음수\n옮기지 않거나\n거절이 정직한 답", C["red"]),
        ("NASA milling", "재료 1→2 전이\n검증 unit 극소", "메커니즘 게이트가\n실패를 사전 탐지\n점수 −4.8", C["navy"]),
    ]
    for i, (h, a, b, col) in enumerate(cards):
        x = 70 + (i % 2) * 590
        y = 155 + (i // 2) * 230
        rect(s, x, y, 560, 210, C["white"], C["line"], True)
        rect(s, x, y, 8, 210, col)
        add_text(s, x + 24, y + 16, 510, 28, h, 16, C["ink"], True)
        add_text(s, x + 24, y + 60, 510, 55, a, 13, C["grey"])
        add_text(s, x + 24, y + 125, 510, 70, b, 14, C["ink"], True)
    foot(s, p())

    # A3 ablation appendix mirror
    s = blank(prs)
    topbar(s, "Apx", "Ablation 요약 (공부용)", "본문 C절과 동일 메시지")
    bullets = [
        "핵심: frozen affine + nonlinear residual (둘 다 필요)",
        "Fixed bound: Sun/RWTH ↑, MICH ↓ → dual-scale로 worst 회복",
        "Full rate history: Sun/RWTH 필수, MICH는 단순 history 반례",
        "Transport: HUST +0.13, MATRb2 +0.19 수준 기여",
        "Evidence gate: 13설정에서 악화 0 — 틀린 모듈 거절이 역할",
        "원문: FINAL_PP_COMPONENT_ABLATION_RESULTS_KO.md",
    ]
    for i, t in enumerate(bullets):
        y = 170 + i * 60
        rect(s, 90, y, 18, 18, C["teal"] if i < 5 else C["orange"], None, True)
        add_text(s, 130, y - 4, 1050, 40, t, 15, C["ink"])
    foot(s, p())

    # A3 fit conditions appendix
    s = blank(prs)
    topbar(s, "Apx", "조건 적합도 체크리스트", "허가증이지 성적표가 아니다")
    rows = [
        ["① 문제", "안 본 개체 + 학습 밖 RUL인가?", "같은 개체 미래만 / 랜덤 split"],
        ["② 계약", "쓰는 prior에 필요한 관측이 있나?", "없는 경계·방향 강제"],
        ["③ Val", "검증 끝단에서 이미 도움이 되나?", "test만 보고 채택"],
        ["④ 안정", "거리·seed가 허용 범위인가?", "너무 멀거나 재학습 붕괴"],
        ["⑤ 호환", "검증→시험 이동이 비슷한가?", "방향 반대·관계 급변"],
        ["⑥ 메커니즘", "재료·고장·센서 의미가 같은가?", "다른 현상을 같은 모델에"],
    ]
    headers = ["검사", "통과 조건 (쉬운 말)", "실패하면"]
    xs = [70, 250, 780]
    for i, h in enumerate(headers):
        add_text(s, xs[i], 148, 480, 24, h, 13, C["navy"], True)
    for ri, row in enumerate(rows):
        y = 185 + ri * 52
        rect(s, 60, y - 8, 1160, 48, C["wash"] if ri % 2 == 0 else C["white"], C["line"], True)
        for i, val in enumerate(row):
            col = C["teal"] if i == 1 else (C["red"] if i == 2 else C["ink"])
            add_text(s, xs[i], y, 500 if i else 160, 30, val, 12, col, i == 0)
    add_text(
        s,
        90,
        520,
        1100,
        55,
        "판정: Pass=예측 사용 · Weak=보정 축소/identity · Fail=ABSTAIN.\n"
        "출력은 숫자만이 아니라 적용증명·거리·불확실성을 함께. 원문: APPLICABILITY_NOVELTY_LIMITS_KO.md",
        12,
        C["grey"],
        True,
    )
    foot(s, p())

    # A4 method detail
    s = blank(prs)
    topbar(s, "Apx", "방법 상세 (최종 PP)", "수식 하나 + 공용 구조")
    add_text(s, 90, 155, 1100, 36, "ŷ = ŷ_affine + w(d) · B tanh(r_θ(x)/B)", 20, C["ink"], True, "center")
    left = [
        "ŷ_affine: 관측 밖에서도 폭주하지 않는 기본 추세",
        "r_θ(x): history에서 학습한 비선형 보정",
        "B tanh(·/B): 보정 크기 상한",
        "w(d): support에서 멀수록 NN 보정 감쇠",
        "공용: affine + bounded residual + gate",
        "데이터별: 계수·NN·범위·gate 강도만 학습",
    ]
    for i, t in enumerate(left):
        y = 210 + i * 48
        rect(s, 90, y + 8, 10, 10, C["teal"], None, True)
        add_text(s, 115, y, 520, 35, t, 14, C["ink"])
    right = [
        "배터리 열화식 사전 주입 아님",
        "도메인별은 입력 표현만 다름",
        "재학습 시드 42–46 (5회)",
        "검증 MSE로 설정·조기종료",
        "주 점수: 테스트 전체 R²",
        "경계·보정 위반: 0/17,645",
    ]
    for i, t in enumerate(right):
        y = 210 + i * 48
        rect(s, 680, y + 8, 10, 10, C["orange"], None, True)
        add_text(s, 705, y, 500, 35, t, 14, C["ink"])
    add_text(s, 90, 560, 1100, 30, "원문: UNIFIED_SUPPORT_GATED_PP_RESULTS_KO.md · BOUNDARY_QUOTIENT_PP_RESULTS_KO.md", 12, C["grey"])
    foot(s, p())

    # A5 main table
    s = blank(prs)
    topbar(s, "Apx", "표 1. 통합 모델 비교 (개발 3데이터)", "같은 구조·설정을 Sunwoda / RWTH / MICH에 공동 적용")
    headers = ["모형", "Sunwoda", "RWTH", "MICH", "평균", "최고", "최저"]
    rows = [
        ["고정 경계형", "0.939", "0.878", "0.468", "0.762", "0.939", "0.468"],
        ["보정 제한 없음", "0.718", "0.788", "0.759", "0.755", "0.788", "0.718"],
        ["전체 적응형", "0.719", "0.738", "0.746", "0.734", "0.746", "0.719"],
        ["거리 기반 보정", "0.894", "0.800", "0.736", "0.810", "0.894", "0.736"],
        ["최종 모델", "0.934", "0.842", "0.751", "0.842", "0.934", "0.751"],
    ]
    xs = [55, 300, 430, 550, 680, 820, 960]
    ws = [230, 110, 100, 100, 110, 110, 110]
    for i, h in enumerate(headers):
        add_text(s, xs[i], 165, ws[i], 24, h, 12, C["navy"], True, "center" if i else "left")
    for ri, row in enumerate(rows):
        y = 205 + ri * 55
        bg = C["wash"] if ri == 4 else C["white"]
        rect(s, 50, y - 8, 1160, 48, bg, C["teal"] if ri == 4 else C["line"], True)
        for i, val in enumerate(row):
            add_text(s, xs[i], y, ws[i], 28, val, 13, C["teal"] if ri == 4 and i else C["ink"], ri == 4, "center" if i else "left")
    add_text(
        s,
        90,
        500,
        1100,
        40,
        "해석: 최종 모델은 평균·최저를 같이 올림. 최고는 고정 경계형(0.939)과 비슷하나, 최저(0.751 vs 0.468)에서 차이.",
        13,
        C["ink"],
        True,
    )
    add_text(s, 90, 555, 1100, 30, "원문: UNIFIED_SUPPORT_GATED_PP_RESULTS_KO.md · results/bq_dual_scale_final_replay_v1/", 12, C["grey"])
    foot(s, p())

    # A6 positive 8
    s = blank(prs)
    topbar(s, "Apx", "표 2. 잘 된 8곳 — 알고리즘별 R²", "TabPFN 8곳 전체 반영")
    headers = ["데이터", "PP", "V-REx", "G-DRO", "Mono", "LinRBF", "Engr.", "GP", "TabPFN"]
    rows = [
        ["HUST", "0.958", "0.809", "0.934", "0.822", "0.710", "0.878", "-0.32", "0.218"],
        ["Virkler", "0.888", "0.583", "0.554", "0.565", "0.805", "0.552", "0.54", "0.621"],
        ["NASA", "0.584", "0.285", "0.286", "0.283", "0.550", "0.549", "0.44", "-0.69"],
        ["Sunwoda", "0.865", "-0.24", "-0.30", "-0.05", "0.838", "0.619", "-1.60", "-0.89"],
        ["RWTH", "0.743", "0.645", "0.602", "-0.01", "0.385", "0.526", "-0.47", "-2.18"],
        ["MATR19", "0.466", "0.044", "0.272", "0.018", "-2.64", "-0.73", "-2.46", "0.202"],
        ["MATRb2", "0.862", "0.850", "0.777", "0.674", "-0.78", "0.739", "0.21", "0.618"],
        ["NCMAPSS", "0.937", "0.883", "0.880", "0.892", "0.819", "0.932", "0.80", "0.934*"],
    ]
    xs = [55, 200, 320, 430, 540, 650, 770, 880, 1000]
    for i, h in enumerate(headers):
        add_text(s, xs[i], 148, 110, 22, h, 10, C["navy"], True)
    for ri, row in enumerate(rows):
        y = 178 + ri * 40
        for i, val in enumerate(row):
            col = C["teal"] if i == 1 else C["ink"]
            add_text(s, xs[i], y, 120 if i == 0 else 100, 26, val, 11, col, i == 1)
    add_text(s, 90, 515, 1100, 55, "TabPFN: FAIR_PFN(HUST/Virkler/NASA) · TABPFN_EXTERNAL(Sunwoda/RWTH/MATRb2) · EXTENDED_NN(MATR19) · NCMAPSS(*단일 seed).", 11, C["grey"])
    add_text(s, 90, 570, 1100, 25, "보조 비교(train 상한). MATRb2는 TabPFN이 기본 PP보다 높을 수 있어 그 분할만으로 PP 우위를 주장하지 않음.", 11, C["crimson"])
    foot(s, p())

    # A7 failures
    s = blank(prs)
    topbar(s, "Apx", "표 3. 실패 설정 — 적용 한계", "모든 방법이 음수인 경우도 논문에 남긴다")
    fails = [
        ("MICH (기본 PP)", "−1.522", "보정이 거의 안 켜짐\n형태/관계 실패\n→ 경계형으로 일부 회복", C["red"]),
        ("XJTU", "−1.229", "검증과 테스트의\n이동 방향이 반대\n→ 옮기지 않는 게 맞음", C["orange"]),
        ("FEMTO", "−1.378", "베어링당 끝점 1개\n유닛 점수 정의 어려움", C["blue"]),
        ("NASA milling", "−4.826", "검증 유닛이 너무 적음\n나눔 자체가 불안정", C["navy"]),
    ]
    for i, (h, v, b, a) in enumerate(fails):
        x = 75 + i * 295
        rect(s, x, 175, 280, 300, C["white"], C["line"], True)
        rect(s, x, 175, 280, 8, a)
        add_text(s, x + 14, 200, 250, 40, h, 14, C["ink"], True)
        add_text(s, x + 14, 250, 250, 36, v, 24, a, True)
        add_text(s, x + 14, 310, 250, 140, b, 13, C["grey"])
    add_text(s, 90, 520, 1100, 50, "원문: ADDITIONAL_REAL_DATASETS_RESULTS_KO.md · XJTU_UNTOUCHED_RESULTS_KO.md · FEMTO_PP_PROSPECTIVE_RESULTS_KO.md · MILLING_LOCKED_RESULTS.md", 12, C["grey"])
    add_text(s, 90, 575, 1100, 30, "논문 메시지: 실패를 숨기지 않고, ‘언제 거절해야 하는지’의 근거로 쓴다.", 14, C["crimson"], True)
    foot(s, p())

    # A8 mich recovery
    s = blank(prs)
    topbar(s, "Apx", "사례 연구: MICH 회복", "기본 PP −1.522 → 경계형 0.468 → 최종 통합 0.751")
    pic(s, "mich_units.png", 50, 150, 780, 400)
    card(s, 870, 170, 340, 140, "무슨 일이었나", "건강도–남은수명 관계가\n학습 집단과 달라져\n보정이 꺼졌다", C["red"])
    card(s, 870, 330, 340, 140, "무엇을 바꿨나", "경계를 몫으로 두고\n기본경향 고정 +\n보정 상한", C["teal"])
    card(s, 870, 490, 340, 90, "한계", "unit 31은 여전히\n가장 어렵고\n완전 해결은 아님", C["orange"])
    foot(s, p())

    # A9 stats paper style
    s = blank(prs)
    topbar(s, "Apx", "통계 근거 (Results 보강)", "알고리즘 안정성 + 유닛 단위 효과")
    card(s, 80, 165, 560, 280, "재학습 합의 규칙", "보정 채택을 5번 재학습 투표로 봄\n5/5일 때만 승인 (단측 p=0.031)\n4/5는 거절\n\n주의: 물리 반복실험이 아니라\n초기화 안정성 검정", C["teal"])
    card(s, 680, 165, 520, 280, "유닛 짝 비교 (20,000회)", "HUST  −25.2  [−34.4, −14.5]\nRWTH  −23.3  [−41.8, −8.9]\nMATR  −10.6  [−15.7, −6.4]\n\n세 곳 모두 신뢰구간이 0 아래\n= 긴 시퀀스 몇 개만의 착시 아님", C["blue"])
    add_text(s, 90, 480, 1100, 40, "안전성 감사: 7곳 중 3곳 개선 · 4곳 그대로 · 악화 0", 15, C["ink"], True)
    add_text(s, 90, 530, 1100, 40, "원문: STATISTICAL_NOVELTY_EVIDENCE_KO.md · results/consensus_statistical_audit_v1/", 12, C["grey"])
    add_text(s, 90, 575, 1100, 30, "도메인 수가 적어 ‘모든 분야에서 유의’라고 쓰면 안 된다.", 13, C["red"], True)
    foot(s, p())

    # A10 contributions
    s = blank(prs)
    topbar(s, "Apx", "기여 / 비기여 (논문 경계)", "심사에서 바로 물어보는 것")
    card(s, 80, 165, 560, 360, "기여로 쓸 수 있는 것", "1) 개체 분리 + 범위 밖 남은수명 문제 정의\n2) 고정 기본경향 + 제한 보정 예측기\n3) 검증으로만 승인하는 보정 옮기기\n4) 실패·거절까지 포함한 실험 지도", C["teal"])
    card(s, 680, 165, 520, 360, "기여로 쓰면 안 되는 것", "최초 물리–신경망 혼합\n최초 affine+residual\n최초 범위 밖 평가\n모든 도메인 1등\n보정=이론적 안전 보증", C["red"])
    add_text(s, 90, 560, 1100, 30, "원문: NOVELTY_LITERATURE_AUDIT_KO.md · BQ_PP_NOVELTY_AUDIT_KO.md · PP_FIRST_PUBLICATION_CLAIMS_KO.md", 12, C["grey"])
    foot(s, p())

    # A11 reading list / file index
    s = blank(prs)
    topbar(s, "Apx", "공부 체크리스트 — 파일 연결", "읽을 순서대로")
    items = [
        ("① 문헌", "ppt/외삽자료/extrapolation-papers/외삽_문헌조사_통합.pptx\n+ 외삽_50분_장표별_설명_v5.md"),
        ("② 문제·분할", "pp-extrapolation/MODEL_AND_SPLIT_KO.md"),
        ("③ 최종 결과", "UNIFIED_SUPPORT_GATED_PP_RESULTS_KO.md\nBOUNDARY_QUOTIENT_PP_RESULTS_KO.md"),
        ("④ 경쟁·실패", "ALL_DATASET_EXTRAPOLATION_COMPETITORS_KO.md\nENGRESSION_GP_EXTRAPOLATION_RESULTS_KO.md"),
        ("⑤ 통계·노벨티", "STATISTICAL_NOVELTY_EVIDENCE_KO.md\nNOVELTY_LITERATURE_AUDIT_KO.md"),
        ("⑥ 확증·한계", "MATR_BATCH2_CONFIRMATORY_RESULTS_KO.md\nAPPLICABILITY_NOVELTY_LIMITS_KO.md"),
    ]
    for i, (h, b) in enumerate(items):
        x = 80 + (i % 3) * 390
        y = 165 + (i // 3) * 220
        card(s, x, y, 370, 195, h, b, [C["orange"], C["teal"], C["blue"], C["navy"], C["green"], C["crimson"]][i])
    foot(s, p())

    # A12 more file index dense
    s = blank(prs)
    topbar(s, "Apx", "실험 원문 색인 (더 보고 싶을 때)", "주제 → 파일명")
    lines = [
        "기본 PP 프로토콜 · ADDITIONAL_REAL_DATASETS_PROTOCOL / RESULTS",
        "교차 비교 · ALL_DATASET_COMPARISON_KO · CROSS_DATASET_PP_FT_*",
        "보정·합의 · OUTPUT_CALIBRATION_CROSS_DATASET · STATISTICAL_NOVELTY_*",
        "거리·껍질 · HULL_EXTRAPOLATION_QUANTIFICATION · DISTANCE_SHELL_*",
        "선택 거절 · SELECTIVE_GATE_RESULTS · APPLICABILITY_*",
        "미접촉 확인 · XJTU_UNTOUCHED_* · OXFORD_* · MATR_BATCH2_* · MATR_2019_*",
        "엔진/균열 · NCMAPSS_PP_COMPARISON · FEMTO_* · MILLING_*",
        "프레임 전체 · ASSUMPTION_AWARE_EXTRAPOLATION_PROGRAM_KO · PAE_PP_*",
    ]
    for i, t in enumerate(lines):
        y = 160 + i * 48
        rect(s, 90, y + 6, 8, 8, C["orange"], None, True)
        add_text(s, 115, y, 1080, 35, t, 14, C["ink"])
    foot(s, p())

    # A13 lit map
    s = blank(prs)
    topbar(s, "Apx", "문헌조사와 PP의 연결", "50분 발표에서 가져온 것 / PP로 이어진 것")
    card(s, 80, 170, 540, 320, "문헌에서 배운 것", "밖 = 가정\nReLU는 직선으로 이어짐\n강한 식은 틀리면 위험\n범위 밖으로 검증해야 함\n모르면 불확실성·거절", C["orange"])
    card(s, 660, 170, 540, 320, "PP로 옮긴 것", "약한 가정만 쓰기\n기본 경향을 직선적으로 고정\n보정은 상한으로 막기\n개체 분리 + 범위 밖 평가\n안 되면 보정 안 함/거절", C["teal"])
    add_text(s, 90, 530, 1100, 40, "연결 파일: ppt/외삽자료/.../외삽_50분_장표별_설명_v5.md · INDEX.md · paper_pdfs/", 12, C["grey"])
    add_text(s, 90, 575, 1100, 30, "추천 PPT: 외삽_문헌조사_통합.pptx (먼저) → 본 자료 본문 → 이 부록", 14, C["ink"], True)
    foot(s, p())

    # A14 paper outline
    s = blank(prs)
    topbar(s, "Apx", "논문 목차 초안 (공부용)", "쓰고 싶은 순서")
    outline = [
        ("1 Intro", "학습 밖 RUL · 가정 문제 · 기여 3개"),
        ("2 Related", "hybrid/PINN · selective · hull 검증 · 우리 자리"),
        ("3 Problem", "개체 분리 · 범위 밖 · 지표"),
        ("4 Method", "기본경향+보정 · 경계형 · 승인 규칙"),
        ("5 Experiments", "표1 통합 · 표2 경쟁 · 표3 실패 · 통계"),
        ("6 Discussion", "적용 조건 · 한계 · PAE 컴파일로의 연결"),
    ]
    for i, (h, b) in enumerate(outline):
        y = 165 + i * 70
        rect(s, 90, y, 200, 55, [C["orange"], C["teal"], C["blue"], C["navy"], C["green"], C["crimson"]][i], None, True)
        add_text(s, 105, y + 14, 170, 28, h, 14, C["white"], True)
        add_text(s, 320, y + 12, 860, 35, b, 16, C["ink"])
    foot(s, p())

    # A15 closing appendix
    s = blank(prs)
    topbar(s, "Apx", "부록 사용법", "발표 / 공부 / 원고")
    cards = [
        ("발표할 때", "본문만\n질문 시 데이터셋·표", C["orange"]),
        ("공부할 때", "데이터셋→스플릿→\n성공/실패→결과 표", C["teal"]),
        ("원고 쓸 때", "Methods=스플릿\nResults=표1–3", C["navy"]),
    ]
    for i, (h, b, a) in enumerate(cards):
        card(s, 90 + i * 380, 200, 350, 220, h, b, a)
    add_text(s, 90, 480, 1100, 50, "고정본 파일: ppt/pp/PP_Research_Detailed_v2.pptx\n실험 원문 루트: Desktop/연구/pp-extrapolation/", 15, C["ink"], True)
    add_text(s, 90, 560, 1100, 30, "끝. 새 버전 파일은 만들지 않고 이 파일만 갱신한다.", 13, C["grey"])
    foot(s, p())

    prs.save(str(OUT))
    print(f"Saved {OUT} ({len(prs.slides)} slides)")
    print("Figures:", sorted(p.name for p in FIGS.glob('*.png')))


if __name__ == "__main__":
    build()
