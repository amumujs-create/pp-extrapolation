#!/usr/bin/env python3
"""PP-X 20-minute talk deck.

Talk body (~14 slides) + Appendix (skipped detail from the full 47-slide deck).
Output: ppt/PP-X_Talk_20min.pptx
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.util import Pt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "ppt" / "PP-X_Talk_20min.pptx"
HELPER = Path(__file__).resolve().parent / "build_pp_research_deck_v2.py"

spec = importlib.util.spec_from_file_location("deck_v2", HELPER)
deck = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deck)

W, H = deck.W, deck.H
C = deck.C
blank = deck.blank
head = deck.head
foot = deck.foot
add_text = deck.add_text
add_table = deck.add_table
rect = deck.rect
hline = deck.hline
right_arrow = deck.right_arrow
pic = deck.pic
px = deck.px


def build() -> None:
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    n = 0
    # Talk 15 + appendix divider 1 + appendix 9 = 25
    TOTAL = 25

    def p():
        nonlocal n
        n += 1
        return n

    # ---------- TALK BODY ----------
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
    add_text(s, 80, 200, 1120, 40, "PP-X", 30, C["ink"], True, "center")
    add_text(s, 80, 250, 1120, 36, "Validation-Approved Prior-Residual Extrapolation", 18, C["ink"], True, "center")
    add_text(s, 80, 310, 1120, 28, "20분 발표용  ·  본편 + 부록", 16, C["blue"], True, "center")
    add_text(s, 80, 420, 1120, 24, "Smart Production Systems Lab.  ·  박사과정 박진서", 14, C["ink"], False, "center")
    add_text(s, 80, 470, 1120, 22, "상세 47장 덱의 압축판  ·  스킵 장은 부록", 13, C["muted"], False, "center")
    p()

    # 2 Agenda
    s = blank(prs)
    head(s, "오늘 구성 — 본편 / 부록", "본편만으로 약 20분. 부록은 질문·상세 근거용.")
    add_table(
        s, 48, 100, 1184, 360,
        ["구분", "장", "내용", "시간 감각"],
        [
            ["본편", "1–15", "문제 · 연구 루트 · 기여 · 게이트 · 주결과 · 통계 · DS03 · 한계", "~20분"],
            ["부록 A", "A1–A3", "약한 prior · 외삽 정의 · 도메인 안정성", "질문 시"],
            ["부록 B", "B1–B6", "ablation · 비교 heatmap · CCMR · PAE · 실패 상세", "질문 시"],
        ],
        font_size=14,
    )
    add_text(s, 48, 500, 1184, 60,
             "발표 때 부록 슬라이드는 넘기지 않는다. 필요하다고 말하면 해당 번호로 점프.",
             14, C["ink"], True)
    foot(s, p(), TOTAL)

    # 3 Problem
    s = blank(prs)
    head(s, "문제", "학습 support 밖에서는 데이터가 답을 정하지 못한다.")
    pic(s, "nn_vs_saar_curves.png", 36, 90, 620, 380)
    add_text(s, 36, 490, 620, 50,
             "점선 왼쪽=학습, 오른쪽=밖. 제약 없는 NN은 열화 중에도 예측이 되살아날 수 있다.",
             12, C["muted"])
    rect(s, 690, 96, 542, 150, C["soft_orange"], C["orange"], False)
    add_text(s, 706, 110, 510, 24, "왜 위험한가", 14, C["orange"], True)
    add_text(s, 706, 148, 510, 80,
             "RUL·피로·열화는 시험 시점이 학습 분포 밖인 경우가 많다. 보간 성능을 외삽으로 착각하기 쉽다.",
             13, C["ink"])
    rect(s, 690, 270, 542, 200, C["soft_blue"], C["blue"], False)
    add_text(s, 706, 284, 510, 24, "PP-X가 하는 일", 14, C["blue"], True)
    add_text(s, 706, 322, 510, 130,
             "① 믿을 prior만 contract로 허용\n"
             "② prior를 뒤집지 못하게 residual 제한\n"
             "③ Val unit-risk로 실행 승인\n"
             "④ 실패 시 사전 fallback",
             13, C["ink"])
    foot(s, p(), TOTAL)

    # 4 Research route — shared visual diagram with the detailed deck
    s = blank(prs)
    head(s, "연구 루트 — 식의 근거에 따라 두 경로로 간다",
         "식이 없거나 약한 prior만 있으면 PP-X, 적용 가능한 식이 정당화되면 PAE로 확장한다.")
    deck.research_route_diagram(s)
    foot(s, p(), TOTAL)

    # 5 Contributions + punchline merge
    s = blank(prs)
    head(s, "기여 — C1·C2·C3 + 한 줄 효과",
         "Declare → Learn → Approve. 숫자는 retrospective 메인 9-setting.")
    add_table(
        s, 48, 96, 1184, 220,
        ["", "질문", "핵심"],
        [
            ["C1", "무엇을 믿나?", "Boundary·history·support·허용 prior·fallback을 test 전 선언"],
            ["C2", "얼마나 수정하나?", "ŷ = prior + bounded residual"],
            ["C3", "써도 되나?", "Gain≥2% · unit wins≥60% · worst≤1.10"],
        ],
        font_size=13,
    )
    add_table(
        s, 48, 350, 1184, 180,
        ["효과 (메인 9셋)", "수치", "주의"],
        [
            ["Equal-budget 최강 대비", "8/9 · p=.039", "Virkler는 FT가 +.002"],
            ["vs matched direct", "9/9 · p=.0039 · CI[0.17,0.67]", "요약 효과. 전 셋 BH 유의는 아님"],
            ["GM-RMSE 감소", "33.8%", "retrospective. prospective 보장 아님"],
        ],
        font_size=13,
    )
    foot(s, p(), TOTAL)

    # 5 Gate overview
    s = blank(prs)
    head(s, "게이트 전체도 — 어디서 승인·거절되나",
         "Prior gate = BQ vs affine  ·  Executor gate = 어떤 PP")
    stages = [
        (48, "Step 1 · C1", "Contract\n후보·fallback", C["soft"], C["ink"]),
        (280, "Step 2 · C1", "Prior gate\nBQ / affine", C["soft_blue"], C["blue"]),
        (512, "Step 3 · C2", "prior +\nbounded residual", C["soft"], C["ink"]),
        (744, "Step 4 · C3", "Executor gate\n2%·60%·1.10", C["soft_orange"], C["orange"]),
        (976, "Step 5 · Freeze", "C3 이후 실행\nroute or fallback", C["soft"], C["ink"]),
    ]
    for left, title, body, fill, edge in stages:
        rect(s, left, 100, 216, 110, fill, edge, False)
        add_text(s, left + 8, 108, 200, 24, title, 12, edge, True, "center")
        add_text(s, left + 8, 140, 200, 60, body, 12, C["ink"], False, "center")
    for x in (264, 496, 728, 960):
        right_arrow(s, x, 140, 16, 28, C["muted"])
    rect(s, 48, 250, 580, 180, C["white"], C["blue"], False)
    add_text(s, 64, 262, 548, 24, "Step 2 · Final은 Prior ON", 14, C["blue"], True)
    add_text(s, 64, 300, 548, 110,
             "경계 → BQ, 없음 → affine\nOOF/group/mode 계산 없음\nneural_safety는 v1_declared 보관",
             13, C["ink"])
    rect(s, 652, 250, 580, 180, C["white"], C["orange"], False)
    add_text(s, 668, 262, 548, 24, "Step 4 · C3 거절 → Val FAIL", 14, C["orange"], True)
    add_text(s, 668, 300, 548, 110,
             "비교는 했으나 세 조건 미달\n→ 같은 Direct fallback\n(사전 contract)",
             13, C["ink"])
    add_text(s, 48, 470, 1184, 70,
             "승인 유일한 길  Contract → Prior ON → Fit → Val PASS → Freeze\n"
             "test label로 route를 바꾸지 않는다.",
             14, C["ink"])
    foot(s, p(), TOTAL)

    # 6 Prior gate
    s = blank(prs)
    head(s, "Prior gate — known_boundary 분기",
         "select_ppx_route.  고장 조건 ≠ 고장 시점.")
    rect(s, 440, 90, 400, 40, C["ink"], None, True)
    add_text(s, 450, 96, 380, 28, "known_boundary ?", 15, C["white"], True, "center")
    rect(s, 48, 155, 560, 320, C["soft_blue"], C["blue"], False)
    add_text(s, 64, 168, 528, 26, "TRUE → boundary_pp · Prior ON", 14, C["blue"], True)
    add_text(s, 64, 210, 528, 240,
             "예) capacity≤80%, crack 33 mm\n"
             "OOF 검사 없음\n"
             "→ prior_weight=1\n"
             "→ Fit → Executor gate",
             14, C["ink"])
    rect(s, 672, 155, 560, 320, C["soft_orange"], C["orange"], False)
    add_text(s, 688, 168, 528, 26, "FALSE → affine · Prior ON", 14, C["orange"], True)
    add_text(s, 688, 210, 528, 240,
             "Final은 사다리를 안 돌림.\n"
             "경계 없으면 affine.\n\n"
             "OOF / group / mode는\n"
             "계산하지 않음.\n\n"
             "그 검사는 v1_declared 보관.",
             14, C["ink"])
    add_text(s, 48, 510, 1184, 50,
             "경계 있음 = BQ.  경계 없음 = affine.  둘 다 Prior ON.\n"
             "Final에서 Prior OFF는 없다. 거절은 Executor Val FAIL만.",
             13, C["ink"])
    foot(s, p(), TOTAL)

    # 7 Executor + branch dictionary MERGED
    s = blank(prs)
    head(s, "C3 Executor gate + 분기 이름",
         "select_paper_ppx. Prior ON일 때만. 세 조건 AND.")
    add_text(s, 48, 92, 1184, 40,
             "후보  prior-only · unbounded · bounded · dual* · transport* · history* · direct fallback",
             13, C["ink"])
    crits = [
        (48, "Gain ≥ 2%", C["soft_blue"], C["blue"]),
        (440, "Unit wins ≥ 60%", C["soft_blue"], C["blue"]),
        (832, "Worst ≤ 1.10", C["soft_blue"], C["blue"]),
    ]
    for left, title, fill, edge in crits:
        rect(s, left, 145, 368, 56, fill, edge, False)
        add_text(s, left + 10, 155, 348, 36, title, 14, edge, True, "center")
    rect(s, 48, 220, 560, 90, C["soft_blue"], C["blue"], False)
    add_text(s, 64, 235, 528, 60, "PASS → Val loss 최소 executor freeze\n(동점이면 더 단순)", 13, C["ink"], True, "center")
    rect(s, 672, 220, 560, 90, C["soft_orange"], C["orange"], False)
    add_text(s, 688, 235, 528, 60, "FAIL → Direct fallback", 13, C["ink"], True, "center")
    add_table(
        s, 48, 340, 1184, 230,
        ["이름", "언제", "Prior", "다음"],
        [
            ["boundary_pp", "known_boundary", "ON · BQ", "executor"],
            ["transferable_prior", "경계 없음", "ON · affine", "executor"],
            ["neural_safety", "v1_declared만", "OFF", "바로 fallback"],
            ["승인 executor", "Val PASS", "ON", "route freeze"],
            ["Direct fallback", "Val FAIL (Final)", "—", "예측만"],
        ],
        font_size=12,
    )
    foot(s, p(), TOTAL)

    # 8 Main results
    s = blank(prs)
    head(s, "주 결과 — 메인 9-setting",
         "1D 엄격 외삽 · retrospective development · PAE 없음")
    add_table(
        s, 48, 96, 1184, 420,
        ["셋", "Executor", "R²"],
        [
            ["HUST", "regime transport", "0.958"],
            ["Sunwoda", "fixed BQ", "0.939"],
            ["N-CMAPSS DS02", "multiscale", "0.937"],
            ["Virkler", "gated residual", "0.888"],
            ["RWTH", "fixed BQ", "0.878"],
            ["MATR-b2", "decay+transport", "0.862"],
            ["MICH", "dual-scale BQ", "0.751"],
            ["NASA battery", "multiscale history", "0.584"],
            ["MATR2019", "affine latent-regime + Val calibration", "0.466"],
        ],
        font_size=13,
    )
    add_text(s, 48, 545, 1184, 40,
             "보편 SOTA 주장 아님. contract-conditioned route로 외삽을 유지한 결과.",
             13, C["muted"])
    foot(s, p(), TOTAL)

    # 9 Stats + stability MERGED
    s = blank(prs)
    head(s, "통계적 근거 — 요약 효과는 확보",
         "unit 추론 · exact sign · hierarchical bootstrap · BH. 전 셋 개별 유의는 아님.")
    add_table(
        s, 48, 96, 1184, 280,
        ["검정", "결과", "해석"],
        [
            ["Domain sign vs direct", "9/9 · p=.0039", "방향성 확보"],
            ["Hierarchical CI (log-RMSE)", "[0.17, 0.67]", "요약 이득 CI가 0 밖"],
            ["GM-RMSE 감소", "33.8%", "평균 0.41 log-ratio"],
            ["BH 개별 유의", "Sunwoda · MATR-b2", "나머지 셋은 요약으로만"],
            ["Equal-budget 최강", "8/9 · p=.039", "공정 예산 비교"],
        ],
        font_size=13,
    )
    add_text(s, 48, 420, 1184, 100,
             "말하지 않는 것\n"
             "· common-backbone 12셋 전역 우세  ·  DS03 정확도 우월\n"
             "· 8모델 Holm 동시 보정 후 안정성 유의  ·  미래 cohort 보장",
             13, C["red"])
    foot(s, p(), TOTAL)

    # 10 Fair comparison
    s = blank(prs)
    head(s, "공정 비교 — 동일 30후보 예산",
         "9 settings × 8 baselines · search seed 42 · refit 42–46")
    add_table(
        s, 48, 96, 1184, 380,
        ["Setting", "PP-X", "최강 baseline", "ΔR²"],
        [
            ["HUST", "0.958", "GroupDRO 0.955", "+.003"],
            ["Virkler", "0.888", "FT 0.890", "−.002"],
            ["NASA", "0.584", "Engression 0.583", "~0"],
            ["Sunwoda", "0.939", "RBF 0.838", "+.102"],
            ["RWTH", "0.878", "RBF 0.732", "+.146"],
            ["MICH", "0.751", "monotone −0.686", "+1.44"],
            ["MATR2019", "0.466", "FT 0.342", "+.123"],
            ["MATR-b2", "0.862", "MLP 0.813", "+.049"],
            ["N-CMAPSS", "0.937", "Engression 0.932", "+.005"],
        ],
        font_size=12,
    )
    add_text(s, 48, 510, 1184, 50,
             "결론  8/9 우세 · exact p=.0391. 예외는 Virkler(+.002)·NASA 동률.",
             14, C["ink"], True)
    foot(s, p(), TOTAL)

    # 11 DS03
    s = blank(prs)
    head(s, "Prospective — N-CMAPSS DS03",
         "방법·임계값 동결 후 test Y 공개. route와 정확도를 분리해서 말함.")
    rect(s, 48, 110, 580, 280, C["soft_blue"], C["blue"], False)
    add_text(s, 64, 124, 548, 28, "PASS — route selection", 16, C["blue"], True)
    add_text(s, 64, 170, 548, 190,
             "prior routes 거절\n→ direct fallback 선택\n\n"
             "실제 test-best PP-X route와 일치\n\n"
             "unsupported prior를 강제로 안 씀",
             14, C["ink"])
    rect(s, 652, 110, 580, 280, C["soft_orange"], C["orange"], False)
    add_text(s, 668, 124, 548, 28, "FAIL — predictive superiority", 16, C["orange"], True)
    add_text(s, 668, 170, 548, 190,
             "PP-X fallback R² 0.882\n"
             "Engression 0.901\n"
             "ΔR² −0.0195\n\n"
             "정확도 1등 주장은 안 함",
             14, C["ink"])
    add_text(s, 48, 430, 1184, 80,
             "정직한 한 줄\n"
             "prospective route-selection success, predictive superiority not confirmed.",
             14, C["ink"], True)
    foot(s, p(), TOTAL)

    # 12 Limitations
    s = blank(prs)
    head(s, "한계 · 주장 경계", "오늘 표는 무엇을 말하고, 무엇을 말하지 않는가.")
    add_table(
        s, 48, 100, 1184, 320,
        ["말함", "말하지 않음"],
        [
            ["메인 9셋 요약 이득·방향성", "모든 holdout 정확도 SOTA"],
            ["contract-conditioned 실행·거절", "validation gate의 미래 완전 식별"],
            ["DS03 prior 거절 성공", "DS03 fallback 우월"],
            ["optional executor 반례 존재", "전역 dual/bound default"],
        ],
        font_size=14,
    )
    add_text(s, 48, 460, 1184, 70,
             "Limitation tier (메인 승수 제외): XJTU · FEMTO · NASA milling",
             13, C["muted"])
    foot(s, p(), TOTAL)

    # 13 Takeaway
    s = blank(prs)
    head(s, "가져갈 한 줄", "20분 발표의 결론.")
    rect(s, 80, 140, 1120, 200, C["ink"], None, True)
    add_text(s, 110, 180, 1060, 120,
             "PP-X는 모든 데이터에서 1등이 아니라,\n"
             "근거 있는 prior-residual만 승인하고\n"
             "없으면 fallback으로 외삽 위험을 제한한다.",
             20, C["white"], True, "center")
    add_text(s, 80, 400, 1120, 100,
             "Prior gate → Executor gate → Freeze\n"
             "known_boundary = 고장 조건, 고장 시점 아님",
             16, C["ink"], True, "center")
    foot(s, p(), TOTAL)

    # 15 Next / Q&A
    s = blank(prs)
    head(s, "다음 · 질문", "상세 ablation·PAE·CCMR은 부록.")
    add_table(
        s, 48, 110, 1184, 280,
        ["다음", "내용"],
        [
            ["단기", "fallback 표현력 · typed contract 강화"],
            ["중기", "PAE (검증된 식이 있을 때)"],
            ["부록", "지금 질문하시면 해당 장으로 이동합니다"],
        ],
        font_size=15,
    )
    add_text(s, 48, 450, 1184, 60,
             "질문 받겠습니다.",
             20, C["blue"], True, "center")
    foot(s, p(), TOTAL)

    # ---------- APPENDIX DIVIDER ----------
    s = blank(prs)
    rect(s, 0, 0, 1280, 720, C["ink"], None, False)
    add_text(s, 80, 260, 1120, 50, "부록", 36, C["white"], True, "center")
    add_text(s, 80, 340, 1120, 80,
             "본편에서 스킵한 상세 장표\n발표 본편(1–15)에서는 사용하지 않음",
             16, C["white"], False, "center")
    p()  # no foot on dark slide - add muted page
    add_text(s, 1100, 660, 100, 20, f"{n}/{TOTAL}", 10, C["muted"], True, "right")

    # A1 Weak prior
    s = blank(prs)
    head(s, "A1  약한 prior란", "부록 · Prior와 executor를 분리해서 읽는다")
    add_table(
        s, 48, 100, 1184, 280,
        ["층", "후보", "언제", "하는 일"],
        [
            ["Prior A · 경계형", "BQ: 경계거리×quotient", "EOL 경계를 앎", "끝점 0 보장"],
            ["Prior B · 비경계형", "Direct affine: X→RUL", "known_boundary=False (Final)", "RUL 직접 회귀"],
            ["Executor", "bound · dual · history · transport", "flag + Val PASS", "prior 수정·보정"],
            ["Fallback", "direct / persistence", "Executor Val FAIL (Final)", "PP 경로 미사용"],
        ],
        font_size=11,
    )
    add_text(s, 48, 400, 1184, 36,
             "BQ는 margin=(health−EOL)을 곱해 경계에서 0을 보장한다. Direct affine은 경계 없이 RUL을 바로 Ridge 회귀한다.",
             11, C["blue"], True, "center")
    add_text(s, 48, 450, 1184, 70,
             "예  Sun·RWTH: BQ latent-affine→bounded  ·  MICH: BQ latent-affine→dual-scale\n"
             "NASA·N-CMAPSS: Direct-RUL affine→history  ·  HUST·MATRb2: affine+residual→transport",
             12, C["ink"])
    foot(s, p(), TOTAL)

    # A2 Extrapolation definition
    s = blank(prs)
    head(s, "A2  외삽 구간 정의", "부록 · 1D 열화좌표 hull-out")
    add_table(
        s, 48, 110, 1184, 300,
        ["규칙", "의미"],
        [
            ["unit-disjoint", "train/val/test 셀 비겹침"],
            ["hull-out", "선언한 1D 축에서 test가 train 밖"],
            ["val-only", "게이트·보정은 val만"],
            ["test once", "고정 시험 분할 한 번만"],
        ],
        font_size=14,
    )
    foot(s, p(), TOTAL)

    # A3 Domain stability table
    s = blank(prs)
    head(s, "A3  도메인 안정성 표", "부록 · 9셋 macro / SD")
    add_table(
        s, 48, 100, 1184, 320,
        ["모델", "macro R²", "도메인 SD", "R²>0"],
        [
            ["PP-X", "0.807", "0.174", "9/9"],
            ["Engression", "0.257", "0.949", "7/9"],
            ["plain MLP", "0.093", "1.007", "6/9"],
            ["GroupDRO", "0.011", "0.998", "5/9"],
        ],
        font_size=14,
    )
    add_text(s, 48, 460, 1184, 50,
             "Holm 8모델 동시 보정은 비유의. 개별 대표 비교와 분리 보고.",
             13, C["muted"])
    foot(s, p(), TOTAL)

    # B1 Ablation dual
    s = blank(prs)
    head(s, "B1  Ablation — dual-scale은 전역 default가 아님",
         "부록 · MICH만 dual, RWTH는 dual 악화")
    add_text(s, 48, 140, 1184, 200,
             "MICH: fixed bound가 residual을 과도 제한 → dual로 복구\n"
             "RWTH: dual이 unit 전면 악화\n\n"
             "→ optional executor는 contract+Val로만 켬",
             16, C["ink"])
    foot(s, p(), TOTAL)

    # B2 Failure analysis
    s = blank(prs)
    head(s, "B2  실패 유형 요약", "부록")
    add_table(
        s, 48, 110, 1184, 320,
        ["유형", "예"],
        [
            ["제약≠shift", "MICH fixed / RWTH dual"],
            ["val↛test", "RMSE-only 정책 FA"],
            ["fallback 약함", "DS03 vs Engression"],
            ["정보 부족", "Virkler short tail"],
        ],
        font_size=14,
    )
    foot(s, p(), TOTAL)

    # B3 Policy audit
    s = blank(prs)
    head(s, "B3  정책 감사 (12-domain)", "부록 · common-backbone")
    add_table(
        s, 48, 120, 1184, 280,
        ["정책", "정확", "FA", "FR"],
        [
            ["Always direct", "6/12", "0", "6"],
            ["Always PP", "6/12", "6", "0"],
            ["Val RMSE only", "7/12", "4", "1"],
            ["Unit-risk PP-X", "8/12", "2", "2"],
        ],
        font_size=14,
    )
    foot(s, p(), TOTAL)

    # B4 BQ+Affine reject
    s = blank(prs)
    head(s, "B4  BQ+Affine 혼합 — 승격 거절", "부록 · ablation")
    add_text(s, 48, 140, 1184, 220,
             "OOF α=0 (순수 Affine)\n"
             "mix_pp equal R² −1.65 vs BQ-PP 0.75\n\n"
             "→ α knob 메인 미채택. BQ-PP 유지.",
             16, C["ink"])
    foot(s, p(), TOTAL)

    # B5 CCMR note
    s = blank(prs)
    head(s, "B5  CCMR v2.2", "부록 · 메커니즘 증거. 논문 메인 모델명 아님")
    add_text(s, 48, 160, 1184, 200,
             "trajectory-domain risk-aware executor 사례.\n"
             "PP-X Algorithm 1의 typed selection 프레임 안의 한 executor.",
             16, C["ink"])
    foot(s, p(), TOTAL)

    # B6 PAE
    s = blank(prs)
    head(s, "B6  후속 PAE", "부록 · 검증된 식이 있을 때")
    add_text(s, 48, 160, 1184, 200,
             "출처 고정 식 → 적용 가능 검증 → 실행.\n"
             "이득 없으면 PP-X로 되돌림.",
             16, C["ink"])
    foot(s, p(), TOTAL)

    # Fix page numbers in footers (TOTAL may mismatch if count drifted)
    actual = len(prs.slides)
    for index, slide in enumerate(prs.slides, 1):
        for shape in slide.shapes:
            if not hasattr(shape, "text_frame"):
                continue
            for paragraph in shape.text_frame.paragraphs:
                for run in paragraph.runs:
                    if re.fullmatch(r"\d+/\d+", run.text.strip()):
                        run.text = f"{index}/{actual}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        prs.save(str(OUT))
        print(f"Saved {OUT} ({actual} slides: talk 1-15, appendix divider, appendix)")
    except PermissionError:
        alt = OUT.with_name("PP-X_Talk_20min_alt.pptx")
        prs.save(str(alt))
        print(f"Primary locked; saved {alt} ({actual} slides)")


if __name__ == "__main__":
    build()
