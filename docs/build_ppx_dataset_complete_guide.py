#!/usr/bin/env python3
"""Build a beginner-friendly Korean guide to every dataset used in the PP-X deck."""
from pathlib import Path
import shutil
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
DOCS = Path(__file__).resolve().parent
if str(DOCS) not in sys.path:
    sys.path.insert(0, str(DOCS))
from ppt_deck_detail_section import append_ppt_deck_detail  # noqa: E402

OUT = ROOT / "docs" / "PPX_PPT_DATASET_COMPLETE_GUIDE_KO.pdf"
OUT_MIRROR = ROOT.parent / "pp_pae_total" / "output" / "PP_Research_Detailed_Study.pdf"
RESULT_FIG = ROOT / "docs" / "ppx_dataset_result_summary.png"
SCATTER_FIG = ROOT / "docs" / "ppx_actual_vs_predicted.png"
TRAJECTORY_FIG = ROOT / "docs" / "ppx_actual_vs_predicted_representative_units.png"
ARCH_FIG = ROOT / "docs" / "ppx_model_architecture.png"
TABPFN_FIG = ROOT / "docs" / "tabpfn_v3_architecture.png"
BASELINE_FIG = ROOT / "docs" / "ppx_baseline_algorithm_map.png"
FONT = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
pdfmetrics.registerFont(TTFont("AppleGothic", FONT))

INK = colors.HexColor("#1E293B")
BLUE = colors.HexColor("#075985")
SKY = colors.HexColor("#E0F2FE")
PALE = colors.HexColor("#F8FAFC")
ORANGE = colors.HexColor("#C2410C")
RULE = colors.HexColor("#CBD5E1")
RED = colors.HexColor("#991B1B")

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", fontName="AppleGothic", fontSize=25, leading=34,
                            textColor=INK, alignment=TA_CENTER, spaceAfter=10),
    "sub": ParagraphStyle("sub", fontName="AppleGothic", fontSize=12, leading=18,
                          textColor=BLUE, alignment=TA_CENTER, spaceAfter=8),
    "h1": ParagraphStyle("h1", fontName="AppleGothic", fontSize=17, leading=23,
                         textColor=BLUE, spaceBefore=4, spaceAfter=9),
    "h2": ParagraphStyle("h2", fontName="AppleGothic", fontSize=13, leading=18,
                         textColor=INK, spaceBefore=8, spaceAfter=5),
    "body": ParagraphStyle("body", fontName="AppleGothic", fontSize=9.4, leading=14.2,
                           textColor=INK, spaceAfter=5),
    "small": ParagraphStyle("small", fontName="AppleGothic", fontSize=7.8, leading=11,
                            textColor=colors.HexColor("#475569"), spaceAfter=3),
    "box": ParagraphStyle("box", fontName="AppleGothic", fontSize=9.2, leading=14,
                          textColor=INK, leftIndent=7, rightIndent=7, spaceBefore=5, spaceAfter=7),
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def table(headers, rows, widths=None, size=7.7):
    data = [[P(f"<b>{x}</b>", "small") for x in headers]]
    data += [[P(str(x), "small") for x in row] for row in rows]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "AppleGothic"),
        ("FONTSIZE", (0, 0), (-1, -1), size),
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), PALE),
        ("GRID", (0, 0), (-1, -1), .4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def callout(text, color=SKY):
    t = Table([[P(text, "box")]], colWidths=[174 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("BOX", (0, 0), (-1, -1), .8, BLUE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(BLUE)
    canvas.line(18 * mm, 285 * mm, 192 * mm, 285 * mm)
    canvas.setFont("AppleGothic", 7.5)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(18 * mm, 9 * mm, "PP-X 발표 데이터셋 완전 학습 가이드 · 박진서")
    canvas.drawRightString(192 * mm, 9 * mm, str(doc.page))
    canvas.restoreState()


DATASETS = [
    {
        "name": "HUST (화중과기대 배터리)", "role": "메인 9-setting", "domain": "리튬이온 배터리 열화",
        "thing": "서로 다른 충전 프로토콜로 반복 충·방전한 셀. 질문은 ‘지금 상태에서 수명이 몇 사이클 남았나?’이다.",
        "split": "충전 프로토콜 1–6의 셀 중 capacity ≥ 1.05 Ah를 train. 프로토콜 7–8의 capacity가 train 최저보다 낮은 행을 validation. 프로토콜 9–10의 같은 late-tail 행을 test. 프로토콜이 갈라지므로 물리 셀도 겹치지 않는다.",
        "x": [
            ("capacity_ah", "현재 방전용량(Ah)", "배터리가 지금 담을 수 있는 전하량. 작아질수록 열화가 진행됨."),
            ("cycle", "현재 사이클 번호", "몇 번째 충·방전인지 나타내는 시간축."),
            ("prefix_rate", "처음부터 현재까지 용량 감소 기울기", "현재까지 모든 과거만으로 구한 장기 열화 속도. 미래값 미사용."),
            ("recent_rate", "최근 50사이클 용량 감소 속도", "최근 열화가 빨라졌는지 보는 단기 속도. 초반에는 prefix_rate 사용."),
        ],
        "y": "RUL(남은 사이클). 단, 다수 셀이 0.880 Ah에 실제 도달하기 전에 기록이 끝나므로 마지막 20% 구간의 기울기를 연장해 proxy EOL을 만들고, proxy EOL cycle − 현재 cycle로 계산한다.",
        "coord": "capacity_ah 1개 축. test/validation 용량이 train에서 본 최저 용량보다 더 낮아 1D hull-out.",
        "warning": "이 Y는 완전 관측 수명이 아니라 <b>우측 검열 자료의 기울기 기반 대리정답</b>이다. ‘실제 고장까지 관측된 RUL’이라고 쓰면 틀린다.",
    },
    {
        "name": "Virkler (알루미늄 피로균열)", "role": "메인 9-setting", "domain": "구조재 피로·균열 성장",
        "thing": "68개 알루미늄 시편에 반복 하중을 주며 균열이 커지는 과정을 측정. 배터리가 아니라 금속판 문제다.",
        "split": "seed 42로 시편 68개를 섞어 48개 train, 10개 validation, 10개 test. train은 균열 길이 ≤33 mm, validation/test는 처음 보는 시편의 >33 mm만 사용.",
        "x": [
            ("crack_length_mm", "현재 균열 길이(mm)", "금이 얼마나 길어졌는지."),
            ("elapsed_kcycles", "현재까지 반복 하중 횟수(kcycle)", "금속판을 지금까지 몇 천 번 흔들었는지."),
            ("prefix_rate_mm_per_kcycle", "전체 과거 균열 성장속도", "시작부터 현재까지의 선형 기울기."),
            ("recent_rate_mm_per_kcycle", "직전 구간 성장속도", "가장 최근 두 측정점 사이에서 금이 자란 속도."),
        ],
        "y": "49.8 mm 파단 기준까지 남은 반복 하중(kcycle). 재구성된 최종 49.8 mm 시점의 누적 cycle − 현재 누적 cycle.",
        "coord": "crack_length_mm. train 최대 33 mm를 넘어선 39 mm와 49.8 mm 쪽 tail을 평가.",
        "warning": "Virkler는 후보 물리식(Paris 법칙)을 닫을 수 있는 예시다. 그러나 PP-X 표의 X는 위 4개 인과 특징이며, Paris 예측값을 일반 입력으로 몰래 넣은 것이 아니다.",
    },
    {
        "name": "NASA PCoE 배터리", "role": "메인 9-setting", "domain": "실험용 18650 리튬이온 셀",
        "thing": "NASA 연구소가 실험한 B0005, B0006, B0007, B0018 셀. 로켓 비행 데이터가 아니다.",
        "split": "4-fold leave-one-cell-out. 매 fold에서 1셀 test, 다음 1셀 validation, 나머지 2셀 train. train은 정규화 건강도 φ≥0.5, val/test는 train의 최저 φ보다 더 낮은 행만 사용.",
        "x": [
            ("health_phi", "(현재용량−1.4)/(초기용량−1.4)", "초기에는 약 1, 1.4 Ah 고장경계에서는 0이 되는 무차원 건강도. 최종 paper route의 기본 입력은 이 1개다."),
        ],
        "y": "처음으로 용량이 1.401 Ah 이하가 되는 방전 cycle까지 남은 cycle 수.",
        "coord": "health_phi 1개 축. test는 train 최저 φ 아래.",
        "warning": "‘NASA’라는 이름만 보고 엔진 데이터와 혼동하면 안 된다. 이 행은 <b>배터리 셀 4개</b>다.",
    },
    {
        "name": "Sunwoda", "role": "메인 9-setting·개발 3배터리", "domain": "상용 리튬이온 셀",
        "thing": "선우다 상용셀을 25°C와 35°C에서 열화시킨 자료.",
        "split": "25°C 셀 9개 중 앞 7개 train, 뒤 2개 validation, 35°C 셀 9개 test. 먼저 source-only early region을 만든 뒤, train-window endpoint health의 25% 분위수보다 건강한 train과 더 나쁜 validation/test tail을 사용.",
        "x": [
            ("health_last", "8-step 창의 마지막 건강도", "Sunwoda에서는 방전용량(mAh). 현재 상태."),
            ("rate_last", "마지막 누적 열화속도", "과거 전체로 계산한 현재 감소속도."),
            ("health_mean", "최근 8점 건강 평균", "단일 측정 잡음을 줄인 최근 수준."),
            ("rate_mean", "최근 8점 속도 평균", "최근 평균 열화속도."),
            ("health_change", "마지막 health−첫 health", "최근 8점 동안 얼마나 감소했는지."),
            ("rate_change", "마지막 rate−첫 rate", "열화속도가 빨라지는지/느려지는지."),
            ("cycle_norm", "현재 cycle/train 최대 cycle", "시간 진행도를 train 기준으로 무차원화."),
        ],
        "y": "최초 880 mAh 도달 cycle − 현재 cycle.",
        "coord": "health_last. test endpoint가 train 최저 건강도 아래.",
        "warning": "온도 이동(25→35°C)과 미지 셀 late-tail이 동시에 포함된다. 1D hull-out은 health 축에 대한 말이지 7차원 X 전체 convex hull을 뜻하지 않는다.",
    },
    {
        "name": "RWTH", "role": "메인 9-setting·개발 3배터리", "domain": "아헨공대 리튬이온 셀",
        "thing": "독일 RWTH Aachen의 셀 열화 자료. 건강도는 각 셀의 첫 관측 용량으로 나눈 상대용량.",
        "split": "셀 ID 2–24 train(23개), 25–32 validation(8개), 33–40 test(8개). 서로 다른 셀. 공통 8-step 창 및 strict late-tail 규칙 적용.",
        "x": [("7개 요약 X", "Sunwoda와 동일한 health/rate 8-step 요약", "last·mean·endpoint change 각각 2개 + normalized cycle 1개.")],
        "y": "상대용량이 처음 0.8 이하가 될 때까지 남은 cycle.",
        "coord": "상대 health. test/validation은 train의 late boundary 아래.",
        "warning": "RWTH는 raw 용량을 첫 관측 용량으로 나눈다. MICH의 ‘공개 nominal capacity’ 정규화와 다르다.",
    },
    {
        "name": "MICH", "role": "메인 9-setting·개발 3배터리", "domain": "미시간대 리튬이온 셀",
        "thing": "미시간대 배터리 수명 자료. 학습 셀과 시험 셀 사이의 health↔RUL 관계 이동이 커서 고정 bound가 실패한 반례.",
        "split": "셀 1–18 train, 19–24 validation, 25–32 test. 8-step causal window. test는 처음 보는 셀의 train-health-support 아래 tail.",
        "x": [("7개 요약 X", "Sunwoda와 동일한 health/rate 8-step 요약", "현재·평균·변화량과 normalized cycle.")],
        "y": "공개된 MICH life label − 현재 cycle.",
        "coord": "공개 nominal capacity로 정규화한 health.",
        "warning": "MICH Y는 단순히 코드에서 80% crossing을 다시 찾지 않고 <b>공개 life label</b>을 쓴다. 고정 boundary executor가 나빠지고 dual-scale이 필요했던 핵심 반례다.",
    },
    {
        "name": "MATR 2019", "role": "메인 9-setting", "domain": "MIT/Stanford 고속충전 배터리 수명 벤치",
        "thing": "2019-01-24 공개 batch의 셀별 방전용량 궤적. 다양한 고속충전 조건에서 셀 수명을 예측한다.",
        "split": "유효 셀을 원래 순서대로 60% train, 20% validation, 20% test. train 셀의 endpoint 용량 25% 분위수로 초기 cutoff를 만들고, train은 그보다 높은 용량, val/test는 실제 train 최저 용량보다 낮은 행.",
        "x": [
            ("Q_last", "최근 8점의 마지막 방전용량", "현재 건강."),
            ("rate_last", "마지막 비음수 용량감소량", "직전 cycle의 열화량."),
            ("Q_mean / rate_mean", "최근 8점 평균 2개", "최근 수준과 평균 속도."),
            ("Q_change / rate_change", "최근 창 양끝 차이 2개", "최근 추세와 가속 변화."),
        ],
        "y": "해당 셀 기록의 마지막 cycle − 현재 cycle.",
        "coord": "Q_last. train 최저 Q보다 낮은 val/test만 남김.",
        "warning": "여기서 EOL은 코드상 각 셀 파일의 마지막 기록점이다. 별도의 80% threshold RUL이라고 바꿔 말하지 않는다.",
    },
    {
        "name": "MATR batch 2", "role": "메인 9-setting", "domain": "MIT/Stanford 배터리 다른 생산 배치",
        "thing": "2017-06-30 batch의 정확히 48개 셀. 같은 연구계열이지만 MATR2019와 파일·셀·split이 다르다.",
        "split": "셀 0–29(30개) train, 30–38(9개) validation, 39–47(9개) test. train endpoint의 25% 용량 분위수로 자르고 100% strict 1D tail 확인.",
        "x": [("6개 요약 X", "MATR2019와 동일", "Q/rate의 last, mean, endpoint change. 이 confirmatory row builder에는 cycle_norm이 없다.")],
        "y": "셀 마지막 cycle − 현재 cycle.",
        "coord": "현재 방전용량. 1D hull-out 100%; 단 PCA 2D hull-out은 0%였으므로 ‘전체 다차원 공간 밖’이라 하면 안 됨.",
        "warning": "0.862는 고정 split을 본 뒤 개발된 최종 route다. 미개봉 confirmatory 수치 0.523과 증거 등급이 다르다.",
    },
    {
        "name": "N-CMAPSS DS02-006", "role": "메인 9-setting", "domain": "NASA 디지털 터보팬 엔진 시뮬레이션",
        "thing": "실제 비행기 센서 로그가 아니라 고충실도 시뮬레이터가 만든 엔진 열화 궤적. 처음 보는 엔진과 높은 TRA 조건의 late life를 다룬다.",
        "split": "TRA-hard split을 사용하며 물리 엔진(unit)을 분리한다. 발표의 0.937은 retrospective DS02-006 portfolio split. prospective DS03의 train/val/test와는 별개.",
        "x": [
            ("운전조건 4개", "alt, Mach, TRA, T2", "고도, 비행속도, 스로틀, 입구온도."),
            ("실측 센서 14개", "T24,T30,T48,T50,P15,P2,P21,P24,Ps30,P40,P50,Nf,Nc,Wf", "온도 T, 압력 P, 축속도 N, 연료유량 Wf. 숫자는 엔진 station 식별자."),
            ("가상센서 14개", "T40,P30,P45,W21,W22,W25,W31,W32,W48,W50,SmFan,SmLPC,SmHPC,phi", "시뮬레이터에서 계산된 내부 상태·유량·stall margin 계열."),
        ],
        "y": "엔진이 고장 시점까지 버틸 남은 cycle/시간 인덱스인 RUL.",
        "coord": "발표의 hard TRA 축에서는 1D hull-out 100%. PCA2는 0%, PCA3는 83.6%, target-out은 0%이므로 ‘먼 RUL 외삽’보다 <b>운전조건 regime 외삽</b>에 가깝다.",
        "warning": "센서 약어의 정확한 물리 station 정의는 N-CMAPSS 공식 데이터 설명서를 따라야 한다. 코드에서 확인되는 이름 이상을 임의로 풀어 쓰지 않았다.",
    },
]

COUNTS = {
    "HUST (화중과기대 배터리)": ("77셀", "45셀·11,250행", "16셀·2,560행", "16셀·7,775행", "0.958", "10/16"),
    "Virkler (알루미늄 피로균열)": ("68시편", "48시편·240행", "10시편·20행", "10시편·20행", "0.888", "8/10"),
    "NASA PCoE 배터리": ("4셀", "fold마다 2셀·118~125행", "fold마다 1셀·late-tail", "fold마다 1셀; 합계 255행", "0.584", "2/4"),
    "Sunwoda": ("18셀", "7셀·1,750창", "2셀·500창", "9셀·1,468창", "0.939", "9/9"),
    "RWTH": ("39셀(ID 2~40)", "23셀·3,060창", "8셀·976창", "8셀·1,859창", "0.878", "7/8"),
    "MICH": ("32셀", "18셀·4,500창", "6셀·1,500창", "8셀·202창", "0.751", "6/8"),
    "MATR 2019": ("원본 45셀, 유효 44셀", "26셀·17,003행", "8셀·1,306행", "10셀·2,235행", "0.466", "6/10"),
    "MATR batch 2": ("48셀", "30셀·11,553행", "9셀·673행", "9셀·733행", "0.862", "9/9"),
    "N-CMAPSS DS02-006": ("9엔진(개발6+공식시험3)", "hard split의 개발 엔진", "별도 개발 엔진/조건", "3엔진·159행", "0.937", "3/3"),
}


def make_result_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["HUST", "Virkler", "NASA batt.", "Sunwoda", "RWTH", "MICH", "MATR19", "MATRb2", "N-CMAPSS"]
    r2 = [.958, .888, .584, .939, .878, .751, .466, .862, .937]
    test_units = [16, 10, 4, 9, 8, 8, 10, 9, 3]
    wins = [10/16, 8/10, 2/4, 9/9, 7/8, 6/8, 6/10, 9/9, 3/3]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.3))
    ax[0].barh(names[::-1], r2[::-1], color="#0284c7")
    ax[0].set_xlim(0, 1); ax[0].set_title("PP-X test pooled R²")
    ax[0].axvline(0, color="black", lw=.7)
    for i, v in enumerate(r2[::-1]): ax[0].text(v + .015, i, f"{v:.3f}", va="center", fontsize=8)
    ax[1].barh(names[::-1], test_units[::-1], color="#0f766e")
    ax[1].set_title("Test physical units")
    for i, v in enumerate(test_units[::-1]): ax[1].text(v + .2, i, str(v), va="center", fontsize=8)
    ax[2].barh(names[::-1], wins[::-1], color="#ea580c")
    ax[2].set_xlim(0, 1.05); ax[2].set_title("Units won vs paired baseline")
    ax[2].set_xlabel("fraction")
    fig.suptitle("Stored PP-X result artifacts (retrospective main 9 settings)", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(RESULT_FIG, dpi=190, bbox_inches="tight")
    plt.close(fig)


def load_actual_predictions():
    """Load row-aligned frozen paper-route predictions without retraining."""
    import sys
    import numpy as np
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments"),
                    str(ROOT.parent / "ca-css-ncmapss")]
    load = lambda p: np.load(ROOT / p, allow_pickle=True)
    out = {}
    z = load("results/hust_regime_transport_pp_v1/predictions.npz")
    out["HUST"] = (z["y"], z["groups"].astype(str), z["transported"].mean(0))
    z = load("results/support_gated_cross_domain_v1/predictions_virkler.npz")
    out["Virkler"] = (z["truth"], z["groups"].astype(str),
                      np.stack([z[f"gated_seed{s}"] for s in range(42, 47)]).mean(0))
    z = load("results/nasa_causal_multiscale_pp_v1/predictions.npz")
    out["NASA batt."] = (z["y"], z["groups"].astype(str), z["prediction"].mean(0))
    final = load("results/bq_dual_scale_final_replay_v1/predictions.npz")
    fixed = load("results/bq_pp_matched_controls_v1/predictions.npz")
    for index, name in enumerate(("Sunwoda", "RWTH", "MICH")):
        mask = final["dataset"] == index
        pred = final["prediction"][:, mask] if name == "MICH" else fixed["bq_pp"][:, mask]
        out[name] = (final["y"][mask], final["units"][mask].astype(str), pred.mean(0))
    z = load("results/matr_pp_validation_calibration_v1/predictions.npz")
    out["MATR19"] = (z["y"], z["groups"].astype(str), z["prediction"].mean(0))
    z = load("results/matr_batch2_pp_five_seed_replay/predictions.npz")
    out["MATRb2"] = (z["truth"], z["groups"].astype(str), z["transported"].mean(0))
    from final_modular_pp_evidence import ncmapss_final
    y, groups, pred = ncmapss_final()
    out["N-CMAPSS"] = (y, groups, pred.mean(0))
    return out


def make_actual_prediction_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    data = load_actual_predictions()
    def r2_score(y, pred):
        den = float(np.sum((y - np.mean(y)) ** 2))
        return float(1 - np.sum((y - pred) ** 2) / den) if den > 0 else float("nan")

    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    for ax, (name, (y, groups, pred)) in zip(axes.flat, data.items()):
        y, pred = np.asarray(y, float), np.asarray(pred, float)
        ax.scatter(y, pred, s=7, alpha=.28, color="#0284c7", edgecolors="none")
        lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "--", color="#ea580c", lw=1)
        ax.set_title(f"{name}  ·  pooled R² = {r2_score(y, pred):.3f}", fontsize=10, weight="bold")
        ax.set_xlabel("Actual Y"); ax.set_ylabel("Predicted Y")
        ax.grid(alpha=.15)
    fig.suptitle("Actual RUL vs PP-X ensemble prediction — every test row", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(SCATTER_FIG, dpi=190, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    for ax, (name, (y, groups, pred)) in zip(axes.flat, data.items()):
        y, groups, pred = np.asarray(y, float), np.asarray(groups), np.asarray(pred, float)
        units, counts = np.unique(groups, return_counts=True)
        unit = units[np.argmax(counts)]
        mask = groups == unit
        ax.plot(np.arange(mask.sum()), y[mask], color="#0f172a", lw=1.8, label="Actual Y")
        ax.plot(np.arange(mask.sum()), pred[mask], color="#0284c7", lw=1.4, label="PP-X prediction")
        unit_r2 = r2_score(y[mask], pred[mask])
        ax.set_title(f"{name} · unit {unit} · R²={unit_r2:.3f}", fontsize=9, weight="bold")
        ax.set_xlabel("Test-row order"); ax.set_ylabel("RUL")
        ax.grid(alpha=.15)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Representative test unit: actual and predicted RUL", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(TRAJECTORY_FIG, dpi=190, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        ROOT / "docs" / "ppx_actual_vs_predicted_all_main_datasets.npz",
        **{f"{name}_{field}": value for name, values in data.items()
           for field, value in zip(("y_true", "groups", "y_pred_ensemble"), values)}
    )


def make_architecture_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis("off")
    def box(x, y, w, h, text, color, fs=10):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.03,rounding_size=.08",
                           facecolor=color, edgecolor="#334155", linewidth=1.1)
        ax.add_patch(p); ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs)
    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, color="#475569", linewidth=1.2))
    box(.2, 4.5, 2.0, .8, "Typed contract\nboundary · units · prior", "#e0f2fe")
    box(2.7, 4.5, 2.0, .8, "Causal adapter\ncurrent · rate · history", "#e0f2fe")
    box(5.2, 4.5, 2.0, .8, "Train-only scale\nz=(x−μ)/σ", "#e0f2fe")
    box(7.7, 4.5, 2.1, .8, "Validation approval\nexecutor · fallback", "#ffedd5")
    box(10.3, 4.5, 2.3, .8, "Frozen test route\nno test-time routing", "#dcfce7")
    for x in (2.2, 4.7, 7.2, 9.8): arrow(x, 4.9, x+.5, 4.9)
    box(.6, 2.3, 2.4, 1.0, "Prior path\nLinear(d→1) or quotient\nfit on train → FROZEN", "#dbeafe")
    box(4.0, 2.3, 2.5, 1.0, "Residual network\nLinear(d→w) → tanh\n→ Linear(w→w) → tanh\n→ Linear(w→1)", "#fef3c7", 9)
    box(7.5, 2.3, 2.4, 1.0, "Authority envelope\nb(z)·tanh(r/b)\nfixed or dual-scale", "#fee2e2")
    box(10.8, 2.3, 1.7, 1.0, "Decoder D\nRUL≥0\nboundary→0", "#dcfce7")
    arrow(3.0, 2.8, 4.0, 2.8); arrow(6.5, 2.8, 7.5, 2.8); arrow(9.9, 2.8, 10.8, 2.8)
    box(3.0, .5, 2.1, .8, "Optional history\nshort / long / multiscale", "#f8fafc")
    box(5.5, .5, 2.1, .8, "Support gate\nresidual decay / dual scale", "#f8fafc")
    box(8.0, .5, 2.1, .8, "Regime transport\nOOF scale / offset", "#f8fafc")
    arrow(4.05, 1.3, 5.0, 2.3); arrow(6.55, 1.3, 8.1, 2.3); arrow(9.05, 1.3, 9.0, 2.3)
    ax.text(6.5, 5.75, "PP-X = selection framework outside + prior-residual executor inside",
            ha="center", va="center", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(ARCH_FIG, dpi=190, bbox_inches="tight")
    plt.close(fig)


def make_tabpfn_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.2); ax.axis("off")
    def box(x, y, w, h, text, color, fs=9):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.03,rounding_size=.08",
                           facecolor=color, edgecolor="#334155", linewidth=1.1)
        ax.add_patch(p); ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs)
    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, color="#475569", linewidth=1.2))
    box(.2, 2.9, 1.7, 1.1, "Train table\nXtrain + ytrain", "#dbeafe")
    box(.2, 1.1, 1.7, 1.1, "Test rows\nXtest only", "#e0f2fe")
    box(2.4, 2.0, 2.0, 1.4, "Feature grouping\ncyclic triplets\ncell embeddings\n+ target-aware train embed", "#fef3c7", 8.5)
    box(4.9, 2.0, 2.0, 1.4, "Stage 1\nColumn distribution embedder\ninducing-point attention", "#ffedd5", 8.5)
    box(7.4, 2.0, 2.0, 1.4, "Stage 2\nRow-wise aggregation\nlearned CLS tokens", "#fee2e2", 8.5)
    box(9.9, 2.0, 2.0, 1.4, "Stage 3\nICL Transformer\ntrain↔train, test→train", "#dcfce7", 8.5)
    box(10.2, .3, 1.4, .8, "Regression\ndecoder", "#e9d5ff")
    box(12.1, .3, .7, .8, "ŷ", "#d1fae5", 12)
    arrow(1.9, 3.45, 2.4, 2.9); arrow(1.9, 1.65, 2.4, 2.5)
    arrow(4.4, 2.7, 4.9, 2.7); arrow(6.9, 2.7, 7.4, 2.7); arrow(9.4, 2.7, 9.9, 2.7)
    arrow(10.9, 2.0, 10.9, 1.1); arrow(11.6, .7, 12.1, .7)
    ax.text(6.5, 4.75, "TabPFN v3: pretrained tabular in-context learner",
            ha="center", fontsize=15, weight="bold")
    ax.text(6.5, 4.35, "No gradient-based task-specific weight training in the usual sense; prediction uses the table as context",
            ha="center", fontsize=9, color="#475569")
    fig.tight_layout()
    fig.savefig(TABPFN_FIG, dpi=190, bbox_inches="tight")
    plt.close(fig)


def make_baseline_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13); ax.set_ylim(0, 6); ax.axis("off")
    items = [
        ("Plain MLP", "Dense nonlinear\npoint predictor", "#dbeafe"),
        ("FT-Transformer", "Feature tokens\nself-attention", "#e0f2fe"),
        ("V-REx", "Variance of\ngroup risks", "#dcfce7"),
        ("GroupDRO", "Worst-group\nrobust loss", "#d1fae5"),
        ("Monotone NN", "Derivative-sign\npenalty", "#fef3c7"),
        ("Engression", "Noise-injected\ngenerative regression", "#ffedd5"),
        ("Linear-tail RBF", "Linear trend +\nrandom Fourier RBF", "#fee2e2"),
        ("SVGP", "Kernel GP +\ninducing points", "#f3e8ff"),
        ("TabPFN v3", "Pretrained\nin-context Transformer", "#e9d5ff"),
    ]
    for i, (name, body, color) in enumerate(items):
        row, col = divmod(i, 3); x=.35+col*4.25; y=4.0-row*1.65
        p=FancyBboxPatch((x,y),3.75,1.15,boxstyle="round,pad=.03,rounding_size=.08",
                         facecolor=color,edgecolor="#334155",linewidth=1.1)
        ax.add_patch(p); ax.text(x+.18,y+.76,name,fontsize=11,weight="bold")
        ax.text(x+.18,y+.22,body,fontsize=9)
    ax.text(6.5,5.6,"Compared algorithm families",ha="center",fontsize=15,weight="bold")
    ax.text(6.5,.25,"Same frozen train/validation/test adapters; TabPFN is a separate capped supplementary comparison",
            ha="center",fontsize=9,color="#475569")
    fig.tight_layout()
    fig.savefig(BASELINE_FIG,dpi=190,bbox_inches="tight")
    plt.close(fig)


def dataset_pages(story, d):
    story.append(P(d["name"], "h1"))
    story.append(table(["발표 역할", "도메인", "무엇을 예측?"],
                       [[d["role"], d["domain"], d["thing"]]],
                       [33 * mm, 42 * mm, 99 * mm]))
    total, tr, va, te, r2, wins = COUNTS[d["name"]]
    story.append(P("제품/부품은 실제로 몇 개인가", "h2"))
    story.append(table(["전체 고유 unit", "Train", "Validation", "Test", "PP-X 결과"],
                       [[total, tr, va, te, f"pooled R² {r2}<br/>unit 승리 {wins}"]],
                       [30 * mm, 38 * mm, 35 * mm, 37 * mm, 34 * mm]))
    story.append(P("즉, 제품 하나를 외워 여러 시점을 맞힌 실험이 아니다. 여러 train 제품에서 규칙을 배우고, "
                   "학습 때 한 번도 보지 않은 validation/test 제품의 더 나쁜 구간을 예측했다. "
                   "NASA 배터리는 제품 수가 4개뿐이라 leave-one-cell-out을 돌린 예외다.", "small"))
    story.append(P("스플릿을 그림 없이 말로 풀면", "h2"))
    story.append(P(d["split"]))
    story.append(P("X 컬럼 사전", "h2"))
    story.append(table(["코드 이름/묶음", "쉬운 뜻", "모델이 얻는 정보"], d["x"],
                       [42 * mm, 54 * mm, 78 * mm]))
    story.append(P("Y(정답)와 외삽 좌표", "h2"))
    story.append(table(["Y: 모델이 맞힐 값", "hull-out을 판정한 좌표"],
                       [[d["y"], d["coord"]]], [88 * mm, 86 * mm]))
    story.append(Spacer(1, 4 * mm))
    story.append(callout("<b>절대 헷갈리지 말 것:</b> " + d["warning"],
                         colors.HexColor("#FFF7ED")))
    story.append(PageBreak())


def append_ppx_defense(story):
    """Append the model-theory, ablation, statistics, and oral-defense handbook."""
    story += [
        P("부록 A. PP-X를 이해하는 가장 중요한 생각", "h1"),
        P("<b>외삽에서는 데이터가 하나의 답을 정해 주지 못한다.</b> Train 범위 안에서는 여러 곡선이 모두 비슷하게 "
          "맞지만, 범위 밖에서는 직선으로 갈지, 휘어질지, 다시 증가할지 데이터만으로 결정할 수 없다. "
          "그래서 PP-X는 ‘어떤 NN이 가장 복잡한가’보다 ‘어떤 구조 가정을 사용할 자격이 있는가’를 먼저 묻는다."),
        table(["핵심 이념", "뜻", "왜 필요한가"], [
            ["Declare", "고장경계, unit, 시간, causal X, 외삽좌표, 허용 prior, fallback을 test 전에 선언",
             "결과를 본 뒤 유리한 정의로 바꾸는 일을 막음"],
            ["Learn", "동결 prior 주변에서 source가 지지하는 residual만 학습",
             "prior만으로 못 잡는 곡률은 배우되 tail 방향을 NN이 마음대로 덮지 못하게 함"],
            ["Approve", "동일 unit-disjoint validation에서 optional executor의 이득과 unit 위험을 확인",
             "모듈이 있다는 이유만으로 항상 켜지 않음"],
            ["Decline", "근거가 부족하면 미리 정한 direct/persistence fallback 또는 abstention",
             "틀린 prior를 강제로 출력하는 것보다 거절이 정직함"],
        ], [29 * mm, 77 * mm, 68 * mm]),
        callout("<b>한 줄 정의:</b> PP-X는 contract가 허용한 prior-residual 후보 중 validation evidence가 지지하는 "
                "executor 하나를 시험 전에 동결하고, 지지되지 않으면 fallback/abstention하는 외삽 framework다."),
        PageBreak(),

        P("부록 B. 모델 식을 말로 해체하기", "h1"),
        P("논문용 공통 표현은 <b>ŷ = D{ y_prior + b(z)·tanh[r(z)/b(z)] }</b>이다."),
        table(["기호", "역할", "초보자식 비유"], [
            ["z", "train 평균·표준편차로 변환한 causal feature", "단위가 다른 단서들을 같은 눈금으로 맞춤"],
            ["y_prior", "train에서 정한 뒤 고정한 affine 또는 boundary-quotient tail", "밖으로 갈 기본 철길"],
            ["r(z)", "NN이 학습한 nonlinear residual", "철길에서 필요한 만큼만 미세 조정"],
            ["b(z)", "residual 수정량의 허용 폭", "핸들을 꺾을 수 있는 최대 각도"],
            ["tanh", "수정량을 유한 범위에 포화", "NN이 밖에서 무한히 폭주하지 못하게 함"],
            ["D", "비음수 RUL, 경계에서 RUL=0 같은 domain decoder", "말이 안 되는 음수 수명을 차단"],
        ], [25 * mm, 76 * mm, 73 * mm]),
        P("Affine을 왜 동결하나", "h2"),
        P("Train 안에서는 NN이 affine path까지 함께 바꾸면 loss를 더 잘 줄일 수 있지만, 그 변화가 support 밖에서도 "
          "옳다는 보장은 없다. 동결은 ‘성능을 무조건 올리는 마법’이 아니라, 외삽의 기준선을 식별 가능하게 만들고 "
          "residual이 무엇을 수정했는지 분리하기 위한 parameterization이다. 단독 유의성은 확증되지 않았으므로 과장하지 않는다."),
        P("Boundary quotient란", "h2"),
        P("알려진 실패경계까지의 margin을 출력에 곱하거나 좌표화해, 경계에 가까워질수록 RUL이 0으로 가게 만드는 방법이다. "
          "고장경계가 독립적으로 정당화된 데이터에서만 admissible하다. 경계만 안다고 열화 메커니즘 전체를 안다는 뜻은 아니다."),
        PageBreak(),

        P("부록 C. Executor와 gate는 무엇이 다른가", "h1"),
        table(["구성", "무엇을 하는가", "승인 조건", "대표 결과"], [
            ["Unbounded core", "frozen prior + 제한 없는 residual", "기본 admissible reference", "MICH 0.759이나 Sun/RWTH에는 과보정"],
            ["Fixed bound", "모든 점에서 residual 폭을 고정 제한", "validation이 unbounded보다 개선", "Sun +.221, RWTH +.090, MICH −.291"],
            ["Dual scale", "support 상태에 따라 local bound 2와 broad bound 6을 혼합", "fixed 대비 개선 + train support heterogeneity≥.50", "MICH +.283; RWTH에는 악화"],
            ["Support decay", "train support에서 멀수록 NN residual 감쇠", "validation-selected", "MATRb2 단독 +.002; seed SD 감소"],
            ["Regime transport", "source group에서 반복되는 scale/offset 오차를 OOF 방식으로 전달", "group-LOO에서 일관된 개선", "HUST +.128, MATRb2 +.187"],
            ["History route", "짧은/긴 causal history 표현 선택", "validation loss 개선", "NASA +.012, N-CMAPSS +.009"],
            ["Evidence gate", "optional executor를 켤지 거절할지 결정", "validation gain·unit evidence", "13개 중 3 개선, 10 유지, 0 악화"],
            ["Safety fallback", "prior가 부적합할 때 direct/persistence 또는 abstain", "prior admissibility 실패", "DS03에서 direct fallback 선택 성공"],
        ], [27 * mm, 57 * mm, 48 * mm, 42 * mm], size=7.1),
        P("Gate는 예측식을 새로 만드는 부품이 아니다", "h2"),
        P("Gate의 1차 목적은 평균점수를 직접 올리는 것이 아니라, 틀릴 가능성이 큰 optional executor를 실행하지 못하게 "
          "막는 것이다. 따라서 ‘gate를 켜서 10개는 그대로였다’는 실패가 아니라 보존 동작이다. 그러나 MICH history처럼 "
          "validation이 잘못 승인한 반례가 있어 완벽한 미래 판별기라고 주장하지 않는다."),
        PageBreak(),

        P("부록 D. 데이터셋별 어떤 executor가 왜 도움됐나", "h1"),
        table(["데이터셋", "최종/대표 executor", "도움된 이유", "주의"], [
            ["HUST", "regime transport", "충전 프로토콜별 출력 scale 오차가 validation group-LOO에서 반복", "raw .829→.958; post-hoc development"],
            ["Virkler", "support-gated residual", "긴 균열 tail에서 residual 폭주를 줄임", "동일예산 FT가 .002 높아 8/9의 유일한 미승리"],
            ["NASA battery", "validation-selected causal multiscale", "셀별 열화 속도의 유효 history 길이가 다름", ".572→.584로 이득은 작음"],
            ["Sunwoda", "fixed bounded BQ", "880mAh 경계가 있고 unbounded NN의 과수정이 큼", ".718→.939"],
            ["RWTH", "fixed bounded BQ", "0.8 경계와 rate history가 tail을 안정화", "dual-scale 전역 적용은 오히려 낮춤"],
            ["MICH", "support-adaptive dual scale", "고정 bound가 필요한 관계이동 correction을 차단", ".468→.751; fixed bound 반례"],
            ["MATR2019", "validation output calibration", "순위/상관은 괜찮지만 예측이 평균 25.2 cycle 낮은 scale bias", ".257→.466; test 후 개발"],
            ["MATR batch2", "decay + regime transport", "cell 간 state/rate scale 차이와 먼 support 과보정을 함께 처리", ".674→.862"],
            ["N-CMAPSS", "causal multiscale route", "운전조건과 열화의 시간척도가 다름", ".928→.937; unit 효과는 이질적"],
        ], [27 * mm, 44 * mm, 69 * mm, 34 * mm], size=7),
        callout("<b>왜 하나의 executor를 전부에 쓰지 않나:</b> fixed bound는 MICH에서 −.291, dual-scale은 RWTH에서 악화, "
                "full rate history는 MICH에서 simple history보다 −.247이었다. 이 반례가 조건부 선택의 직접 근거다."),
        PageBreak(),

        P("부록 E. Frozen 선택 순서와 임계값", "h1"),
        table(["순서", "검사", "동결 기준"], [
            ["1 Prior admissibility", "경계 prior를 쓸 수 있는가", "독립 선언 boundary가 있거나 complete source group≥5, regime당≥2"],
            ["", "source에서 prior가 direct보다 해롭지 않은가", "source-group OOF prior regret ≤0"],
            ["", "latent mode가 재학습에도 유지되는가", "필요할 때 mode stability≥0.60"],
            ["2 Executor approval", "optional 기능이 실제 개선하는가", "동일 group-disjoint validation loss를 단순 reference보다 ≥2% 감소"],
            ["", "dual-scale을 쓸 만큼 support가 이질적인가", "train-only support heterogeneity≥0.50"],
            ["", "거의 동률이면 무엇을 고르나", "수치 tolerance 안에서는 더 단순한 executor"],
            ["3 Safety", "prior가 거절되면", "사전 지정 fallback 또는 abstention; test sample별 route 변경 금지"],
        ], [24 * mm, 76 * mm, 74 * mm]),
        P("Unit-risk 강화 규칙", "h2"),
        P("단순 validation RMSE-only 선택은 12개 retrospective setting에서 7/12만 맞고 false accept 4개를 냈다. "
          "그래서 paper selector는 상대 validation gain뿐 아니라 validation physical-unit win fraction과 worst-unit RMSE ratio를 함께 본다. "
          "구현 기본값은 최소 unit 승률 60%, worst-unit RMSE ratio 1.10이다. 그래도 false accept 2개가 남아 완전한 안전보장은 아니다."),
        PageBreak(),

        P("부록 F. Matched 6-arm ablation — 무엇을 하나씩 뺐나", "h1"),
        table(["Arm", "제거·변경한 것", "Sun / RWTH / MICH ensemble R²", "질문"], [
            ["Direct NN", "prior와 affine tail 제거", "−1.352 / .633 / .684", "구조 prior 없이 NN만으로 충분한가"],
            ["Soft-boundary NN", "경계를 exact 구조가 아닌 penalty로만 적용", "−3.107 / .483 / −.057", "약한 벌점만으로 경계를 지키나"],
            ["Trainable hard-boundary", "경계는 exact, affine은 학습 중 이동", ".900 / .855 / .319", "affine 동결이 필요한가"],
            ["Affine quotient only", "nonlinear residual 제거", ".281 / .659 / −3.343", "단순 prior만으로 충분한가"],
            ["Frozen + unbounded", "residual bound 제거", ".718 / .788 / .759", "수정량 제한이 필요한가"],
            ["Frozen + bounded", "BQ core", ".939 / .878 / .468", "동결 prior+제한 residual의 결합"],
        ], [35 * mm, 52 * mm, 47 * mm, 40 * mm]),
        P("결론", "h2"),
        P("Affine-only는 MICH에서 크게 무너지고 direct NN은 Sunwoda에서 무너졌다. 즉 prior와 residual 둘 다 필요하다. "
          "동결 bounded 조합은 Sun/RWTH에 강하지만 MICH에서는 unbounded보다 낮다. 따라서 ‘bounded가 보편적으로 최고’가 아니라 "
          "core와 optional bound의 역할을 분리해야 한다."),
        PageBreak(),

        P("부록 G. 구성요소별 ΔR² ablation", "h1"),
        table(["기능", "비교", "ΔR²", "무엇을 입증/반박했나"], [
            ["Nonlinear residual", "affine→bounded", "Sun +.658, RWTH +.219, MICH +3.811", "prior-only로 부족"],
            ["Frozen affine", "trainable→frozen bounded", "+.039 / +.023 / +.149", "모두 양의 방향, 단독 유의성은 미확증"],
            ["Fixed bound", "unbounded→bounded", "+.221 / +.090 / −.291", "조건부 기능이며 MICH가 반례"],
            ["Dual scale", "MICH fixed→dual", "+.283", "relationship shift에서 넓은 correction 필요"],
            ["Full rate history", "current margin→full", "Sun +.349, RWTH +1.256", "같은 health라도 속도가 다름"],
            ["Full history 반례", "margin history→full", "MICH −.247", "history를 전역 기본값으로 둘 수 없음"],
            ["Regime transport", "raw→transport", "HUST +.128, MATRb2 +.187", "반복되는 group scale 오차 보정"],
            ["Multiscale", "basic/short→selected", "NASA +.012, N-CMAPSS +.009", "양수지만 효과·통계력 작음"],
        ], [36 * mm, 55 * mm, 39 * mm, 44 * mm]),
        P("Ablation의 올바른 목적", "h2"),
        P("Ablation은 ‘최종 점수가 높다’를 반복하는 실험이 아니다. 특정 요소 하나를 제외한 matched arm과 비교해 "
          "그 요소가 어떤 실패를 막는지 확인한다. 입력, split, seed, optimizer 예산이 다르면 제거 효과와 다른 변화가 섞이므로 matched 비교가 중요하다."),
        PageBreak(),

        P("부록 H. MATR batch2 2×2 ablation", "h1"),
        table(["Support decay", "Transport", "seed 평균±SD", "ensemble R²", "해석"], [
            ["Off", "Off", ".527±.217", ".674", "기본"],
            ["On", "Off", ".556±.148", ".675", "정확도 +.002, seed 변동 약 32% 감소"],
            ["Off", "On", ".815±.041", ".820", "transport가 주효과"],
            ["On", "On", ".852±.061", ".862", "먼 support에서 transport 과보정 완화"],
        ], [31 * mm, 31 * mm, 38 * mm, 34 * mm, 40 * mm]),
        P("왜 2×2인가", "h2"),
        P("두 기능 A와 B가 있을 때 A만, B만, 둘 다, 둘 다 없음 네 조합을 비교하면 각 기능의 주효과와 상호작용을 분리할 수 있다. "
          "여기서는 decay 단독 정확도 이득은 거의 없지만 transport와 함께 쓸 때 transport-only보다 +.042다. "
          "따라서 decay를 독립 accuracy module보다 안정화 상호작용으로 해석한다."),
        PageBreak(),

        P("부록 I. 통계검증 지도 — 무엇을 표본으로 봤나", "h1"),
        table(["검증", "표본 단위", "귀무가설 H0", "이 연구에서 묻는 질문"], [
            ["Wilcoxon signed-rank", "같은 test physical unit의 paired RMSE 차이", "차이 분포의 중앙이 0", "PP-X가 unit 전반에서 일관되게 낮은 오차인가"],
            ["Unit bootstrap", "physical unit", "직접 H0 검정이라기보다 효과 불확실성 추정", "평균 unit 효과의 95% 구간이 0을 넘는가"],
            ["Exact sign-flip", "dataset 내 paired unit log-RMSE ratio", "부호가 교환 가능, 평균 효과 0", "관측 방향성이 우연한 부호 배열로 가능한가"],
            ["Exact binomial", "optimizer seed의 route vote", "선택확률≤.5", "선택이 초기화에 안정적인가"],
            ["Dataset sign test", "서로 다른 setting", "PP-X 승리확률=.5", "여러 setting에서 우세 방향인가"],
            ["Hierarchical bootstrap", "dataset→그 안의 unit", "효과 분포의 계층 불확실성", "행 수가 큰 dataset이 결과를 독점하지 않나"],
            ["Benjamini–Hochberg", "여러 dataset별 p-value", "FDR 통제 절차", "여러 검정을 하며 우연한 유의성을 얼마나 억제했나"],
            ["Bound audit", "모든 예측점", "가설검정 아님", "설계한 residual bound를 실제 forward가 위반했나"],
        ], [35 * mm, 48 * mm, 42 * mm, 49 * mm], size=7.1),
        callout("<b>가장 중요한 원칙:</b> 한 셀의 수백 행을 수백 개 독립 실험으로 세지 않는다. "
                "물리 unit이 독립성에 가까운 추론 단위이고, seed는 optimizer 안정성이지 새로운 배터리가 아니다."),
        PageBreak(),

        P("부록 J. 각 통계방법의 이론과 해석", "h1"),
        P("Wilcoxon signed-rank", "h2"),
        P("같은 unit에서 방법 A와 B의 RMSE 차이를 짝지은 뒤, 차이의 크기에 순위를 매기고 부호가 한쪽으로 몰리는지 본다. "
          "정규분포를 강하게 가정하지 않지만 paired 차이 분포의 대칭성이 해석에 중요하다. Direct 대비 17/25, p=.0028; "
          "soft 대비 24/25, p=1.13×10⁻⁶; affine-only 대비 25/25, p=5.96×10⁻⁸였다."),
        P("미확증 비교", "h2"),
        P("Trainable hard-boundary 대비 20/25 승리지만 p=.071, unbounded 대비 14/25와 p=.578이다. "
          "p>.05는 ‘두 방법이 같다’의 증명이 아니라, 현재 표본으로 차이를 확증하지 못했다는 뜻이다."),
        P("Bootstrap", "h2"),
        P("Unit을 복원추출해 평균 효과를 수만 번 다시 계산한다. 관측 unit들이 모집단을 대표한다는 전제 아래 효과의 표본변동을 근사한다. "
          "행 bootstrap이 아니라 unit bootstrap을 써서 긴 trajectory가 가짜 표본수를 만들지 않게 했다."),
        P("Exact sign-flip", "h2"),
        P("H0 아래 paired 차이의 +/− 부호가 바뀌어도 동등하다고 보고 가능한 부호 조합 또는 충분한 무작위 조합을 만든다. "
          "표본이 작고 분포가 비정규적일 때 유용하다."),
        PageBreak(),

        P("부록 K. 다중검정과 계층검정", "h1"),
        P("Benjamini–Hochberg(BH)", "h2"),
        P("데이터셋별 검정을 여러 번 하면 아무 효과가 없어도 p<.05가 우연히 나올 수 있다. BH는 p-value를 정렬해 "
          "발견된 결과 중 거짓발견 비율(FDR)을 통제한다. 가족 전체가 단 하나라도 틀릴 확률을 통제하는 Bonferroni보다 보통 덜 보수적이다. "
          "최종 동일예산 unit 비교에서는 Sunwoda, RWTH, MICH, MATRb2가 BH q<.05였다."),
        P("Hierarchical bootstrap", "h2"),
        P("먼저 dataset을 다시 뽑고, 선택된 dataset 안에서 unit을 다시 뽑는다. HUST 7,775행과 Virkler 20행을 행 단위로 합치지 않고 "
          "dataset과 unit 두 층을 보존한다. Mixed retrospective 9-setting의 equal-dataset mean log-RMSE ratio는 .412, "
          "95% 계층구간 [.172,.665], geometric-mean RMSE 감소는 33.8% [15.8%,48.6%]였다."),
        P("Bound audit", "h2"),
        P("|ŷ−affine|≤margin×B를 17,645개 예측에서 직접 검사해 위반 0건이었다. 이는 코드가 선언한 수학적 제한을 지켰다는 "
          "검증이지, 예측이 정확하거나 시스템이 안전하다는 통계적 증명은 아니다."),
        PageBreak(),

        P("부록 L. 통계 결과를 한 장으로", "h1"),
        table(["근거 층", "결과", "말할 수 있는 것", "말하면 안 되는 것"], [
            ["Core paired units", "Direct 대비 17/25, Wilcoxon p=.0028", "core가 direct보다 unit 전반에서 개선", "모든 dataset에서 필연적 우월"],
            ["Constraint", "17,645점 bound 위반 0", "구현이 residual envelope 준수", "물리 안전 보장"],
            ["Mixed strongest retrospective", "9/9, exact sign p=.00390625", "기존 same-split 최강 비교에서 일관된 방향", "동일예산 완전 공정성"],
            ["Equal-budget retrospective", "8/9, p=.0391", "8 baseline×30 candidates 조건에서도 우세 방향", "PP-X 전체 개발비용도 30회"],
            ["77-unit hierarchy", "log ratio .412, CI [.172,.665]", "dataset·unit 계층에서 평균 RMSE 감소", "77개 독립 dataset"],
            ["Prospective DS03", "fallback .8818 < Engression .9013", "route-selection PASS", "prospective predictive superiority"],
        ], [34 * mm, 45 * mm, 49 * mm, 46 * mm]),
        P("왜 mixed 9/9와 equal-budget 8/9가 다른가", "h2"),
        P("9/9는 각 setting에서 저장된 strongest same-split comparator를 모은 이질적 비교다. 8/9는 모든 범용 비교군을 "
          "각각 30개 validation 후보와 5 refit seed로 맞춘 비교다. 비교 질문과 예산이 다르므로 합치거나 더 좋은 숫자만 골라 쓰면 안 된다."),
        PageBreak(),

        P("부록 M. 공정성 비교와 ‘이건 해봤냐’ 답변", "h1"),
        table(["질문", "답변"], [
            ["Plain MLP와 비교했나", "했다. 같은 정보의 direct NN 및 matched MLP를 core·fallback 대조군으로 사용했다."],
            ["Transformer와 비교했나", "FT-Transformer를 30 후보로 9개 setting에 비교했다. Virkler에서는 FT가 .890으로 PP-X .888보다 높았다."],
            ["OOD/robust 학습과 비교했나", "V-REx, GroupDRO, monotone NN을 포함했다."],
            ["확률·비모수 모델은", "Engression, full-train SVGP, linear-tail RBF를 포함했다."],
            ["TabPFN은", "보조 비교로 실행했지만 dataset별 train cap이 달라 최강 동일예산 순위에는 사용하지 않았다."],
            ["후보 수가 PP-X에만 많았나", "8개 범용 baseline은 각각 정확히 30 후보. PP-X는 동결된 이질적 contract route라 과거 전체 개발비용이 30이라는 뜻은 아니다."],
            ["총 계산량은", "동일예산 baseline benchmark 3,360 jobs, 기록 실행시간 합계 약 3.52시간."],
            ["Test-time adaptation을 했나", "하지 않았다. test label, test-batch 통계, transductive fitting, sample별 routing을 금지했다."],
        ], [59 * mm, 115 * mm]),
        PageBreak(),

        P("부록 N. 예상 꼬리질문 1 — 모델 정의·노벨티", "h1"),
        table(["질문", "방어 가능한 답"], [
            ["그냥 affine+NN 아닌가", "개별 블록의 최초성을 주장하지 않는다. novelty는 typed admissibility, frozen-prior 주변 residual authority, physical-unit validation approval, 사전 fallback을 하나의 실행 규칙으로 결합한 점이다."],
            ["PINN과 차이는", "완전한 지배방정식을 요구하지 않고 부분 prior의 사용 자격을 승인·거절한다. 식이 틀릴 때 fallback할 수 있다."],
            ["Mixture-of-Experts인가", "아니다. test sample마다 expert를 고르지 않는다. source/validation에서 route 하나를 고정한 뒤 frozen forward한다."],
            ["왜 prior를 학습 가능하게 안 하나", "밖의 tail은 데이터가 식별하지 못하므로 prior까지 자유롭게 움직이면 residual과 기준선의 역할이 섞인다. 다만 동결의 보편 성능 우월성은 미확증이다."],
            ["왜 abstention이 모델인가", "예측값 자체를 좋게 만드는 모듈은 아니지만, 허용되지 않은 가정을 실행하지 않는 것이 deployment policy의 일부다."],
            ["PP-X와 SAAR/CCMR 관계는", "PP-X가 paper-main framework. SAAR는 historical core alias, CCMR v2.2는 trajectory-domain executor evidence다."],
            ["PAE는 포함되나", "아니다. PAE는 명시적 식 카드 compiler라는 future program이며 이번 PP-X 주표와 분리한다."],
        ], [54 * mm, 120 * mm]),
        PageBreak(),

        P("부록 O. 예상 꼬리질문 2 — 데이터·누수·외삽", "h1"),
        table(["질문", "방어 가능한 답"], [
            ["같은 셀의 미래를 맞힌 것 아닌가", "먼저 physical unit을 분리했다. 시험 셀·시편·엔진은 train에 없다."],
            ["Random split보다 왜 어려운가", "새 unit이면서 train 건강범위 밖인 late-tail만 남겨 unit shift와 support extrapolation을 동시에 평가한다."],
            ["전체 feature hull 밖인가", "그렇게 주장하지 않는다. 사전 선언 1D health/crack/TRA 좌표 기준이다. MATRb2·N-CMAPSS는 PCA2 hull-out 0%도 공개했다."],
            ["미래정보를 feature에 썼나", "아니다. prefix/recent rate와 window 요약은 현재까지의 관측만 사용한다. EOL은 Y 생성에만 사용한다."],
            ["HUST 정답이 진짜인가", "완전 관측 EOL이 아니라 말단기울기 proxy RUL이다. 이 한계를 명시한다."],
            ["행이 수천 개인데 n도 수천인가", "성능 계산에는 행을 쓰지만 추론·bootstrap은 physical unit을 쓴다. 9개 main test unit 합계는 77개다."],
            ["N-CMAPSS는 실제 비행기인가", "아니다. NASA 고충실도 디지털 터보팬 시뮬레이션이다."],
        ], [58 * mm, 116 * mm]),
        PageBreak(),

        P("부록 P. 예상 꼬리질문 3 — 선택·통계·주장", "h1"),
        table(["질문", "방어 가능한 답"], [
            ["데이터셋별 oracle 아닌가", "Test 성능으로 route를 고르지 않고 typed contract와 group-disjoint validation으로 선택한다. 그러나 과거 9개는 retrospective development라 완전한 oracle 공격이 해소된 것은 아니다."],
            ["5 seeds면 통계적으로 충분한가", "Seed는 초기화 안정성만 본다. 과학적 표본은 unit/dataset이며 seed를 독립 실험으로 세지 않는다."],
            ["p=.0039면 미래에도 무조건 이기나", "아니다. 관측 9 settings에서 승리 방향이 우연인지 보는 sign test일 뿐 미래 성공확률이 아니다."],
            ["왜 R²만 쓰나", "주 지표는 pooled R²지만 RMSE, MAE, unit-macro R², per-unit RMSE, log-RMSE ratio를 함께 보고한다."],
            ["Negative R²는 무슨 뜻", "평균 Y를 항상 예측하는 기준보다 제곱오차가 크다는 뜻. 외삽 실패로 해석한다."],
            ["Prospective 성공했나", "DS03에서 fallback route 선택은 성공했지만 Engression .9013보다 PP-X .8818이 낮아 predictive superiority는 실패했다."],
            ["안전하다고 할 수 있나", "아니다. constraint 준수와 fallback 정책을 보였을 뿐 실제 시스템 safety certification은 아니다."],
        ], [58 * mm, 116 * mm]),
        PageBreak(),

        P("부록 Q. 실패·기각 실험도 왜 중요한가", "h1"),
        table(["실험/반례", "관측", "최종 결정"], [
            ["MICH fixed bound", "unbounded .759 > bounded .468", "fixed bound를 universal default에서 제외; dual-scale 개발"],
            ["RWTH dual scale", "fixed route보다 악화", "dual-scale을 MICH형 heterogeneity에만 조건부 사용"],
            ["MICH full rate history", "simple margin history .715 > full .468", "validation gate가 관계이동을 항상 탐지한다는 주장 금지"],
            ["Virkler equal budget", "FT .890 > PP-X .888", "8/9로 정직하게 보고"],
            ["DS03 prospective", "route PASS, Engression보다 낮음", "predictive superiority FAIL"],
            ["CRT", "DS03 .8851이나 Engression 미달", "Algorithm 1에서 기각"],
            ["GCIE", ".8850, nested .8727; 기준 동시 미달", "Algorithm 1에서 기각"],
            ["XJTU/FEMTO/milling", "초기 실패와 post-test 복구/불안정", "main superiority 승수에서 제외"],
        ], [40 * mm, 67 * mm, 67 * mm]),
        P("음성 결과의 역할", "h2"),
        P("실패를 숨기지 않으면 어떤 가정이 어디서 깨지는지 알 수 있다. PP-X의 조건부 실행 논리는 성공 데이터만이 아니라 "
          "‘모든 모듈을 항상 켜면 악화된다’는 반례에서 나온다. 실패 후 test를 보며 만든 복구는 개발 가설이지 소급 확증이 아니다."),
        PageBreak(),

        P("부록 R. 발표자가 반드시 지켜야 할 주장 경계", "h1"),
        table(["말해도 됨", "말하면 안 됨"], [
            ["Retrospective 9-setting에서 mixed 9/9, equal-budget 8/9의 우세 방향", "모든 외삽·모든 미래 데이터에서 SOTA"],
            ["77 physical units 계층 분석에서 평균 RMSE 감소 근거", "77개의 독립 데이터셋에서 검증"],
            ["DS03에서 미리 정한 fallback route 선택 성공", "DS03에서 최강 예측모델보다 우월"],
            ["Bound 위반 0건과 비음수 decoder", "실제 설비 안전 인증 완료"],
            ["Contract와 validation이 optional executor를 제한", "Validation이 미래 최적 route를 완벽히 알아냄"],
            ["MICH·RWTH 등 반례를 포함한 조건부 모듈 효용", "모든 PP-X 부품이 모든 데이터에서 성능 향상"],
            ["PAE는 future program", "PAE 결과가 PP-X paper-main 성능"],
        ], [87 * mm, 87 * mm]),
        callout("<b>최종 방어 문장:</b> ‘PP-X는 보편적 정확도를 주장하지 않습니다. 외삽을 구조 가정의 조건부 실행 문제로 "
                "정식화했고, retrospective 다중도메인 근거와 prospective route-selection 근거를 확보했지만, "
                "독립 prospective predictive superiority는 아직 확증되지 않았습니다.’"),
        PageBreak(),

        P("부록 S. 사용 절차 — 새 데이터에 PP-X를 적용하려면", "h1"),
        table(["단계", "실무 질문", "통과 산출물"], [
            ["1 문제 정의", "무엇이 unit이고 failure/EOL은 무엇인가", "unit ID·Y 정의·시간축"],
            ["2 Causal audit", "예측 시점에 실제로 알 수 있는 X인가", "금지 미래정보 목록"],
            ["3 Geometry", "어느 좌표 밖을 extrapolation이라 부를 것인가", "사전 좌표와 hull audit"],
            ["4 Unit split", "같은 제품이 split에 겹치지 않는가", "train/val/test unit manifest"],
            ["5 Contract", "boundary·ordered progression·regime·prior가 정당한가", "typed admissibility card"],
            ["6 Candidate fit", "prior-only/direct/core/optional을 동일 source로 학습했나", "row-aligned validation predictions"],
            ["7 Approval", "≥2% gain, unit evidence, heterogeneity 조건을 통과했나", "frozen selection artifact"],
            ["8 Freeze", "hash·seed·budget·success criterion을 test 전에 고정했나", "protocol commit"],
            ["9 Test", "개별 sample causal X 외 test batch 정보를 안 썼나", "frozen y_pred"],
            ["10 Report", "실패·abstention·per-unit 결과도 전부 보고했나", "R²/RMSE/MAE/unit 결과와 한계"],
        ], [25 * mm, 91 * mm, 58 * mm]),
        P("PP-X를 쓰면 안 되는 경우", "h2"),
        P("Unit이 하나뿐이라 독립 검증이 불가능한 경우, failure boundary가 test 결과로 정해진 경우, feature가 전체 미래 궤적을 요구하는 경우, "
          "validation과 test의 메커니즘이 반대로 이동하는 경우, prior 후보가 source OOF에서 direct보다 해로운 경우에는 "
          "숫자를 억지로 내기보다 fallback 또는 abstention해야 한다."),
        PageBreak(),

        P("부록 T. 30초·2분·5분 설명 템플릿", "h1"),
        P("30초", "h2"),
        P("‘PP-X는 학습범위 밖 RUL을 예측할 때 구조 가정을 무조건 강제하지 않고, 데이터셋의 고장경계와 인과 특징을 contract로 선언한 뒤 "
          "validation이 지지하는 prior-residual executor만 시험 전에 고정합니다. 9개 retrospective setting에서 동일예산 비교 8/9 우세였지만, "
          "DS03 prospective에서는 route 선택만 성공하고 Engression보다 낮았으므로 보편적 SOTA는 주장하지 않습니다.’"),
        P("2분에 반드시 포함할 것", "h2"),
        table(["순서", "내용"], [
            ["문제", "Support 밖에서는 여러 함수가 train에 똑같이 맞아도 tail이 달라짐"],
            ["방법", "Declare→Learn frozen-prior residual→Approve→Decline"],
            ["데이터", "새 unit + 사전 1D hull-out + val-only 선택"],
            ["근거", "core ablation, conditional executor 반례, 77-unit 계층통계"],
            ["공정성", "8 baselines×30 candidates×5 refit, PP-X 8/9"],
            ["한계", "DS03 .8818 < Engression .9013; 독립 predictive superiority 미확증"],
        ], [30 * mm, 144 * mm]),
        P("5분 추가", "h2"),
        P("Sun/RWTH의 fixed bound 성공과 MICH의 −.291 반례, MATRb2 transport 주효과, Wilcoxon·unit bootstrap·BH의 추론단위, "
          "그리고 CRT/GCIE 기각까지 설명하면 설계가 결과를 보고 임의로 붙인 모듈 모음이 아니라 실패를 통해 조건부 실행 규칙으로 좁혀졌음을 보여줄 수 있다."),
    ]


def append_architecture_evidence(story):
    story += [
        PageBreak(), P("부록 U. PP-X 전체 모델 구조", "h1"),
        P("PP-X는 ‘한 개의 고정 신경망’ 이름이 아니라 <b>바깥의 선택 framework</b>와 "
          "<b>안쪽의 prior-residual executor</b>를 합친 알고리즘이다. 바깥층은 어떤 prior와 optional module을 "
          "쓸지 test 전에 결정하고, 안쪽층은 선택된 구조로 RUL을 계산한다."),
        Image(str(ARCH_FIG), width=174 * mm, height=80 * mm),
        table(["층", "입력→출력", "학습/동결"], [
            ["Contract layer", "boundary·unit·ordered coordinate·regime→허용 후보 집합", "사람/프로토콜이 test 전에 선언"],
            ["Causal adapter", "원시 과거 관측→현재값·평균·rate·history·context", "미래값 없이 결정적 변환"],
            ["Prior path", "z→affine tail 또는 boundary quotient y_prior", "train Ridge/구조로 적합 후 frozen"],
            ["Residual path", "d차원 z→폭 w tanh hidden layers→scalar r(z)", "train에서 AdamW로 학습"],
            ["Authority envelope", "r→b(z)tanh(r/b)", "fixed/dual-scale/decay를 validation 승인"],
            ["Transport", "raw prediction→source OOF가 지지한 scale/offset 보정", "group-LOO validation에서만 선택"],
            ["Decoder", "latent output→비음수·경계 일치 RUL", "domain contract로 고정"],
            ["Selector", "validation evidence→executor 또는 fallback", "test outcome 인자 자체가 없음"],
        ], [34 * mm, 89 * mm, 51 * mm]),
        PageBreak(),

        P("부록 V. 신경망 내부와 학습 설정", "h1"),
        P("Legacy base PP의 표준 residual branch는 <b>Linear(d,32)–tanh–Linear(32,32)–tanh–Linear(32,1)</b>이고, "
          "affine branch는 Linear(d,1)이다. Paper portfolio의 boundary matched ablation은 동일 비교를 위해 width 64를 사용했다. "
          "즉 폭 32/64는 실험 adapter의 구현 설정이고 PP-X의 본질은 특정 폭이 아니라 frozen prior와 제한 residual의 권한 분리다."),
        table(["학습 요소", "설정/의미"], [
            ["Affine 초기화", "unit-balanced weighted Ridge; alpha {0.1,1,10,100,1000,10000} 중 validation 선택"],
            ["Residual optimizer", "AdamW; base frozen protocol lr 5e−4, nonlinear weight decay 2.0"],
            ["Batch/epoch", "batch 512, 최대 300 epoch, patience 70"],
            ["안정화", "gradient clipping 2.0, seeds 42–46"],
            ["정규화", "X 평균·표준편차와 Y scale을 train에서만 계산"],
            ["손실 가중치", "각 physical unit의 총 loss weight를 같게 설정"],
            ["출력", "비음수 및 train-derived cap; boundary executor는 failure margin에서 0으로 수렴"],
            ["선택", "alpha·epoch·executor는 validation; test는 frozen forward 한 번"],
        ], [47 * mm, 127 * mm]),
        P("왜 tanh인가", "h2"),
        P("tanh는 출력이 −1과 1 사이로 포화되므로 residual을 유한 envelope 안에 넣기 쉽다. ReLU처럼 support 밖에서 "
          "선형으로 계속 커지는 correction보다 authority를 명시적으로 제한할 수 있다. 다만 tanh 자체가 정확도를 보장하는 것은 아니다."),
        P("왜 unit-balanced loss인가", "h2"),
        P("수명이 긴 셀은 행이 많다. 행 평균 loss만 쓰면 긴 셀이 학습을 지배한다. Unit별 총 가중치를 같게 하면 "
          "‘제품 수’에 가까운 균형을 유지한다."),
        PageBreak(),

        P("부록 W. 데이터셋별 실제 활성 구조", "h1"),
        table(["셋", "Prior/core", "Bound·support", "History", "Transport", "최종 관측"], [
            ["HUST", "affine prior+residual", "bounded/기본 안정화", "capacity+장·단기 rate", "ON", ".829→.958 (+.128)"],
            ["Virkler", "affine residual", "support gate ON", "crack prefix/recent rate", "OFF", "PP-X .888; FT .890"],
            ["NASA batt.", "health-φ affine residual", "causal multiscale", "validation-selected", "OFF", ".572→.584 (+.012)"],
            ["Sunwoda", "boundary quotient+residual", "fixed bound ON", "full rate history", "OFF", ".718→.939 (+.221)"],
            ["RWTH", "boundary quotient+residual", "fixed bound ON", "full rate history", "OFF", ".788→.878 (+.090)"],
            ["MICH", "boundary quotient+residual", "dual-scale ON", "simple/limited route", "OFF", ".468→.751 (+.283)"],
            ["MATR19", "latent affine residual", "calibration route", "8-step Q/rate", "output calibration", ".257→.466"],
            ["MATRb2", "affine residual", "decay ON", "8-step Q/rate", "ON", ".674→.862 (+.189)"],
            ["N-CMAPSS", "causal latent residual", "multiscale route", "basic/moments/multiscale 선택", "OFF", ".928→.937 (+.009)"],
        ], [23 * mm, 36 * mm, 34 * mm, 38 * mm, 24 * mm, 31 * mm], size=6.6),
        P("ON의 뜻", "h2"),
        P("해당 모듈을 모든 데이터에서 무조건 켰다는 뜻이 아니라, 그 setting의 contract가 허용했고 validation evidence가 선택한 "
          "최종 paper route에서 활성화됐다는 뜻이다. 일부 route는 test를 본 뒤 개발된 retrospective 결과이므로 새 cohort 확증과 구분한다."),
        PageBreak(),

        P("부록 X. 활성 모듈이 도움된 개념적 이유와 직접 실험", "h1"),
        table(["개념적 문제", "모듈이 하는 일", "직접 on/off 근거", "결론"], [
            ["같은 health라도 열화속도가 다름", "causal rate history로 상태를 구별", "Sun current .590→full .939; RWTH −.378→.878", "Sun/RWTH에서 강한 도움"],
            ["NN이 tail에서 과수정", "fixed tanh bound로 correction 제한", "Sun .718→.939; RWTH .788→.878", "경계 cohort에서 도움"],
            ["필요 correction이 고정 bound보다 큼", "support heterogeneity에 따라 broad bound 허용", "MICH .468→.751", "MICH형 shift 복구"],
            ["Cohort마다 출력 scale/offset 이동", "group-LOO에서 반복된 오류만 transport", "HUST .829→.958; MATRb2 .675→.862", "가장 큰 executor 효과 중 하나"],
            ["먼 support에서 transport 과보정", "residual decay로 correction 감쇠", "MATRb2 transport-only .820→둘 다 .862", "상호작용 도움"],
            ["유효 시간척도가 다름", "short/basic/multiscale 중 validation 선택", "NASA +.012; N-CMAPSS +.009", "방향은 양수, 효과 작음"],
            ["Prior-only가 곡률을 못 잡음", "nonlinear residual 추가", "Sun +.658, RWTH +.219, MICH +3.811", "공통 core의 가장 강한 근거"],
        ], [41 * mm, 50 * mm, 49 * mm, 34 * mm], size=6.9),
        callout("<b>추가 실험이 필요하지 않았던 이유:</b> 요청한 ‘무엇이 켜져서 왜 도움됐나’는 이미 동일 입력·split·seed를 유지한 "
                "matched on/off, 6-arm, history ablation, MATR-b2 2×2 실험으로 직접 측정돼 있다. "
                "새 test-aware 튜닝을 하면 오히려 증거 등급이 낮아지므로 기존 frozen 결과를 사용했다."),
        PageBreak(),

        P("부록 Y. 반례까지 포함한 activation 규칙", "h1"),
        table(["모듈", "켜야 할 신호", "끄거나 거절할 신호", "실제 반례"], [
            ["Fixed bound", "독립 failure boundary + validation에서 unbounded 과보정", "큰 relationship shift로 넓은 correction 필요", "MICH −.291"],
            ["Dual scale", "train support heterogeneity≥.50 + fixed보다 ≥2% 개선", "균질 tail 또는 fixed가 이미 안정", "RWTH에서 악화"],
            ["Full rate history", "같은 health에서 rate가 RUL을 분리하고 val 개선", "새 cohort에서 health–RUL 관계 방향 이동", "MICH simple history가 +.247"],
            ["Transport", "complete source groups와 group-LOO scale/offset 반복", "OOF regret>0, regime coverage 부족", "근거 없으면 identity"],
            ["Support decay", "거리가 멀수록 residual/transport 불안정", "거리와 오차 관계가 validation에 없음", "MATRb2 단독 +.002"],
            ["Multiscale", "시간척도별 validation ranking이 안정", "unit 수가 작아 선택 불안정", "NASA/N-CMAPSS 효과 작음"],
            ["Prior route", "boundary 또는 source OOF prior regret≤0", "complete groups 부족·prior regret 양수", "FEMTO prior 거절"],
        ], [31 * mm, 57 * mm, 57 * mm, 29 * mm], size=7),
        P("활성화 표를 인과 증명으로 읽으면 안 되는 이유", "h2"),
        P("Ablation은 해당 split에서 모듈 제거와 성능 변화의 연관을 matched하게 보여준다. 그러나 ‘이 물리 메커니즘 때문에 "
          "반드시 +.221이 발생했다’는 인과적 자연법칙 증명은 아니다. 특히 retrospective 개발에서 선택된 route는 독립 cohort 재현이 필요하다."),
        PageBreak(),

        P("부록 Z. 구조 관련 예상 질문", "h1"),
        table(["질문", "답변"], [
            ["최종 network의 정확한 layer 수는", "Base residual은 2개 tanh hidden layer와 scalar output. Boundary matched 실험은 width 64, base frozen protocol은 width 32다."],
            ["데이터마다 layer가 다른가", "Top-level PP-X는 adapter/executor 후보가 다르다. 공통 철학은 prior+residual authority이며 12 route가 하나의 동일 network는 아니다."],
            ["Dual-scale은 network 두 개인가", "아니다. 같은 residual 신호를 local bound BL=2와 broad bound BH=6으로 포화시킨 두 envelope를 support gate로 혼합한다."],
            ["Gate가 test sample마다 모델을 바꾸나", "Executor 선택 gate는 validation에서 한 번 동결한다. Support gate는 선택된 함수 안에서 train-derived 거리로 residual 크기를 연속 조절할 뿐 test label routing이 아니다."],
            ["Transport가 test 통계를 보나", "아니다. Source group-LOO에서 얻은 오류 구조만 운반하며 test-batch 평균·정답을 사용하지 않는다."],
            ["Affine prior가 물리식인가", "항상 그렇지 않다. 데이터 기반 affine tail일 수 있고, 독립 boundary가 있으면 quotient prior가 admissible하다."],
            ["모델 파라미터 수 비교는 공정한가", "동일예산 benchmark는 후보 수를 맞췄다. 파라미터 수가 완전히 동일한 비교만 있는 것은 아니므로 matched direct/core ablation과 강한 baseline benchmark를 분리해 보고한다."],
            ["왜 더 큰 Transformer 하나로 안 하나", "Support 밖 거동은 용량보다 가정이 문제다. FT도 비교했고 Virkler에서는 이겼지만 모든 setting의 구조 경계·fallback을 제공하지 않는다."],
        ], [59 * mm, 115 * mm]),
    ]


def append_novelty_argument(story):
    story += [
        PageBreak(), P("부록 AA. 논문의 노벨티 — 무엇을 새롭다고 주장하는가", "h1"),
        P("이 논문의 기여는 affine, tanh, residual, validation, fallback 각각을 최초로 발명했다는 것이 아니다. "
          "기존 요소를 단순히 나열하는 것도 아니다. PPT·제출 서사와 동일하게, PP-X는 "
          "<b>universal neural predictor가 아니라 contract-conditioned, validation-approved "
          "prior-residual execution architecture</b>다."),
        P("PPT 고정 기여 3항", "h2"),
        table(["기여", "내용", "직접 근거"], [
            ["C1 Contract-conditioned model", "허용 prior 선언 + frozen prior 주변 bounded residual", "core ablation · typed contract"],
            ["C2 Validation-approved execution", "executor 조건부 실행 · 거절/fallback", "MICH/RWTH 반례 · DS03 route PASS"],
            ["C3 Equal-coverage evidence", "unit 통계 · domain stability · Selective equal-coverage", "8/9 · 33.8% · selective 5/5"],
        ], [42 * mm, 72 * mm, 60 * mm], size=7.2),
        P("세부 분해(참고용). 아래 D1–D6는 위 3항을 더 잘게 풀어 쓴 것이며, 발표 기여 번호와 혼동하지 않는다.", "h2"),
        table(["세부", "새로운 통합", "직접 근거", "주장 한계"], [
            ["D1 문제 정식화", "Extrapolation=함수 fitting이 아니라 structural assumption approval", "모듈별 상반된 성공·실패", "모든 OOD 문제의 유일한 정의 아님"],
            ["D2 Typed admissibility", "boundary·state·unit·regime에 따라 평가할 후보 자체를 outcome-free 제한", "12-domain RMSE-only false accept 4", "Contract 완전성이 자동 보장되지는 않음"],
            ["D3 Residual authority", "Frozen prior를 중심으로 NN 수정 권한을 유한·조건부 부여", "Residual 강한 ΔR²; 17,645 bound 위반 0", "동결/고정 bound의 보편 우월은 미확증"],
            ["D4 Conditional executor", "bound·dual-scale·history·transport를 전역 default가 아닌 val-approved 옵션화", "MICH·RWTH 반례와 executor ablation", "Validation이 미래 route를 완벽히 식별하지 못함"],
            ["D5 Unit-risk decision", "행 loss가 아니라 physical-unit evidence로 승인·fallback", "Wilcoxon·bootstrap·sign-flip·BH", "작은 domain 수와 독립성 한계"],
            ["D6 Honest deployment rule", "승인 실패를 exact fallback/abstention으로 알고리즘에 포함", "DS03 fallback route PASS", "Engression보다 낮아 superiority FAIL"],
        ], [31 * mm, 60 * mm, 50 * mm, 33 * mm], size=6.8),
                PageBreak(),

        P("부록 AB. 하나의 귀납적 주장으로 묶는 논리 사슬", "h1"),
        table(["단계", "관찰/증거", "도출되는 결론", "도출하면 안 되는 결론"], [
            ["1 식별성", "Train 안에서 비슷한 함수도 support 밖 tail이 다름", "외삽에는 추가 가정이 필요", "특정 prior가 언제나 맞음"],
            ["2 반례", "Fixed bound는 Sun/RWTH↑, MICH↓; dual-scale은 MICH↑, RWTH↓", "가정과 executor 효용은 regime-conditional", "dataset 이름별 oracle 허용"],
            ["3 후보 제한", "RMSE-only route는 false accept 4/12", "점수 전에 typed contract로 admissible 후보 제한 필요", "Contract만 있으면 미래 성공 보장"],
            ["4 권한 분리", "Affine-only·direct-only 모두 실패; residual은 세 배터리에서 큰 이득", "Prior와 residual을 결합하되 역할을 분리", "모든 bound가 유의"],
            ["5 승인 규칙", "Optional module always-on은 악화 사례 존재", "Group-disjoint validation과 unit-risk로 실행권 승인", "Validation oracle"],
            ["6 다중 setting", "Retrospective mixed 9/9; equal-budget 8/9", "조건부 framework가 여러 설정에서 일관된 가능성", "미래 일반화 확률 100%"],
            ["7 Prospective 반증", "DS03 route PASS, predictive superiority FAIL", "선택 절차 일부는 미래 자료에서 작동; 정확도 우월은 미확증", "보편 SOTA"],
        ], [20 * mm, 58 * mm, 58 * mm, 38 * mm], size=6.8),
        callout("<b>귀납적 최종 결론:</b> 서로 다른 도메인에서 optional structural prior의 효용이 일관되지 않았고, "
                "matched ablation은 prior-residual 역할 분리와 조건부 executor의 필요성을 반복해 보여줬다. "
                "따라서 외삽 모델은 하나의 강한 prior를 항상 실행하기보다, outcome-free contract와 physical-unit validation이 "
                "허용한 구조만 실행하고 나머지는 fallback해야 한다. 현재 증거는 이 설계 원칙을 지지하지만 보편적 정확도 우월을 확증하지 않는다."),
        PageBreak(),

        P("부록 AC. Claim–Evidence–Warrant 구조", "h1"),
        P("논리 오류를 피하려면 결과와 주장 사이에 <b>warrant(왜 그 증거가 그 결론을 지지하는가)</b>를 명시해야 한다."),
        table(["Claim", "Evidence", "Warrant", "Qualifier"], [
            ["Residual이 core에 필요", "Affine-only 대비 Sun +.658, RWTH +.219, MICH +3.811", "다른 조건을 맞춘 제거 arm에서 residual 추가만으로 개선", "세 boundary battery settings에서"],
            ["Fixed bound는 조건부", "Sun +.221, RWTH +.090, MICH −.291", "같은 조작의 효과 부호가 setting별 반전", "보편 default 아님"],
            ["Executor selection이 필요", "Always-on 악화; gated 3 개선·10 유지·0 악화", "거절이 base route 손실을 줄임", "Retrospective audit"],
            ["PP-X 우세 방향", "Equal-budget 8/9, sign p=.0391", "동일 후보예산의 setting-level 승패가 우연 .5보다 한쪽", "관측 9 settings에서"],
            ["Unit 수준 효과 존재", "계층 log ratio CI [.172,.665]", "행 수가 아니라 dataset/unit 재표집에서도 평균 방향 유지", "Retrospective 모집단 가정 아래"],
            ["미래 route 선택 가능성", "DS03가 fallback을 사전 선택하고 PP-X 대안보다 높음", "Test 전에 고른 route가 내부 후보 중 최선", "한 N-CMAPSS 계열 cohort"],
            ["미래 최강 정확도 우월", "PP-X .8818 < Engression .9013", "성공 기준을 충족하지 못함", "주장하지 않음"],
        ], [37 * mm, 50 * mm, 55 * mm, 32 * mm], size=6.7),
        P("대표 논리 오류와 차단", "h2"),
        table(["오류", "잘못된 문장", "수정"], [
            ["성급한 일반화", "9개에서 좋아 모든 데이터에 좋다", "Retrospective 9 settings에서 일관된 방향"],
            ["사후선택", "Test 최고 route가 PP-X다", "Contract와 validation으로 test 전에 고른 route만 PP-X 결과"],
            ["구성의 오류", "각 모듈이 일부에서 좋으니 모두 함께 좋다", "상호작용·반례를 2×2/on-off로 확인하고 조건부 사용"],
            ["p-value 오해", "p<.05이므로 효과가 크고 미래에도 참", "H0 아래 관측 이상 통계량의 확률; 효과크기·CI·설계와 함께 해석"],
            ["비유의=동일", "p>.05이므로 두 방법은 같다", "현재 표본으로 차이를 확증하지 못함"],
        ], [34 * mm, 68 * mm, 72 * mm], size=7),
        PageBreak(),

        P("부록 AD. 논문 Introduction→Conclusion 서사", "h1"),
        table(["논문 위치", "논리적 역할", "권장 핵심 문장"], [
            ["Problem", "Support 밖 비식별성 제시", "Data fit alone cannot determine extrapolative behavior."],
            ["Gap", "기존 방법이 특정 가정을 실행하지만 사용 자격·거절 절차는 분리됨", "The missing layer is test-independent approval of structural assumptions."],
            ["Method", "Declare–Learn–Approve–Decline 제안", "PP-X limits residual authority around an admissible prior and freezes one validation-approved route."],
            ["Ablation", "왜 각 설계가 필요한지 반례와 함께 보임", "Opposite module effects motivate conditional rather than universal execution."],
            ["Benchmark", "범용 비교군 대비 breadth 확인", "PP-X wins 8/9 under a uniform 30-candidate retrospective budget."],
            ["Statistics", "행 착시·다중검정 방어", "Inference is performed at physical-unit and dataset levels, not over correlated rows."],
            ["Prospective", "방법을 반증 가능한 상태로 시험", "DS03 confirms the fallback decision but falsifies predictive superiority."],
            ["Conclusion", "강한 주장과 미확증을 동시에 고정", "Evidence supports conditional execution, not universal accuracy."],
        ], [30 * mm, 62 * mm, 82 * mm], size=7),
        P("논문 기여문 예시", "h2"),
        P("“We contribute (i) a typed admissibility contract that restricts extrapolative priors before outcomes are observed, "
          "(ii) a prior-centered residual architecture with explicit correction authority, and (iii) a physical-unit validation rule "
          "that freezes an approved executor or reverts to a prespecified fallback. Matched ablations across heterogeneous RUL settings "
          "show that executor effects reverse across regimes, motivating conditional execution. Retrospective equal-budget evidence "
          "supports broad utility, while a prospective DS03 replay validates route selection but not predictive superiority.”"),
        PageBreak(),

        P("부록 AE. 노벨티 공격 예상과 답변", "h1"),
        table(["공격", "답변"], [
            ["기존 요소 조합일 뿐이다", "맞다, 개별 블록 최초성은 주장하지 않는다. 기여는 구조 가정의 admissibility·residual authority·unit-risk approval·decline을 하나의 frozen executable policy로 정식화한 통합이다."],
            ["Validation model selection과 뭐가 다른가", "일반 선택은 모든 후보를 점수화한다. PP-X는 outcome-free contract로 부적합 prior를 먼저 제외하고, physical-unit 위험과 fallback까지 실행 의미론에 포함한다."],
            ["Dataset별로 잘된 모델을 고른 oracle다", "Retrospective 개발 위험은 인정한다. 그래서 selector signature에 test outcome이 없고 frozen protocol과 DS03 prospective 실패까지 공개한다."],
            ["Bounded residual은 흔하다", "Bound 자체 최초성이 아니라 correction authority의 명시적 역할과 조건부 승인, 위반 audit가 기여다. MICH 반례로 universal bound 주장을 피한다."],
            ["성능이 novelty를 증명하나", "성능만으로 novelty를 증명하지 않는다. Ablation은 설계 necessity를, 비교는 utility를, protocol은 test-independent executability를 각각 지지한다."],
            ["Prospective에서 졌는데 논문이 성립하나", "보편 SOTA 논문으로는 약하다. 그러나 structural-assumption approval framework와 정직한 falsification 결과는 성립하며, predictive superiority는 후속 확증 과제다."],
        ], [55 * mm, 119 * mm]),
        callout("<b>가장 안전하고 강한 포지셔닝:</b> ‘새 신경망 블록’ 논문이 아니라 "
                "‘외삽 구조 가정의 조건부 실행과 거절을 재현 가능한 알고리즘으로 만든 방법론’ 논문이다."),
    ]


def append_latest_and_tabpfn(story):
    story += [
        PageBreak(), P("부록 AF. 이 문서가 사용하는 최신 PP-X 논문 타깃", "h1"),
        callout("<b>확인 결과:</b> 이 PDF는 저장소의 현재 canonical paper model인 <b>PP-X</b>를 기준으로 한다. "
                "정식 최신 설명은 ‘contract-conditioned prior-residual architecture with transferability-aware abstention’이다."),
        table(["항목", "현재 논문 위치"], [
            ["Paper main", "PP-X — 상위의 학습·실행·승인 framework"],
            ["Core", "frozen extrapolative prior + source-supported nonlinear residual"],
            ["Selection", "typed contract + validation/source OOF evidence"],
            ["Decline", "direct/neural 또는 persistence fallback, 필요 시 abstention"],
            ["CCMR v2.2", "trajectory-domain risk-certified executor이자 구조·안전성 mechanism evidence"],
            ["CCMR v2.3", "개발 geometric-mean RMSE를 엄격히 개선하지 못해 rejected"],
            ["CRT / GCIE", "DS03에서 기준을 충족하지 못해 rejected; Algorithm 1 미포함"],
            ["SAAR", "historical core/architecture alias; PP-X 전체와 동의어 아님"],
            ["PAE", "명시적 equation-card compiler라는 future program; paper main 아님"],
        ], [43 * mm, 131 * mm]),
        P("논문 타깃 설명", "h2"),
        P("목표는 모든 tabular/OOD 데이터에서 정확도 1등인 범용 모델을 주장하는 것이 아니다. "
          "연속 열화·RUL의 strict-tail inductive extrapolation에서, <b>어떤 구조 가정을 실행해도 되는지</b>를 "
          "test-independent하게 선언·검증하고, 근거가 없으면 후퇴시키는 방법론이 논문 타깃이다. "
          "Retrospective 9-setting breadth, equal-budget 8/9, unit-level 통계, CCMR v2.2의 risk-aware mechanism, "
          "DS03 prospective route-selection PASS와 predictive-superiority FAIL을 함께 보고하는 것이 현재 주장 경계다."),
        P("버전 표기 주의", "h2"),
        P("‘PP-X v2.2’라고 부르면 틀린다. v2.2는 CCMR executor 버전이다. PP-X paper method는 frozen method v1이며, "
          "상위 모델 정체성은 PP-X다."),
        PageBreak(),

        P("부록 AG. TabPFN은 무엇인가", "h1"),
        P("<b>TabPFN(Tabular Prior-data Fitted Network)</b>은 많은 합성 tabular task에서 미리 학습된 Transformer 계열 "
          "foundation model이다. 새 데이터셋마다 일반 신경망처럼 수백 epoch로 weight를 처음부터 최적화하기보다, "
          "train 표의 X와 y를 문맥(context)으로 넣고 test X의 y를 in-context inference한다. "
          "사전학습 prior가 다양한 표형 데이터 생성과정을 압축해 둔 셈이다."),
        Image(str(TABPFN_FIG), width=174 * mm, height=70 * mm),
        table(["개념", "설명"], [
            ["Meta/pretraining", "많은 합성 데이터셋과 예측 task에서 ‘표를 보고 규칙을 추론하는 법’을 사전학습"],
            ["In-context learning", "현재 train rows와 labels가 prompt/context 역할을 하고 test row를 한 forward 과정에서 예측"],
            ["Amortized inference", "매 task마다 처음부터 Bayesian inference/대규모 tuning을 반복하는 대신 사전학습 계산을 재사용"],
            ["Tabular foundation model", "특정 배터리식이 아니라 일반 수치·범주형 tabular 관계를 위한 강한 범용 prior"],
            ["Regression", "Train X,y와 test X의 관계를 attention으로 읽어 연속 y 분포/예측을 출력"],
            ["Ensemble", "전처리·feature/class 순서·checkpoint/추론 구성 등을 바꾼 여러 estimator를 합칠 수 있음"],
        ], [45 * mm, 129 * mm]),
        PageBreak(),

        P("부록 AH. TabPFN v3 내부 구조", "h1"),
        P("TabPFN v3 technical report 기준 핵심은 feature 수와 row 수를 다루기 위한 <b>3단 압축·ICL 구조</b>다."),
        table(["단계", "작동", "왜 필요한가"], [
            ["0 Feature grouping", "각 feature를 cyclic neighbor 두 개와 묶어 triplet을 만들고 linear cell embedding", "개별 열과 주변 열 조합을 일정 hidden dimension으로 변환"],
            ["Target-aware embedding", "Train row에는 y 정보를 embedding으로 더하고 test row에는 정답을 주지 않음", "어떤 X가 어떤 y와 연결됐는지 context에 표시"],
            ["1 Column distribution embedder", "Inducing-point attention으로 각 열의 값 분포를 요약", "모든 row×row attention 비용을 줄이며 열 분포 파악"],
            ["2 Row-wise feature aggregation", "한 row의 feature embeddings와 learned CLS tokens가 non-causal attention", "가변 feature 수를 고정 차원 row embedding으로 압축"],
            ["3 ICL Transformer", "Train row는 서로 attend하고 test row는 train context를 attend", "새 표의 함수관계를 in-context로 추론"],
            ["Regression decoder", "Test row representation을 연속 target 예측으로 변환", "최종 ŷ 생성"],
            ["QASSMax", "입력 길이에 따라 query scale을 조절하는 attention softmax", "긴 context 길이로의 일반화 안정화"],
            ["Caching/chunking", "Train context 표현을 재사용하고 test를 chunk로 예측 가능", "큰 표에서 메모리·반복 계산 절감"],
        ], [40 * mm, 83 * mm, 51 * mm], size=7),
        P("v2 계열과 차이", "h2"),
        P("TabPFN v2.x는 row-wise와 feature-wise attention을 교대로 사용했다. v3는 column distribution embedding→row aggregation으로 "
          "한 row를 먼저 압축한 뒤, row sequence에 TabPFN v1형 ICL Transformer를 적용한다. 공식 v3는 최대 1M rows×200 features "
          "또는 100K rows×2,000 features 범위를 목표로 하지만, 실제 가능 크기는 행·열·하드웨어·추론 설정의 trade-off를 따른다."),
        PageBreak(),

        P("부록 AI. 우리 실험에서 TabPFN을 어떻게 사용했나", "h1"),
        table(["설정", "우리 실행"], [
            ["모델", "다운로드된 로컬 TabPFN v3 regressor checkpoint (`tabpfn-v3-regressor-v3_default.ckpt`)"],
            ["API", "`TabPFNRegressor.create_default_for_version(ModelVersion('v3'))`"],
            ["장치", "CPU"],
            ["Seeds", "42–46"],
            ["Estimator", "seed당 n_estimators=1; 다섯 예측을 별도 보고하고 평균 ensemble 계산"],
            ["선택", "고정 v3 구성; test label로 모델·hyperparameter 선택하지 않음"],
            ["출력", "비음수/해당 실행에서는 train-derived cap으로 clip한 경우가 있음"],
            ["Train cap A", "HUST·Virkler·NASA·C-MAPSS 최대 3,000 equal-unit rows"],
            ["Train cap B", "Sunwoda·RWTH·MATRb2 최대 1,000 equal-unit evenly-spaced rows"],
            ["Test", "PP-X와 동일 frozen test rows와 pooled R²"],
        ], [45 * mm, 129 * mm]),
        callout("<b>공정성 한계:</b> 이는 동일 정보·동일 test의 강한 보조 비교지만 full-data TabPFN 최적 성능은 아니다. "
                "특히 1,000/3,000행 cap과 estimator 1개 때문에 TabPFN의 v3 최대 규모·대규모 ensemble 능력을 사용하지 않았다. "
                "따라서 최종 equal-budget 8-baseline 순위와 TabPFN 보조표를 구분한다."),
        PageBreak(),

        P("부록 AJ. TabPFN 결과와 해석", "h1"),
        table(["데이터", "TabPFN v3", "대응 PP 결과", "해석"], [
            ["HUST", ".218±.030", "same-budget support PP .820±.055", "PP의 causal degradation structure가 크게 우세"],
            ["Virkler", ".621±.066", "support PP .886±.025", "적은 시편·엄격 crack tail에서 PP 우세"],
            ["NASA battery", "−.691±.009", "PP .513±.002", "4셀 health-tail에서 범용 prior 실패"],
            ["C-MAPSS FD002/004", ".607±.029", "support PP .747±.011", "115 features·3,000 cap에서도 PP +.140"],
            ["Sunwoda", "ensemble −.886", "PP-X .939", "1,000/1,312 train rows; strict unseen-cell tail 실패"],
            ["RWTH", "ensemble −2.174", "PP-X .878", "1,000/2,295 rows; strict tail 실패"],
            ["MATR batch2", "ensemble .618", "최종 개발 PP-X .862", "양수지만 transport route보다 낮음"],
            ["MICH", "ensemble −1.860", "PP-X .751", "동일 202 test rows의 보조 비교"],
        ], [35 * mm, 35 * mm, 47 * mm, 57 * mm], size=7),
        P("왜 TabPFN이 여기서 약할 수 있나", "h2"),
        P("TabPFN의 강점은 일반 tabular task prior와 in-context 관계 추론이다. 그러나 이번 평가는 train support 밖의 ordered degradation tail, "
          "새 physical unit, 특정 failure boundary를 동시에 요구한다. 범용 synthetic prior가 ‘RUL은 경계에서 0’, ‘열화 방향’, "
          "‘support 밖 residual authority’를 자동으로 보장하지 않는다. 반대로 이는 TabPFN 전체가 열등하다는 뜻이 아니라 "
          "우리 strict-tail protocol과 capped local 구성에서의 결과다."),
        P("TabPFN 비교가 PP-X 노벨티에 주는 의미", "h2"),
        P("대형 pretrained foundation model도 구조 계약 없는 외삽에서는 실패할 수 있다는 보조 근거다. PP-X의 기여는 파라미터 규모 경쟁보다 "
          "도메인 prior의 사용 자격·수정 권한·fallback을 명시하는 데 있다. 다만 TabPFN이 8개 전 setting의 완전 동일예산 주 비교군은 아니므로 "
          "‘PP-X가 TabPFN v3보다 보편적으로 우월’이라고 주장하지 않는다."),
        PageBreak(),

        P("부록 AK. TabPFN 관련 예상 질문", "h1"),
        table(["질문", "답변"], [
            ["TabPFN도 prior를 쓰는데 PP-X와 차이는", "TabPFN prior는 사전학습된 일반 tabular task 분포에 암묵적으로 들어 있다. PP-X prior는 현재 domain contract에서 명시적으로 admissible 여부와 residual 권한을 검사한다."],
            ["TabPFN은 fit을 안 하나", "`fit(X,y)` API는 train table을 준비·캐시하지만 일반 NN처럼 이 task의 모든 weight를 처음부터 epoch 학습하는 의미와 다르다. 핵심 예측은 pretrained model의 in-context inference다."],
            ["왜 estimator 하나만 썼나", "CPU·비용을 통제한 보조 비교 프로토콜이었다. Seeds 5개는 별도 estimator 실행 후 prediction ensemble했다."],
            ["왜 v3 최대 1M행을 안 썼나", "로컬 CPU 예산과 기존 protocol을 맞추기 위해 1,000/3,000 cap을 사용했다. Full-scale v3 성능 주장은 하지 않는다."],
            ["전처리와 categorical 처리는", "이번 X는 주로 연속형 causal summaries다. TabPFN 자체 전처리 prior를 사용했으며 test label 선택은 없었다."],
            ["TabPFN 결과가 음수면 모델이 나쁜가", "일반적으로 그렇다는 뜻이 아니다. 해당 strict hull-out RUL split에서 평균예측보다 제곱오차가 컸다는 뜻이다."],
            ["Equal-budget 8모델에 TabPFN이 왜 없나", "Local cap·estimator·계산 구조가 달라 완전한 30-candidate baseline protocol에 넣지 않고 보조 비교로 분리했다."],
            ["다시 실험한다면", "Full train rows, 공식 권장 ensemble/Thinking 설정, 고정 validation selection, 동일 compute/time budget을 사전등록하고 untouched cohort에서 비교해야 한다."],
        ], [59 * mm, 115 * mm]),
        P("근거", "h2"),
        P("구조 설명은 TabPFN-3 Technical Report(arXiv:2605.13986)와 Prior Labs 공식 v3 문서를 기준으로 했고, "
          "우리 실행 설정·수치는 `FAIR_PFN_COMPARISON_KO.md`, `TABPFN_EXTERNAL_BATTERIES_RESULTS_KO.md`, "
          "`experiments/tabpfn_external_batteries.py`의 저장 artifact를 기준으로 했다.", "small"),
    ]


def append_comparator_algorithms(story):
    story += [
        PageBreak(), P("부록 AL. 비교 알고리즘 전체 지도", "h1"),
        P("주 비교는 같은 train/validation/test adapter에서 8개 범용 방법을 각각 30개 validation 후보로 탐색하고, "
          "선택된 설정을 seeds 42–46으로 재학습했다. TabPFN은 train cap과 estimator 조건이 달라 별도 보조 비교다."),
        Image(str(BASELINE_FIG), width=174 * mm, height=80 * mm),
        table(["계열", "알고리즘", "핵심 질문"], [
            ["일반 예측기", "Plain MLP, FT-Transformer", "강한 구조 prior 없이 비선형 관계를 학습하면 충분한가"],
            ["그룹 강건학습", "V-REx, GroupDRO", "Train group 간 위험을 균등화하면 새 unit/regime에 전이되는가"],
            ["형상 제약", "Monotone NN", "입력–출력 방향 제약이 tail을 안정화하는가"],
            ["확률·생성 회귀", "Engression", "조건부 분포와 pre-additive noise 학습이 외삽에 도움되는가"],
            ["커널·기저함수", "Linear-tail RBF, SVGP", "지역적 smoothness와 선형 tail/GP uncertainty가 유효한가"],
            ["사전학습 foundation", "TabPFN v3", "대규모 합성 tabular prior의 in-context inference가 통하는가"],
            ["제안법", "PP-X", "명시적 contract와 조건부 prior authority가 필요한가"],
        ], [37 * mm, 54 * mm, 83 * mm]),
        PageBreak(),

        P("부록 AM. Plain MLP와 FT-Transformer", "h1"),
        P("Plain MLP", "h2"),
        table(["구성", "설명"], [
            ["구조", "X→Linear(d,w)→활성함수→(depth만큼 hidden block)→Linear(w,1)→ŷ"],
            ["학습", "Train MSE를 Adam 계열 optimizer로 최소화; PP-X와 같은 causal X와 unit-balanced 구성"],
            ["30후보 축", "width {16,32,64}, depth {1,2,3} 조합, lr·weight decay 조합"],
            ["장점", "단순하고 강한 universal approximator; 입력 관계가 train/test에서 유지되면 우수"],
            ["외삽 한계", "Train 밖 함수값은 loss가 직접 제약하지 않아 tail이 상수화·폭주·비물리적 반전 가능"],
            ["PP-X와 차이", "Frozen prior, explicit residual envelope, contract approval이 없음; PP-X의 direct fallback이 될 수 있음"],
        ], [42 * mm, 132 * mm]),
        P("FT-Transformer", "h2"),
        table(["구성", "설명"], [
            ["Tokenization", "각 scalar feature를 별도 embedding/token으로 변환하고 feature identity를 보존"],
            ["Encoder", "Feature token 사이 multi-head self-attention→FFN→residual/normalization block 반복"],
            ["Readout", "CLS 또는 집약 token representation→regression head→ŷ"],
            ["30후보 축", "token dim 16/32/64, block 1/2, head 2/4, lr·weight decay"],
            ["장점", "Feature 간 고차 상호작용과 조건부 중요도를 attention으로 학습"],
            ["한계", "Attention도 support 밖 tail 방향·failure boundary를 자동 보장하지 않음"],
            ["실제 결과", "Virkler .890으로 PP-X .888보다 .002 높아 유일한 equal-budget 승리"],
        ], [42 * mm, 132 * mm]),
        PageBreak(),

        P("부록 AN. V-REx와 GroupDRO", "h1"),
        P("두 방법에서 group은 셀·시편·엔진 또는 source regime이다. 목적은 평균 train loss만 낮추는 대신 group별 성능 차이를 줄이는 것이다."),
        P("V-REx: Variance Risk Extrapolation", "h2"),
        P("<b>L = mean_g R_g + λ·Var_g(R_g)</b>. 각 group 위험 R_g의 분산에 벌점을 줘 특정 group만 잘 맞는 표현을 억제한다. "
          "Invariant Risk Minimization 계열의 실용적 대안으로, 여러 source environment에서 위험이 비슷한 predictor가 새 environment에도 "
          "전이되길 기대한다."),
        table(["장점", "한계", "PP-X와 차이"], [[
            "평균 성능을 유지하며 group 위험 불균형 감소",
            "Source group 분산이 낮다고 unseen tail 함수가 식별되는 것은 아님; λ 선택 민감",
            "명시적 prior/boundary가 없고 group invariance를 학습목표로 사용"
        ]], [58 * mm, 58 * mm, 58 * mm]),
        P("GroupDRO: Group Distributionally Robust Optimization", "h2"),
        P("<b>min_θ max_g R_g(θ)</b>를 근사한다. 학습 중 손실이 큰 group의 가중치를 높여 최악 group 성능을 개선한다. "
          "평균적으로 쉬운 셀보다 어려운 셀을 더 보게 만드는 방식이다."),
        table(["장점", "한계", "PP-X와 차이"], [[
            "알려진 source group 중 worst-case 개선",
            "새 test group이 source worst group과 같은 이동을 갖는다는 보장 없음; noise group 과대가중 가능",
            "PP-X는 group loss만 강건화하지 않고 admissible prior와 executor 실행 자체를 승인/거절"
        ]], [58 * mm, 58 * mm, 58 * mm]),
        P("실험 해석", "h2"),
        P("HUST에서는 GroupDRO가 강한 비교군이었다. Mixed 표에서는 .934, 동일예산 30후보에서는 .955로 PP-X .958과 매우 가까웠다. "
          "이는 HUST의 protocol group shift에서 robust group weighting이 유효함을 보여주지만, 다른 tail의 명시적 경계·residual 권한을 제공하지는 않는다."),
        PageBreak(),

        P("부록 AO. Monotone NN과 Engression", "h1"),
        P("Monotone NN", "h2"),
        P("기본 MLP loss에 선택 입력에 대한 출력 미분 부호 위반 penalty를 더한다. 예를 들어 health가 증가할수록 RUL이 감소하면 안 된다는 "
          "방향을 지정한다. 구현 후보는 MLP width/depth/lr/weight decay와 monotonic penalty 강도를 함께 탐색했다."),
        table(["보장/기대", "보장하지 않는 것"], [[
            "지정 좌표에서 증가·감소 방향 위반을 줄여 비물리적 되살아남 억제",
            "정확한 외삽 기울기·곡률·경계에서 0·새 regime의 scale은 단조성만으로 결정되지 않음"
        ]], [87 * mm, 87 * mm]),
        P("Engression", "h2"),
        P("Pre-additive noise를 입력/latent 생성과정에 주입하고 조건부 출력 분포를 학습하는 생성적 회귀 방법이다. "
          "단순 조건부 평균 하나보다 데이터 분포의 변화와 불확실성을 표현하며, 적절한 구조 가정 아래 extrapolation 성질을 목표로 한다. "
          "이번 비교에서는 hidden dim 32/64/128, lr, beta, layer 2–4, epoch 조합 30개를 validation으로 선택했다."),
        table(["장점", "한계", "실제 결과"], [[
            "조건부 분포와 비선형 관계를 유연하게 표현; DS03에서 강함",
            "Pre-additive noise/분포 가정이 맞아야 하며 명시적 failure boundary나 abstention policy와 다름",
            "N-CMAPSS retrospective .932로 PP-X .937과 근접; DS03 prospective .9013으로 PP-X .8818보다 높음"
        ]], [58 * mm, 58 * mm, 58 * mm]),
        PageBreak(),

        P("부록 AP. Linear-tail RBF와 SVGP", "h1"),
        P("Linear-tail RBF", "h2"),
        P("Train-only 표준화 z에 <b>[1, z, φ_RBF(z)]</b> design을 만든 뒤 Ridge 회귀한다. φ_RBF는 random Fourier features로 "
          "가우시안 RBF kernel을 근사한다. 선형 항은 멀리서 기본 tail을 담당하고 RBF 항은 train 주변의 곡률을 담당한다."),
        table(["30후보", "장점", "한계"], [[
            "gamma .01~1, Ridge alpha .03~100, random components 256~768",
            "지역 nonlinear 보정+명시적 linear tail; 학습 빠르고 안정적",
            "RBF 기저는 support 밖에서 약해지고 선형 tail의 물리 적합성은 자동 보장되지 않음"
        ]], [58 * mm, 58 * mm, 58 * mm]),
        P("SVGP: Sparse Variational Gaussian Process", "h2"),
        P("Gaussian Process는 kernel로 두 X의 유사도를 정의하고 함수값의 확률분포를 추론한다. Full GP의 O(n³) 비용을 줄이기 위해 "
          "소수 inducing points u를 사용하고 variational lower bound(ELBO)를 최적화한다."),
        table(["구조", "장점", "한계"], [
            ["Kernel k(x,x′)+inducing variables→posterior mean·variance", "예측과 불확실성을 함께 제공; smoothness prior가 명확", "Kernel이 멀리서 prior mean으로 돌아가 strict tail 방향이 약할 수 있음"],
            ["30후보: lr .003/.01/.03, lengthscale .3/1/3/10/30, noise .01/.1", "길이척도와 noise를 validation 선택", "Inducing 근사·kernel 선택·output scale에 민감"],
        ], [55 * mm, 59 * mm, 60 * mm]),
        P("PP-X와 차이", "h2"),
        P("두 방법 모두 smoothness/linear-tail이라는 generic function prior를 사용하지만, PP-X는 domain contract의 failure boundary와 "
          "causal history를 명시하고, residual 수정권한과 fallback 실행을 별도로 관리한다."),
        PageBreak(),

        P("부록 AQ. 비교군별 구조·가정·실패모드 요약", "h1"),
        table(["모델", "구조 prior", "주 학습목표", "외삽 기대", "대표 실패모드"], [
            ["MLP", "암묵적 NN smoothness", "평균 MSE", "Train 관계 연장", "Tail 폭주·반전"],
            ["FT-Transformer", "Feature attention", "평균 MSE", "상호작용 전이", "Boundary/방향 미보장"],
            ["V-REx", "Risk invariance", "평균+group risk 분산", "환경불변 관계", "새 이동이 source와 다름"],
            ["GroupDRO", "Worst source group", "최대 group risk", "최악 group 강건성", "Noise group 과대가중"],
            ["Monotone NN", "미분 부호", "MSE+위반 penalty", "방향 보존", "기울기·곡률 미식별"],
            ["Engression", "Pre-additive noise 분포", "생성적 distribution loss", "분포 기반 외삽", "분포가정 mismatch"],
            ["Linear-tail RBF", "선형 tail+RBF smoothness", "Ridge MSE", "멀리서 선형화", "틀린 선형 tail"],
            ["SVGP", "Kernel GP", "Variational ELBO", "Smooth posterior/UQ", "Prior mean 회귀"],
            ["TabPFN", "사전학습 task prior", "Pretrained ICL", "범용 tabular 추론", "Domain boundary 미명시"],
            ["PP-X", "Typed domain prior", "Prior residual+approval", "조건부 구조 실행", "Contract/validation mismatch"],
        ], [27 * mm, 37 * mm, 40 * mm, 35 * mm, 35 * mm], size=6.5),
        PageBreak(),

        P("부록 AR. 비교 알고리즘 관련 예상 질문", "h1"),
        table(["질문", "답변"], [
            ["왜 XGBoost/Random Forest가 주표에 없나", "초기 비교군에는 boosting 등이 있었지만 strict support 밖 tree는 leaf 상수화가 쉬워, 최종 주 비교는 외삽/representation/robustness 계열 8개로 맞췄다."],
            ["왜 PINN이 없나", "9개 setting 모두에 동일하게 정당화되는 지배방정식이 없다. Virkler의 Paris처럼 식이 닫히는 경로는 PAE future program으로 분리했다."],
            ["왜 LSTM/GRU가 없나", "일부 battery backbone·temporal 실험은 수행했지만 모든 9 setting이 동일 sequence schema가 아니어서 공통 equal-budget 표형 비교군으로 강제하지 않았다."],
            ["V-REx와 GroupDRO 차이는", "V-REx는 group risk 분산을 줄이고, GroupDRO는 가장 큰 group risk를 직접 중시한다."],
            ["Monotone이면 PP-X가 필요 없나", "방향만 정할 뿐 경계, 기울기, residual 폭, fallback을 정하지 못한다."],
            ["GP가 uncertainty를 주니 더 안전한가", "Posterior variance는 kernel/model 가정 아래 불확실성이다. OOD 안전인증이나 올바른 tail을 자동 보장하지 않는다."],
            ["Engression이 DS03에서 이겼으면 채택해야 하나", "Predictive benchmark에서는 강한 결과다. 그러나 PP-X 연구질문은 구조 가정 승인/fallback이며, 결과를 본 뒤 paper method를 Engression으로 바꾸면 사후선택이다."],
            ["비교가 완전히 공정한가", "8개 baseline의 후보 수·split·정보·refit seed를 맞췄다. 모델별 계산량과 PP-X 과거 개발비용이 완전히 같다는 주장은 하지 않는다."],
        ], [59 * mm, 115 * mm]),
    ]


def append_comparison_protocol(story):
    story += [
        PageBreak(), P("부록 AS. 발표 비교 장표의 정확한 프로토콜", "h1"),
        P("발표의 비교 장표는 서로 다른 두 근거 층을 포함한다. <b>Tier A mixed strongest same-split retrospective</b>와 "
          "<b>Tier B uniformly tuned equal-budget retrospective</b>를 합치지 않는다."),
        table(["구분", "Tier A: mixed strongest", "Tier B: equal-budget"], [
            ["질문", "각 setting에서 저장된 강한 same-split 비교군보다 PP-X가 높은가", "모든 범용 비교군에 같은 후보 수를 주면 결과가 유지되는가"],
            ["Settings", "Paper-main 9개", "같은 9개"],
            ["비교군", "데이터셋별 locally strongest/row-aligned 비교군이 일부 다름", "8개 baseline 전부"],
            ["튜닝 예산", "과거 artifact별 이질적", "모델당 정확히 30 validation candidates"],
            ["재학습", "Artifact별 조건", "선택 설정을 seeds 42–46으로 5회 refit"],
            ["결과", "9/9, 양측 sign p=.00390625", "8/9, 양측 sign p=.0391"],
            ["한계", "Comparator budget 이질적", "PP-X 과거 전체 구조개발 비용까지 30회라는 뜻 아님"],
        ], [31 * mm, 71 * mm, 72 * mm], size=7),
        callout("<b>장표를 말할 때:</b> ‘9/9’와 ‘8/9’를 동시에 최고 성능처럼 섞지 않는다. "
                "9/9는 breadth를, 8/9는 범용 baseline의 후보 수 공정성을 답한다."),
        PageBreak(),

        P("부록 AT. Equal-budget 공통 실행 순서", "h1"),
        table(["단계", "실제 규칙", "누수 방지 의미"], [
            ["1 Adapter 고정", "각 setting의 기존 train/validation/test X,y,groups를 그대로 사용", "모델마다 쉬운 split 사용 금지"],
            ["2 정보 동일", "모든 범용 비교군은 같은 causal X와 Y를 입력", "PP-X만 미래정보를 갖는 비교 방지"],
            ["3 후보 생성", "각 모델 family마다 서로 다른 30개 hyperparameter configuration", "후보 개수 차이로 인한 선택 이점 통제"],
            ["4 Search", "search seed 42로 각 후보를 train하고 validation MSE 계산", "Test score로 hyperparameter 선택 금지"],
            ["5 선택", "Validation MSE 최소 후보; 동률은 deterministic order", "선택 규칙 고정"],
            ["6 Refit", "선택된 설정을 seeds 42,43,44,45,46으로 재학습", "초기화 안정성과 ensemble 평가"],
            ["7 Test", "고정 test rows를 예측; train-derived output contract 적용", "Test는 최종 평가에만 사용"],
            ["8 집계", "Seed별 pooled R²와 5-seed prediction ensemble R²", "Seed 평균과 ensemble 혼동 방지"],
            ["9 Unit audit", "최강 row-aligned 비교군과 unit log-RMSE ratio", "상관된 row를 독립 표본으로 세지 않음"],
            ["10 Setting sign", "각 setting의 승/패 9개로 exact sign test", "5 seeds를 n=45로 부풀리지 않음"],
        ], [22 * mm, 94 * mm, 58 * mm], size=7),
        P("총 실행량", "h2"),
        P("9 settings×8 baselines×30 candidates의 search와 선택 설정 5회 refit을 포함해 기록된 총 학습 job은 3,360개, "
          "모델 실행시간 합계는 약 3.52시간이다. 하드웨어 wall-clock과 모델별 FLOPs가 동일하다는 뜻은 아니다."),
        PageBreak(),

        P("부록 AU. 모델별 30개 후보 grid", "h1"),
        table(["모델", "후보를 만든 축", "30개 구성 방식"], [
            ["Plain MLP / V-REx / GroupDRO / Monotone", "width·depth·lr·weight decay·penalty",
             "구조 (16×1,32×1,32×2,64×2,64×3) 5개 × 최적화/penalty 묶음 6개"],
            ["FT-Transformer", "d_token·blocks·heads·lr·weight decay",
             "구조 (16,1,2),(32,2,4),(64,2,4) 3개 × lr/wd 10개"],
            ["Engression", "hidden_dim·lr·beta·layers·epochs",
             "hidden 32/64/128 3개 × noise/optimizer/depth/epoch 묶음 10개"],
            ["Linear-tail RBF", "gamma·Ridge alpha·random components",
             "gamma .01/.03/.1/.3/1 5개 × alpha/components 묶음 6개"],
            ["SVGP", "lr·kernel lengthscale·noise",
             "lr .003/.01/.03 3개 × lengthscale .3/1/3/10/30 5개 × noise .01/.1 2개"],
        ], [39 * mm, 66 * mm, 69 * mm], size=7),
        P("Penalty 축 주의", "h2"),
        P("Plain MLP는 monotone branch의 penalty=0인 exact plain arm을 사용한다. V-REx·GroupDRO·Monotone은 같은 기본 "
          "width/depth/optimizer budget 안에서 각 방법의 penalty 의미로 변환한다. 서로 다른 알고리즘이 동일 hyperparameter 이름을 "
          "가질 수 없으므로 후보 <b>수</b>를 맞춘 것이지 search space의 계산 난이도가 수학적으로 동일하다는 뜻은 아니다."),
        P("PP-X는 왜 30-grid에 다시 넣지 않았나", "h2"),
        P("PP-X는 typed contract에 따라 admissible executor 종류 자체가 setting마다 다르고 paper method가 이미 동결돼 있었다. "
          "동일예산 비교를 위해 PP-X를 새 공통 30-grid로 재개발하면 test를 이미 본 뒤 paper method를 바꾸게 된다. "
          "따라서 frozen PP-X prediction을 reference로 두고 8개 범용 baseline의 후보 수 공격을 검증했다."),
        PageBreak(),

        P("부록 AV. 데이터셋별 비교 입력·평가 단위", "h1"),
        table(["Setting", "동일 test 단위", "주 비교 특징", "특별 처리"], [
            ["HUST", "16셀·7,775행", "capacity/cycle/rate causal X", "Protocol-tail; equal-budget 최강 GroupDRO .955"],
            ["Virkler", "10시편·20행", "crack length/time/rate", "FT .890가 PP-X .888보다 .002 높음"],
            ["NASA battery", "4셀·255행", "4-fold LOO health φ", "Fold별 validation 선택 후 row-aligned 결합"],
            ["Sunwoda", "9셀·1,468창", "7개 causal window summary", "Linear-tail RBF .838"],
            ["RWTH", "8셀·1,859창", "동일 causal summary", "Linear-tail RBF .732"],
            ["MICH", "8셀·202창", "동일 causal summary", "Monotone NN −.686; PP-X 관계이동 복구"],
            ["MATR2019", "10셀·2,235행", "Q/rate 8-step summary", "FT .342"],
            ["MATR batch2", "9셀·733행", "Q/rate summary", "Plain MLP .813"],
            ["N-CMAPSS", "3엔진·159행", "운전조건+센서 causal feature", "Engression .932; unit effect 이질적"],
        ], [32 * mm, 37 * mm, 59 * mm, 46 * mm], size=7),
        P("Primary metric", "h2"),
        P("모델 순위의 주 지표는 모든 고정 test row의 <b>raw pooled R²</b>다. 보조로 RMSE·MAE·unit-macro R²·per-unit RMSE를 보고한다. "
          "Prediction ensemble R²는 먼저 5개 seed 예측을 행별 평균한 뒤 한 번 R²를 계산한다. Seed별 R² 평균과 같지 않다."),
        PageBreak(),

        P("부록 AW. Strongest comparator와 paired comparator", "h1"),
        table(["용어", "정의", "왜 분리하나"], [
            ["Strongest pooled comparator", "같은 split에서 보고된 test pooled R²가 가장 높은 비교모델", "기술적 최고 성능과 PP-X gap을 보수적으로 표시"],
            ["Row-aligned paired comparator", "같은 test 행 순서·unit ID의 prediction 배열이 저장된 비교모델", "Unit별 paired RMSE와 통계검정 가능"],
            ["Equal-budget strongest", "30-candidate 재학습 8개 중 test pooled R² 최고", "후보 수 공정성 표의 secondary 보수적 비교"],
        ], [43 * mm, 73 * mm, 58 * mm]),
        P("Test 최고 비교군을 고른 통계의 한계", "h2"),
        P("각 dataset에서 test pooled R²가 가장 높은 baseline을 고른 뒤 PP-X와 unit 검정하면, 비교군 선택 자체가 test를 사용한다. "
          "이는 PP-X에 불리한 보수적 secondary audit이지만 사전 지정된 단일 가설검정은 아니다. 그래서 전체 모델표·dataset sign test와 "
          "unit paired audit을 분리해 보고한다."),
        table(["Mixed paired audit 예", "Strongest pooled와 row-aligned가 다른 이유"], [
            ["RWTH", "최고 표 수치와 prediction array가 저장된 matched direct/linear-tail 비교가 증거 시점에 달랐음"],
            ["MATR-b2", "최강 pooled V-REx 수치와 완전 row-aligned GroupDRO artifact 구분"],
            ["N-CMAPSS", "Engression pooled 최고권과 저장 monotone prediction의 paired 검정 구분"],
        ], [47 * mm, 127 * mm]),
        PageBreak(),

        P("부록 AX. TabPFN 비교 프로토콜은 왜 별도인가", "h1"),
        table(["항목", "TabPFN 보조 프로토콜", "Equal-budget 8-baseline과 차이"], [
            ["Model", "Local TabPFN v3 CPU, seed당 estimator 1", "30개 hyperparameter candidate search 아님"],
            ["Seeds", "42–46", "같지만 seed에는 model state+subsample 변화 포함 가능"],
            ["Train cap", "외부 4셋 최대 3,000; battery 3셋 최대 1,000", "일부 baseline은 full train 사용"],
            ["Sampling", "Unit별 균등, 시간순 균등간격 또는 고정 random subsample", "전체 행 학습과 다름"],
            ["Test", "PP와 동일 frozen rows", "이 부분은 동일"],
            ["Selection", "Fixed v3 config; test-label selection 없음", "Validation 후보 30개 선택과 다름"],
            ["목적", "강한 pretrained tabular foundation model 보조 비교", "Equal-budget 최종 순위 결정 아님"],
        ], [37 * mm, 67 * mm, 70 * mm]),
        callout("<b>정확한 표현:</b> ‘TabPFN v3와 동일 test·동일 causal information의 capped supplementary comparison을 수행했다.’ "
                "‘Full-scale TabPFN v3를 동일 compute로 완전 공정 비교했다’고 쓰면 안 된다."),
        PageBreak(),

        P("부록 AY. 비교 장표 결과를 읽는 순서", "h1"),
        table(["순서", "확인할 것", "발표 문장"], [
            ["1 Score matrix", "각 행은 setting, 각 열은 모델; 색과 평균만 보지 말고 음수 실패 확인", "PP-X 평균 .81, GroupDRO .36, V-REx .35"],
            ["2 Robust count", "양수 R² setting 수", "PP-X 9/9, GroupDRO 7/9, V-REx 7/9"],
            ["3 Equal-budget", "30-candidate 최강과 setting별 승패", "PP-X 8/9, Virkler −.002"],
            ["4 Unit evidence", "Pooled 승리가 특정 긴 unit 때문인지", "Sun/RWTH/MICH/MATRb2는 equal-budget paired BH q<.05"],
            ["5 Exceptions", "NASA 사실상 동률, N-CMAPSS unit 이질성", "보편 우월을 주장하지 않음"],
            ["6 Prospective", "Retrospective 표와 DS03 분리", "DS03 Engression이 더 정확"],
        ], [24 * mm, 83 * mm, 67 * mm]),
        P("최종 비교 주장", "h2"),
        P("동일한 후보 수를 부여한 8개 범용 비교군에 대해 PP-X는 9개 retrospective setting 중 8개에서 가장 높은 pooled R²를 보였다. "
          "그러나 Virkler 열세, NASA 사실상 동률, N-CMAPSS unit 이질성, TabPFN capped protocol, PP-X 과거 개발예산 비대칭, "
          "DS03 prospective 열세 때문에 universal SOTA가 아니라 <b>조건부 구조 실행의 retrospective utility</b>로 해석한다."),
        PageBreak(),

        P("부록 AZ. 비교 프로토콜 예상 질문", "h1"),
        table(["질문", "답변"], [
            ["왜 후보 수만 맞추고 시간은 안 맞췄나", "모델 family별 연산구조가 달라 후보 수를 사전 통제했다. Wall-clock/FLOPs 완전 동일 비교는 아니며 한계로 공개한다."],
            ["왜 search seed는 42 하나인가", "30후보 ranking 비용을 통제하고 선택 후 seeds 42–46 refit으로 초기화 안정성을 평가했다."],
            ["Validation도 한 번뿐인가", "Dataset의 기존 group-disjoint validation 또는 NASA fold별 validation을 사용했다. Test는 선택에 쓰지 않았다."],
            ["왜 test pooled 최고 baseline을 고르나", "PP-X에 유리한 약한 baseline 선택을 피하기 위한 보수적 기술 비교다. 해당 paired p-value는 사전 단일가설이 아니므로 secondary로 표시한다."],
            ["Seed 5개를 t-test 했나", "하지 않았다. Seed는 독립 물리표본이 아니므로 unit paired 검정과 setting sign test를 사용했다."],
            ["R² 음수를 평균에 넣었나", "숨기지 않고 raw pooled R²를 그대로 사용했다. 음수는 평균 predictor보다 나쁜 실패다."],
            ["Output clipping이 공정한가", "각 실행의 사전 output contract와 train-derived cap을 사용했다. Test label로 cap을 정하지 않았다."],
            ["PP-X도 새로 30개 튜닝해야 하지 않나", "이미 동결된 방법을 열린 test 뒤 재개발하면 사후편향이 생긴다. Baseline 후보수 공격만 별도 감사한 설계다."],
            ["결론적으로 SOTA인가", "Retrospective equal-budget 8/9 우세지만 prospective DS03에서 Engression보다 낮아 보편 SOTA는 아니다."],
        ], [59 * mm, 115 * mm]),
    ]


def build():
    make_result_figure()
    make_actual_prediction_figures()
    make_architecture_figure()
    make_tabpfn_figure()
    make_baseline_figure()
    story = [Spacer(1, 30 * mm), P("PP-X 발표 데이터셋<br/>완전 학습 가이드", "title"),
             P("PPT 37장 상세 + 메인·보조 데이터셋, X/Y, 스플릿, 도메인", "sub"),
             Spacer(1, 12 * mm), callout(
                 "<b>한 문장 요약</b><br/>현재까지 본 상태(X)를 보고, 그 물건이 고장 기준에 닿기까지 "
                 "얼마나 남았는지(Y)를 맞힌다. 단, 학습 때 본 물건과 시험 물건을 분리하고, "
                 "학습 때 본 건강 범위보다 더 나빠진 구간만 시험한다.<br/>"
                 "<b>이번 판:</b> <font face='AppleGothic'>PP_Research_Detailed_v2.pptx</font> "
                 "37장 본문을 앞부분에 상세 해설로 수록."),
             Spacer(1, 14 * mm), P("작성자: 박진서 · 2026-09-12", "sub"),
             P("근거: 발표 생성 소스, frozen protocol, 실제 dataset adapter 및 row builder. "
               "코드로 확인할 수 없는 물리 약어는 추측하지 않음.", "small"), PageBreak()]

    append_ppt_deck_detail(story, P, table, callout)

    story += [P("0. 이 자료를 읽기 전에", "h1"),
              P("<b>X</b>는 모델에게 주는 단서, <b>Y</b>는 모델이 맞혀야 하는 정답이다. "
                "예를 들어 현재 배터리 용량과 최근 감소속도가 X이고, 고장까지 남은 사이클이 Y다."),
              table(["말", "초등학생식 뜻", "이 연구에서의 뜻"], [
                  ["unit", "물건 한 개", "배터리 셀 1개, 금속 시편 1개, 엔진 1개"],
                  ["row", "특정 시점의 한 문제", "한 unit의 현재 상태 X와 남은수명 Y 한 쌍"],
                  ["train", "문제집으로 공부", "가중치와 정규화 통계를 학습"],
                  ["validation", "모의고사", "모델·executor·epoch를 선택; test는 보지 않음"],
                  ["test", "최종시험", "선택이 끝난 뒤 한 번 평가"],
                  ["RUL", "고장까지 남은 횟수", "Remaining Useful Life"],
                  ["support", "학습에서 실제로 본 범위", "예: train의 건강도 최저~최고"],
                  ["hull-out", "문제집 범위 밖", "선언한 좌표에서 test가 train 범위 밖"],
                  ["causal feature", "그때 이미 알 수 있는 단서", "미래 관측·최종수명 없이 현재까지 계산"],
              ], [28 * mm, 57 * mm, 89 * mm]),
              P("왜 랜덤 row split을 쓰지 않나", "h2"),
              P("같은 셀의 어제 기록이 train이고 내일 기록이 test라면 모델은 사실상 그 셀을 이미 본 셈이다. "
                "그래서 먼저 unit을 나눈다. 그런 다음 train에는 비교적 건강한 구간, validation/test에는 "
                "train보다 더 열화된 구간만 남긴다."),
              callout("<b>공통 순서</b>: 실패경계·unit·시간·입력·외삽좌표 선언 → unit 분리 → "
                      "train support 제한 → val/test hull-out 필터 → train-only 표준화 → "
                      "validation-only 선택 → frozen test 평가"), PageBreak()]

    story += [P("1. 발표에서 메인과 보조는 무엇인가", "h1"),
              table(["등급", "데이터셋", "올바른 해석"], [
                  ["메인 retrospective 9-setting", "HUST, Virkler, NASA battery, Sunwoda, RWTH, MICH, MATR2019, MATR batch2, N-CMAPSS",
                   "PP-X paper-main 포트폴리오. 이미 관측·개발된 split이므로 새 미개봉 확증은 아님."],
                  ["별도 prospective", "N-CMAPSS DS03", "미리 고정한 route가 fallback을 골라 route-selection PASS. Engression보다 낮아 superiority FAIL."],
                  ["보조·한계", "XJTU, FEMTO, NASA milling", "transfer/endpoint/material 이동의 실패·복구 사례. 메인 9승에 포함 금지."],
                  ["future program", "PAE", "Virkler 등에서 식 기반 경로 가능성을 보인 후속 개념. PP-X 주표와 혼합 금지."],
              ], [35 * mm, 63 * mm, 76 * mm]),
              P("메인 9개가 모두 똑같은 문제는 아니다", "h2"),
              P("배터리 7개, 균열 1개, 엔진 시뮬레이션 1개다. 또한 ‘외삽’의 축도 배터리 건강도, "
                "균열 길이, 엔진 운전조건으로 다르다. 따라서 9개 점수를 한 데이터셋의 반복처럼 보면 안 된다."),
              callout("<b>발표용 핵심 문장:</b> ‘PP-X는 하나의 고정 네트워크를 모든 데이터에 강제한 것이 아니라, "
                      "공통 contract 아래 validation이 승인한 executor를 데이터 설정별로 동결한 framework다.’"),
              PageBreak()]

    story += [P("2. 제품 수와 예측 결과를 한눈에", "h1"),
              P("아래는 저장된 최종 PP-X result artifact에서 읽은 실제 test 점수와 물리 unit 수다. "
                "파란 막대는 실제 Y와 예측 Y 전체를 비교한 pooled R², 청록 막대는 시험 제품 수, "
                "주황 막대는 같은 test unit에서 paired baseline보다 RMSE가 낮았던 비율이다."),
              Image(str(RESULT_FIG), width=174 * mm, height=57.5 * mm),
              table(["셋", "전체 제품", "Train / Val / Test 제품", "Test 행·창", "R²"], [
                  ["HUST", "77셀", "45 / 16 / 16", "7,775", ".958"],
                  ["Virkler", "68시편", "48 / 10 / 10", "20", ".888"],
                  ["NASA battery", "4셀", "fold마다 2 / 1 / 1", "합계 255", ".584"],
                  ["Sunwoda", "18셀", "7 / 2 / 9", "1,468", ".939"],
                  ["RWTH", "39셀", "23 / 8 / 8", "1,859", ".878"],
                  ["MICH", "32셀", "18 / 6 / 8", "202", ".751"],
                  ["MATR2019", "원본45·유효44셀", "26 / 8 / 10", "2,235", ".466"],
                  ["MATR batch2", "48셀", "30 / 9 / 9", "733", ".862"],
                  ["N-CMAPSS", "9엔진", "hard TRA 분할 / test 3", "159", ".937"],
              ], [29 * mm, 34 * mm, 52 * mm, 31 * mm, 28 * mm]),
              P("<b>중요:</b> 행 수는 제품 수가 아니다. 예를 들어 HUST test는 셀 16개에서 시점 7,775개가 나온다. "
                "추론 단위와 일반화 질문은 ‘7,775개의 독립 제품’이 아니라 ‘처음 보는 16개 셀’이다.", "small"),
              PageBreak()]

    for d in DATASETS:
        dataset_pages(story, d)

    story += [P("11. 보조·한계 데이터셋", "h1"),
              table(["셋", "도메인 / X", "Y", "분할·발표에서의 역할"], [
                  ["XJTU-SY", "회전 베어링 진동. 시간영역 요약·스펙트럼/진행도 계열 causal feature.",
                   "베어링 파손까지 남은 기록시간/진행량.",
                   "운전조건 37.5Hz·11kN train, 35Hz·12kN val, 40Hz·10kN test의 condition transfer. 0.257은 test를 본 뒤 만든 retrospective 복구; 메인 아님."],
                  ["FEMTO/PRONOSTIA", "베어링 수평·수직 진동 waveform에서 RMS, 표준편차, kurtosis, crest factor, 주파수 band 등.",
                   "기록 종료/고장 endpoint까지 남은 시간.",
                   "처음 보는 bearing endpoint transfer. 과거 잘못된 시간 metadata 채널 사용 결과는 철회. 최종 ensemble만 양수이고 개별 seed는 음수; limitation."],
                  ["NASA milling", "밀링 공구마모. health, rate 및 causal 공정/센서 특징.",
                   "공식 flank-wear 경계 VB=0.50까지 남은 절삭 진행량.",
                   "material-1 train에서 material-2 test로 이동. 0.341은 test 관측 후 boundary-quotient 개발 결과; 외부 확증 아님."],
              ], [28 * mm, 55 * mm, 42 * mm, 49 * mm]),
              P("왜 이 셋을 메인 표에 섞지 않나", "h2"),
              P("이들은 단순한 late-health 1D hull-out보다 도메인·운전조건·재료·endpoint 정의의 이동이 강하다. "
                "또 일부는 시험 결과를 본 뒤 복구 모델을 개발했다. 학습에는 중요하지만, 독립 확증 성능처럼 세면 데이터 누수와 주장 과장이 된다."),
              PageBreak()]

    story += [P("12. X를 만드는 공통 원리", "h1"),
              P("배터리 8-step 창을 예로 들면 과거 8개 점만 잘라 작은 ‘짧은 영상’을 만든다. "
                "그 영상에서 현재값, 평균, 처음과 끝의 차이를 뽑는다. 미래 9번째 점은 사용하지 않는다."),
              table(["요약", "계산", "직관"], [
                  ["last", "창의 8번째 값", "바로 지금 상태"],
                  ["mean", "8개 평균", "최근 상태의 안정된 대표값"],
                  ["endpoint change", "8번째−1번째", "최근 방향과 변화량"],
                  ["prefix rate", "시작부터 현재까지 누적 선형기울기", "장기 열화 속도"],
                  ["recent rate", "최근 고정 구간의 기울기/차이", "단기 열화 속도"],
                  ["normalized cycle", "현재 cycle ÷ train 최대 cycle", "train 기준 시간 진행도"],
              ], [38 * mm, 62 * mm, 74 * mm]),
              P("표준화는 어떻게 하나", "h2"),
              P("각 X 열에서 train 평균을 빼고 train 표준편차로 나눈다. validation/test의 평균이나 표준편차는 "
                "계산에 넣지 않는다. Y scale과 출력 cap도 train에서만 정한다. 그래야 시험지 전체를 미리 훑어본 효과가 생기지 않는다."),
              P("groups는 X가 아니다", "h2"),
              P("groups(unit ID)는 셀별 균형 가중치와 unit별 통계를 계산하는 표찰이다. 일반적인 모델 입력 X로 넣지 않는다. "
                "같은 셀의 행이 많다고 그 셀이 결과를 독점하지 않도록 전체 unit의 총 가중치를 같게 만든다."),
              PageBreak()]

    story += [P("13. Y와 점수 읽기", "h1"),
              P("대부분 Y는 ‘고장 cycle − 현재 cycle’이다. 하지만 고장 정의는 데이터마다 다르다. "
                "880 mAh, 상대용량 80%, 1.401 Ah, 공개 life label, 파일 마지막 cycle, 균열 49.8 mm처럼 서로 다르다."),
              table(["지표", "쉬운 뜻", "주의"], [
                  ["R²", "항상 평균만 말하는 바보 예측보다 얼마나 나은가", "1이 완벽, 0이 평균예측 수준, 음수면 평균보다 나쁨."],
                  ["RMSE", "큰 실수에 더 벌점을 주는 평균오차", "Y 단위와 같음. cycle RUL이면 cycle 단위."],
                  ["MAE", "틀린 크기의 절댓값 평균", "RMSE보다 큰 이상치 영향이 작음."],
                  ["pooled R²", "모든 test row를 한데 모아 계산", "행이 긴 unit의 영향이 커질 수 있음."],
                  ["unit-macro R²", "unit별 R²를 구해 똑같이 평균", "짧고 긴 unit을 동등하게 봄."],
                  ["5-seed ensemble", "5개 모델 예측의 평균", "seed 5개는 물리 실험 5회가 아니며 독립 표본으로 세면 안 됨."],
              ], [35 * mm, 70 * mm, 69 * mm]),
              callout("<b>R²가 양수라고 안전하다는 뜻은 아니다.</b> 이 연구의 결과는 예측 성능과 구조적 경계를 평가한 것이지, "
                      "실제 설비 안전 인증이나 고장 무발생 보장이 아니다.", colors.HexColor("#FEE2E2")),
              PageBreak()]

    story += [P("14. 실제 Y와 예측 Y — 전체 test 행", "h1"),
              P("각 점 하나가 test의 한 시점이다. 가로축은 실제 RUL, 세로축은 PP-X 5-seed ensemble 예측 RUL이다. "
                "주황 점선에 가까울수록 정확하다. 이 그림은 표의 R²를 만든 동일한 frozen prediction artifact에서 직접 그렸다."),
              Image(str(SCATTER_FIG), width=174 * mm, height=158 * mm),
              P("배터리 3종(Sunwoda/RWTH/MICH)의 저장 배열은 모델 내부 time scale로 정규화되어 있으나 "
                "actual과 prediction에 같은 scale이 적용되어 점의 일치도와 R²는 변하지 않는다.", "small"),
              PageBreak(),
              P("15. 실제 Y와 예측 Y — 대표 test 제품", "h1"),
              P("각 데이터셋에서 test 행이 가장 많은 unit 하나를 대표로 골랐다. 검정은 실제 RUL, 파랑은 예측 RUL이다. "
                "가로축은 저장된 test-row 순서이며, 원래 cycle 열이 공통 artifact에 없는 데이터가 있어 ‘실제 시간’으로 과장하지 않았다."),
              Image(str(TRAJECTORY_FIG), width=174 * mm, height=158 * mm),
              P("대표 unit 그림은 이해를 위한 예시다. 성능 수치는 특정 unit 하나가 아니라 모든 test unit·행으로 계산한다. "
                "전체 행별 y_true/y_pred는 docs/ppx_actual_vs_predicted_all_main_datasets.npz에도 저장했다.", "small"),
              PageBreak()]

    story += [P("16. 자주 나오는 오해 12개", "h1"),
              table(["오해", "정정"], [
                  ["test는 같은 셀의 미래다", "아니다. 기본 원칙은 train/val/test unit-disjoint다."],
                  ["X에 최종 수명이 들어간다", "아니다. Y를 만들 때만 EOL을 쓰고 모델 입력 X에는 넣지 않는다."],
                  ["1D hull-out 100%=모든 특징공간 밖", "아니다. 선언한 health/crack/TRA 축 밖이라는 뜻이다."],
                  ["NASA 두 행은 같은 데이터", "아니다. NASA battery는 4개 셀, N-CMAPSS는 엔진 시뮬레이션이다."],
                  ["HUST Y는 전부 실제 EOL", "아니다. terminal-slope proxy RUL이다."],
                  ["MICH는 코드로 80% crossing을 다시 계산", "아니다. 공개 life label을 쓴다."],
                  ["MATR19와 batch2는 같은 split", "아니다. 파일·연도·셀 구성·split이 다르다."],
                  ["N-CMAPSS 0.937=DS03 prospective", "아니다. 0.937은 DS02-006 retrospective; DS03 selected fallback은 0.8818."],
                  ["모든 셋에 dual-scale을 켰다", "아니다. RWTH에서 악화돼 MICH 등에 조건부 사용."],
                  ["9/9는 미래 성공확률 100%", "아니다. retrospective settings의 sign 결과다."],
                  ["보조 3셋도 9승에 포함", "아니다. XJTU/FEMTO/milling은 limitation/development."],
                  ["PP-X가 Engression을 prospective에서 이겼다", "아니다. DS03에서 0.8818 < 0.9013."],
              ], [68 * mm, 106 * mm]),
              PageBreak()]

    story += [P("17. 발표 직전 1페이지 암기장", "h1"),
              table(["셋", "물건", "핵심 X", "Y/경계", "split 한 줄"], [
                  ["HUST", "배터리", "용량·cycle·장/단기 rate", "proxy RUL / 0.880Ah", "protocol 1–6 / 7–8 / 9–10"],
                  ["Virkler", "금속 시편", "균열길이·시간·성장rate", "49.8mm까지", "48 / 10 / 10 시편; ≤33 vs >33mm"],
                  ["NASA batt.", "배터리 4셀", "health φ 1개", "1.401Ah까지", "2 train / 1 val / 1 test LOO"],
                  ["Sunwoda", "상용 배터리", "7개 window 요약", "880mAh까지", "25°C 7/2, 35°C 9"],
                  ["RWTH", "배터리", "7개 window 요약", "상대 0.8까지", "셀 2–24 / 25–32 / 33–40"],
                  ["MICH", "배터리", "7개 window 요약", "공개 life label", "셀 1–18 / 19–24 / 25–32"],
                  ["MATR19", "배터리", "Q/rate 6개 요약", "파일 마지막 cycle", "유효 셀 60/20/20%"],
                  ["MATRb2", "배터리 48셀", "Q/rate 6개 요약", "파일 마지막 cycle", "30 / 9 / 9셀"],
                  ["N-CMAPSS", "가상 엔진", "조건4+센서14+가상14", "engine RUL", "미지 unit+high TRA hard split"],
              ], [25 * mm, 28 * mm, 48 * mm, 38 * mm, 35 * mm], size=7),
              Spacer(1, 5 * mm),
              callout("<b>20초 답변:</b> ‘배터리·균열·엔진의 남은수명을 맞히는 데이터입니다. "
                      "먼저 셀·시편·엔진 자체를 train/validation/test로 분리하고, train에는 건강한 구간, "
                      "validation/test에는 train에서 본 범위보다 더 열화된 구간만 남겼습니다. "
                      "X는 현재까지 관측 가능한 건강도와 열화속도, 운전조건이고 Y는 각 데이터의 고장경계까지 남은 cycle입니다.’"),
              PageBreak()]

    story += [P("18. 근거와 검증 범위", "h1"),
              P("이 문서는 다음 실제 구현을 교차 확인해 작성했다."),
              table(["근거", "확인한 내용"], [
                  ["pp_pae_total/slides/pp_research/build_pp_research_deck_v2.py", "발표의 메인·appendix 구성과 데이터셋 역할"],
                  ["pp-extrapolation/MODEL_AND_SPLIT_KO.md", "공통 unit-disjoint·hull-out·val-only 원칙"],
                  ["ca-css-ncmapss/run_affine_tail_external_three.py", "HUST·Virkler·NASA 변수, target, split"],
                  ["ca-css-ncmapss/pae_boundary_realdata.py", "Sunwoda·RWTH·MICH 원자료, EOL, unit split, 8-step window"],
                  ["pp-extrapolation/experiments/distance_uncertainty_pp.py", "배터리 7개 요약 X의 정확한 열 순서"],
                  ["matr_2019_latent_confirmatory.py / matr_batch2_confirmatory.py", "MATR 셀 split, 6개 X, Y와 boundary"],
                  ["ca-css-ncmapss/apps/ncmapss_data_utils.py", "N-CMAPSS 32개 feature 이름과 unit 정의"],
                  ["PPX_TOP_JOURNAL_VALIDATION_PACKAGE_KO.md", "메인 9개, prospective DS03, 주장 한계"],
              ], [72 * mm, 102 * mm]),
              P("알려진 한계", "h2"),
              P("N-CMAPSS 센서 station의 세부 물리 정의와 단위는 loader 이름만으로 전부 확정할 수 없어 공식 NASA 설명서를 우선하도록 표시했다. "
                "보조 데이터셋은 발표 main 9개와 달리 여러 후속 adapter가 존재하므로, 본문에는 발표에서 방어 가능한 공통 의미와 증거 등급을 중심으로 적었다."),
              P("최종 체크", "h2"),
              P("메인 9개와 보조 3개를 분리했고, X/Y/외삽좌표/유닛/스플릿/고장경계/대리정답 여부를 각각 표시했다. "
                "작성자 이름은 박진서로 통일했다.")]

    story.append(PageBreak())
    append_ppx_defense(story)
    append_architecture_evidence(story)
    append_novelty_argument(story)
    append_latest_and_tabpfn(story)
    append_comparator_algorithms(story)
    append_comparison_protocol(story)

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=17 * mm, bottomMargin=16 * mm,
                            title="PP-X 발표 데이터셋 완전 학습 가이드", author="박진서")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    OUT_MIRROR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT, OUT_MIRROR)
    print(OUT)
    print(OUT_MIRROR)


if __name__ == "__main__":
    build()
