"""PPT 본문을 상세 학습 형태로 PDF story에 추가."""
from __future__ import annotations

from reportlab.lib.units import mm
from reportlab.platypus import PageBreak


def append_ppt_deck_detail(story, P, table, callout):
    """최신 PP-X_Research_Detailed_v2.pptx(41장)의 슬라이드별 상세 해설."""
    story += [
        P("PPT 본문 전체 상세 해설 (42장)", "h1"),
        callout(
            "<b>원본:</b> <font face='AppleGothic'>ppt/PP-X_Research_Detailed_v2.pptx</font><br/>"
            "아래는 발표 슬라이드를 그대로 읽는 것이 아니라, 각 장의 표·결론·주장 경계를 "
            "공부·Q&amp;A용으로 풀어 쓴 버전이다. 수치·판정·기여 항은 발표 고정본과 일치시킨다."
        ),
        P(
            "표지 제목은 PP-X / Validation-Approved Prior-Residual Extrapolation이다. "
            "전체 연구 지도는 Prior-Adaptive Extrapolation이며, 작성자 박진서, SPS Lab. "
            "‘CCMR v2.2’는 논문 메인이 아니라 trajectory-domain executor 사례다."
        ),
        PageBreak(),

        P("BA-01~02. 문제 · 연구 배경", "h1"),
        P(
            "학습 분포 안에서는 여러 함수가 비슷하게 맞아도, support 밖에서는 데이터가 답을 정하지 못한다. "
            "열화·RUL·피로균열은 실제 평가 시점이 학습 support 밖인 경우가 많아 보간보다 외삽이 중요하다. "
            "같은 데이터라도 prior가 바뀌면 밖으로 나가는 곡선이 달라진다. 맞지 않는 prior를 강제하면 외삽이 나빠진다."
        ),
        callout(
            "<b>출발점:</b> 맞는 답이 하나가 아니다. 밖을 지탱하는 것은 데이터가 아니라 지금 정당화되는 prior다. "
            "prior가 많을수록 좋은 것이 아니라, 그 수준과 신뢰성에 맞춰 외삽 전략을 고른다."
        ),
        PageBreak(),

        P("BA-03. 사전 조사 — 외삽은 가정에 의존한다", "h1"),
        table(
            ["계열", "대표", "가정", "한계"],
            [
                ["제약 없는 NN", "ReLU MLP", "마지막 선형 조각을 밖으로 연장", "tail 형태 통제 어려움"],
                ["형상 제약", "CMNN · monotone NN", "증가/감소 방향을 구조로 강제", "기울기·곡률까지 자동 결정되지 않음"],
                ["식·물리", "EQL · PINN · Physics-ML", "함수식·PDE를 구조/loss에 반영", "식·적용조건이 틀리면 위험"],
                ["통계적 외삽", "Engression · Xtrapolation · Progression", "noise 위치·미분경계·tail dependence", "가정이 약하면 band만 넓음"],
            ],
            [28 * mm, 45 * mm, 52 * mm, 49 * mm],
        ),
        P(
            "조사 결론: 범용 회귀기가 데이터만으로 support 밖 함수를 정해 주지 않는다. "
            "직접 인접 연구인 EV(2024)·LBO(2025)는 boundary-focused validation을 이미 제안했다. "
            "남은 질문은 이질적인 prior를 outcome-free contract로 제한하고, physical-unit 위험으로 "
            "executor와 exact fallback을 시험 전에 함께 고정할 수 있는가다."
        ),
        PageBreak(),

        P("BA-04. 연구 루트 — 식의 근거에 따라 두 경로", "h1"),
        table(
            ["경로", "조건", "내용", "발표에서의 위치"],
            [
                ["PP-X", "식이 없거나 약한 prior만 있음", "weak-prior route · prior-residual + evidence-selected executor", "이번 발표·논문 메인"],
                ["PAE", "적용 가능한 식이 정당화됨", "허용된 식 + 제한 NN; 이득 없으면 PP-X", "future program"],
                ["Assurance", "박사논문 통합", "믿기 / 보류 / 거절", "지도에만 표시"],
            ],
            [28 * mm, 48 * mm, 58 * mm, 40 * mm],
        ),
        PageBreak(),

        P("BA-05. 약한 prior란 무엇인가", "h1"),
        callout(
            "<b>정의:</b> 약한 prior = 닫힌 형태의 열화식·PDE를 쓰지 않고, "
            "경계·방향·저복잡도 tail·support·인과 이력처럼 관측·선언 가능한 구조만 외삽 경로로 쓰는 가정."
        ),
        table(
            ["요소", "가정", "설계 의도"],
            [
                ["경계", "EOL = 880 mAh", "용량→880이면 RUL→0으로 고정"],
                ["방향", "용량↓ → RUL↓", "뒤집힌 수명 예측을 막음"],
                ["저복잡도 tail", "affine / quotient 기본 경로", "밖을 NN이 마음대로 안 그림"],
                ["support·인과", "train health 1.0~0.8만 봄 · 현재까지 관측만 입력", "외삽 보정 권한을 제한"],
            ],
            [32 * mm, 70 * mm, 72 * mm],
        ),
        P(
            "약한 prior 예시: 용량이 줄면 남은 수명도 줄고 880 mAh에서 끝난다. "
            "‘정확히 이 3차함수로 열화한다’까지는 가정하지 않는다. "
            "PAE의 강한 prior는 검증된 식 자체다. PP-X는 식 없이 쓸 최소 구조만 두고, 근거가 없으면 거절한다."
        ),
        PageBreak(),

        P("BA-06. 연구 기여 — execution architecture (PPT 고정 3항)", "h1"),
        table(
            ["기여", "내용"],
            [
                [
                    "C1 Contract-conditioned model",
                    "경계·history·support·causal로 허용 prior를 선언하고, frozen prior 주변의 bounded causal residual만 학습",
                ],
                [
                    "C2 Validation-approved execution",
                    "bound·dual-scale·transport·history를 전역으로 켜지 않고, validation이 지지한 executor만 실행; 반례에서는 거절/fallback",
                ],
                [
                    "C3 Equal-coverage evidence",
                    "unit bootstrap·paired 검정, cross-domain stability, Selective Regression 동일 coverage, prospective DS03까지 함께 검증",
                ],
            ],
            [58 * mm, 116 * mm],
        ),
        callout(
            "<b>포지션:</b> PP-X is a contract-conditioned, validation-approved prior-residual "
            "execution architecture, not a universal neural predictor.<br/>"
            "<b>주장 범위:</b> 평균 SOTA가 아니라, 동일 coverage에서 더 안전한 예측 조건 · "
            "근거 없는 prior 거절 · unrestricted residual·generic selective regression보다 안전한 조건부 실행.<br/>"
            "<b>실증:</b> equal-budget 8/9 · domain MAD Engression p=.0391 · Selective equal-coverage 5/5 · "
            "DS03 prior 거절 PASS / 정확도 우월 FAIL."
        ),
        P(
            "예전 4항(C1 Typed contract / C2 Prior-residual core / C3 Unit-evidence approval / C4 Frozen execution)은 "
            "위 3항으로 압축했다. Typed contract와 prior-residual core는 C1, unit-evidence approval과 frozen "
            "execution은 C2, 동일 coverage·unit 통계·prospective는 C3에 해당한다."
        ),
        PageBreak(),

        P("BA-06b. C1을 숫자로 보면", "h1"),
        table(
            ["비교", "구조"],
            [
                ["일반 NN", "X → NN이 전부 결정 → RUL. 밖에서도 학습 패턴을 마음대로 이어 그릴 수 있다."],
                [
                    "PP-X",
                    "경계·history·support·causal → 약한 Prior(예: ŷ_prior=100) → NN 수정 −20~+20만 허용 → ŷ=108",
                ],
            ],
            [32 * mm, 142 * mm],
        ),
        P(
            "왜 bound가 필요한가: Prior=100인데 residual=−300이면 prior를 둔 의미가 없다. "
            "실제 식은 ŷ=D[y_prior + b tanh(r/b)] 로, 수정량이 대략 −b~+b에 갇힌다."
        ),
        callout(
            "<b>발표용 한 줄:</b> 외삽 구간에서 신경망이 마음대로 예측하게 하지 않고, "
            "현재 데이터에서 정당화할 수 있는 기본 추세를 먼저 정한 뒤, 신경망은 그 추세에서 "
            "제한된 범위만 수정한다."
        ),
        PageBreak(),

        P("BA-07. 그래서 외삽에서 무엇이 좋아졌는가", "h1"),
        table(
            ["효과", "메커니즘", "근거"],
            [
                ["평균 예측오차", "validation이 지지한 executor만 선택", "equal-budget 8/9 · GM-RMSE 33.8% 감소"],
                ["전역 적용 붕괴 방지", "같은 bound·scale을 항상 켜지 않음", "MICH fixed bound ΔR² −.291 · RWTH dual-scale −.037"],
                ["유닛 단위 안정성", "unit wins·worst-unit ratio를 승인 조건에 포함", "HUST 15/16 · MICH 7/8"],
                ["잘못된 승인 감소", "unit-gain CI로 약한 prior 승인 걸러냄", "false accept 2→0 · 정확도 .667→.833"],
            ],
            [40 * mm, 62 * mm, 72 * mm],
        ),
        callout(
            "범위: 위 수치는 retrospective evidence다. 보지 않은 미래 cohort에서 같은 실패 감소를 "
            "보장한다는 주장은 하지 않는다."
        ),
        PageBreak(),

        P("BA-08~09. 방법 · Algorithm 1", "h1"),
        table(
            ["단계", "최종 PP-X가 하는 일"],
            [
                ["(a) contract", "경계·unit·causal X·허용 prior·fallback 선언"],
                ["(b) core", "동결 affine/quotient tail + bounded nonlinear residual"],
                ["(c) executor", "bound·history·transport 중 validation이 지지한 것만 실행"],
            ],
            [38 * mm, 136 * mm],
        ),
        table(
            ["승인 규칙 (val/source만)", "조건"],
            [
                ["Prior admissibility", "known_boundary → BQ, 아니면 affine (Final)"],
                ["Executor gain", "validation RMSE 상대 2% 이상 개선"],
                ["Unit risk", "unit wins ≥60% · worst ratio ≤1.10"],
                ["거절", "Executor Val FAIL → 사전 fallback. Prior OFF는 v1_declared만"],
            ],
            [55 * mm, 119 * mm],
        ),
        P(
            "Algorithm 1 순서 (Final): ① Typed contract → ② Prior "
            "(known_boundary면 BQ, 아니면 affine; OOF/group/mode 미실행) → "
            "③ Executor approval (gain≥2%, unit wins≥60%, worst≤1.10) → "
            "④ Frozen output (승인 executor 1개 또는 Val FAIL fallback; test에서 route 불변)."
        ),
        callout(
            "논문 메인 = PP-X 선택 알고리즘. CCMR은 trajectory risk-aware executor 사례. "
            "12개 과거 route가 하나의 동일 NN이라는 뜻이 아니다. "
            "동결: protocols/PPX_PAPER_METHOD_V1_FROZEN_PROTOCOL.md · paper_ppx.py "
            "(test outcome 인자 없음)."
        ),
        PageBreak(),

        P("BA-10~12. 외삽 구간 · hull · 프로토콜", "h1"),
        table(
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
            [18 * mm, 156 * mm],
        ),
        P(
            "1D ordered-coordinate hull-out을 주 평가로 쓴다. 전체 feature-space convex hull 밖이라고 "
            "말하지 않는다. 엔진은 먼 수명 외삽보다 운전조건(regime) 외삽에 가깝다."
        ),
        PageBreak(),

        P("BA-13~18. 주 결과 · Ablation", "h1"),
        P(
            "메인 9-setting은 1D 열화좌표 기준의 엄격한 외삽 retrospective development 결과다. "
            "Ablation 요지: residual은 세 배터리에서 이득, fixed bound와 full history는 MICH에서 마이너스, "
            "dual-scale은 MICH에서만 복구, transport는 HUST·MATRb2에서 주효과. "
            "따라서 모듈을 쌓지 않고 근거 있는 executor만 켠다."
        ),
        PageBreak(),

        P("BA-19~24. 비교 · 통계 · 도메인 안정성 · Selective equal-coverage", "h1"),
        table(
            ["증거", "요지"],
            [
                ["Equal-budget 8/9", "30후보 동일 예산에서 최강 비교군 대비 8/9, sign p=.0391"],
                ["77-unit", "계층 bootstrap GM-RMSE 감소 33.8% [15.8%, 48.6%]"],
                ["Cross-domain stability", "PP-X domain MAD .061; Engression .280, exact p=.0391"],
                ["Selective Regression", "Alg.1이 4/9 전부 거절; 동일 achieved coverage에서 PP-X 5/5"],
            ],
            [48 * mm, 126 * mm],
        ),
        callout(
            "Selective Regression은 support가 희박하면 예측을 버린다. PP-X는 지지된 prior-residual 경로로 "
            "외삽 coverage를 유지하면서, 같은 coverage에서도 accepted risk를 낮췄다."
        ),
        PageBreak(),

        P("BA-25~27. Prospective · 실패 분석 · 주장 경계", "h1"),
        P(
            "DS03: prior route 거절은 test-best PP-X route와 일치(PASS). "
            "Engression 대비 ΔR² −.0195로 predictive superiority는 FAIL. "
            "실패 유형은 prior 불일치만이 아니라 executor–shift 불일치, validation 전이 실패, "
            "fallback 표현력 부족, 짧은 tail 정보 부족으로 구분한다."
        ),
        callout(
            "<b>PPT 요약 문장:</b> METHOD Declare→Approve→Execute/Decline. "
            "CONTRIBUTION C1–C3. EVIDENCE equal-budget 8/9 · unit 33.8% · selective 5/5. "
            "LIMITATION DS03 route PASS · predictive superiority FAIL. "
            "SCOPE strict-tail · unit-disjoint · val-only · PAE는 future · universal SOTA 아님."
        ),
        PageBreak(),

        P("BA-마무리. PPT ↔ 본 PDF 읽는 순서", "h1"),
        table(
            ["PPT 구간 (42장)", "본 PDF에서 이어서 볼 곳"],
            [
                ["1–3 문제·사전조사", "부록 AA–AD 노벨티·서사, 문헌 매핑"],
                ["4–5 연구 루트·약한 prior", "BA-04~05, PPX_PAPER_NARRATIVE_AND_EVIDENCE_REGISTRY_KO.md"],
                ["6–7 기여·C1 숫자 예시", "BA-06~06b, PPX_PAPER_CONTRIBUTIONS_KO.md"],
                ["8–9 효용·방법", "BA-07~09"],
                ["10–13 프로토콜", "부록 A–E, Algorithm 1 동결 protocol"],
                ["14–19 결과·ablation", "부록 F–H, U–Y"],
                ["20–25 비교·안정성·Selective", "BA-19~24, FULL_EQUAL / DOMAIN_STABILITY / SELECTIVE"],
                ["26–35 실패·CCMR·경계", "DS03 결과, CCMR mechanism evidence"],
                ["36–42 다음·PAE·보조", "future program · 데이터셋 장"],
            ],
            [58 * mm, 116 * mm],
        ),
    ]
