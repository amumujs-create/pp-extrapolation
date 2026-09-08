# PP 신규 비배터리 고호트 선정 — 2026-09-08

## 선정 결과

**주 고호트는 University of Ferrara의 self-aligning double-row bearing
run-to-failure 데이터로 정했다.** 현재 로컬 PP 연구에서 사용한 이력이 없고, 배터리와
다른 회전체 기계 도메인이며, PP가 요구하는 관측 가능한 prior가 가장 명확하다.

- 실제 베어링 6개를 각각 고장까지 운전한 물리 실험이다.
- 5분마다 5초의 방사방향 진동을 25.6 kHz로 기록했다.
- 모든 실험의 종료 조건이 진동 peak 20 g로 공개되어 있다.
- 같은 베어링 형식과 outer-race 손상이라는 공통 고장 메커니즘이 있어, 서로 전혀 다른
  수명 법칙이 섞이는 위험이 비교적 작다.
- E1--E3은 4 kN, E4는 3 kN, E5는 4.7 kN, E6은 5 kN이고 속도는 모두 40 Hz다.
  따라서 E5--E6을 시험에 두면 새로운 장비뿐 아니라 학습보다 큰 하중으로의 외삽도
  동시에 검증한다.

공개 메타데이터와 파일 크기는 확인했지만 실제 진동 궤적, 수명, E5--E6 결과는 보지
않았다. 상세 고정 규약은
`protocols/FERRARA_BEARING_EXTERNAL_COHORT_PROTOCOL.md`에 남겼다.

## 왜 PP가 잘될 가능성이 있는가

현재 최종 PP의 강점은 알려진 고장 경계까지의 margin을 예측식에 직접 보존하고, NN은
그 margin당 남은 시간인 quotient와 잔차만 학습한다는 점이다. Ferrara에서는 배터리
용량 80% 대신 `20 g - causal running peak`가 그 margin이 된다. 경계에서 예측 RUL이
0이 되는 조건이 구조적으로 유지되므로 direct-RUL MLP보다 late tail drift가 작을
가능성이 있다.

이 판단은 결과를 본 예측이 아니다. 베어링 결함은 말기에 급격히 진행될 수 있고, peak
진동은 잡음에 민감하며, 3--4 kN에서 배운 lifetime scale이 4.7--5 kN으로 선형 전이되지
않으면 PP도 실패할 수 있다. 그래서 raw peak의 causal running maximum, load context,
full-development refit을 사전에 고정하고 direct MLP와 같은 정보·예산으로 비교한다.

## 고정 평가 설계

| 구분 | 고정안 |
|---|---|
| train | E1, E2, E3 (4 kN) |
| validation | E4 (3 kN) |
| untouched trajectory test | E5, E6 (4.7, 5 kN) |
| RUL | 최초 20 g 도달까지 남은 5분 간격 수 × 5 |
| train state | 각 train 수명의 앞 70% |
| validation/test state | 각 unit의 뒤 30%, 실제 과거 이력은 제공 |
| PP prior | `margin = 20 g - 지금까지 관측된 최대 peak` |
| 주 비교 | 동일 입력·동일 seed·동일 refit의 direct-RUL MLP |
| 보조 비교 | Ridge, FT-Transformer, boundary/rate, elapsed-time baseline |
| 성공 | PP pooled R² > 0 이고 pooled RMSE가 matched MLP보다 낮음 |
| 정직한 한계 | test unit 2개이므로 통계적 보편 우월성은 주장하지 않음 |

## 후보 비교

| 후보 | 장점 | 결격 또는 부담 | 결정 |
|---|---|---|---|
| **Ferrara bearing** | 실제 RTF 6개, 20 g 경계, 동일 고장모드, 약 수 GB | 시험 unit 2개, peak 잡음 | **먼저 실행** |
| Paderborn bearing | 실제 RTF 17개, 결함을 인위적으로 심지 않음, 변동 하중·속도 | 전체 152 GB, 조건 변화와 health-index 난도가 큼 | Ferrara 뒤 강건성 시험 |
| PHM 2010 milling | wear 경계와 반복 cutter가 PP에 잘 맞음 | 공식 공개 label은 주로 train cutter 3개, test label 접근 문제 | 주 확증 제외 |
| NASA IMS bearing | 공식 NASA 자료, 실제 진동 | 실험 수와 실제 고장 bearing 수가 적고 고장모드가 다름 | 주 확증 제외 |
| IEEE PHM 2014 fuel cell | 비배터리 전기화학 열화, 공개 자료 | aging unit 2개뿐 | 파일럿만 가능 |
| PHM-AP 2025 cutter | 최근 공구마모, train set 6개 | unit당 wear label 6시점, 평가 label 비공개 | sparse 보조실험 |

Paderborn은 17개 실제 run-to-failure 베어링이라 성공하면 논문 근거가 Ferrara보다
강하다. 그러나 152 GB이고 하중·속도가 시간에 따라 변해, 지금 단계에서 실패했을 때
PP 구조 문제와 feature 추출 문제를 구분하기 어렵다. Ferrara에서 비배터리 경계 prior의
전이를 먼저 검증한 뒤, 같은 feature/executor를 고정해 Paderborn으로 옮기는 순서가
논문 스토리에도 더 명확하다.

## 데이터 출처

- Ferrara Part 1: https://doi.org/10.17632/htk59pp5wx.1
- Ferrara Part 2: https://doi.org/10.17632/zz8hpyx939.1
- Ferrara dataset article: https://doi.org/10.1016/j.dib.2024.110620
- Paderborn 17-bearing dataset: https://doi.org/10.5281/zenodo.10868257
- PHM Society 2010 tool-wear challenge:
  https://phmsociety.org/phm_competition/2010-phm-society-conference-data-challenge/
- NASA IMS bearing dataset: https://data.nasa.gov/dataset/ims-bearings
- IEEE PHM 2014 fuel-cell dataset: https://doi.org/10.17028/rd.lboro.3518141

