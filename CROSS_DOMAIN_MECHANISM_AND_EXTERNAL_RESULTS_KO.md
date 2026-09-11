# PP-X 메커니즘·불확실성·외부 코호트 결과

> **Canonical paper pointer:** 논문 메인은 PP-X다. 아래 12-domain 공통
> affine-tail `PP backbone`과 외부 코호트 수치는 역사적 mechanism/development
> evidence이며 PP-X paper-selected route 전체와 같지 않다. 최신 paper 주장과
> DS03 prospective 판정은 `PPX_TOP_JOURNAL_VALIDATION_PACKAGE_KO.md`를 우선한다.
> 독립 prospective predictive superiority는 아직 확보되지 않았다.

## 무엇을 새로 완료했나

1. 기존 12개 도메인에서 같은 입력, width 32, optimizer, seeds 42–46, validation checkpoint를 사용해 plain MLP와 공통 affine-tail PP backbone을 다시 비교했다.
2. test outcome을 사용하지 않고 normalized horizon, train-unit degradation-law heterogeneity, degradation SNR, curvature gain을 계산했다.
3. 각 physical unit을 표본으로 paired RMSE bootstrap과 sign-flip 검정을 수행했다.
4. validation physical unit을 calibration block으로 사용한 support-scaled 90% conformal interval을 평가했다.
5. NASA/UCF Randomized and Recommissioned Battery Dataset을 protocol과 코드 hash 고정 후 다운로드해 one-shot 외부 평가했다.

## 12-domain matched backbone 결과

| 데이터셋 | Plain MLP ens. R² | PP backbone ens. R² | ΔR² | PP 절대 성공 |
|---|---:|---:|---:|---|
| HUST | 0.818 | 0.773 | −0.046 | 예, MLP보다 낮음 |
| Virkler | −0.459 | **0.888** | **+1.347** | 예 |
| MATR2019 | −0.184 | −0.666 | −0.482 | 아니오 |
| Sunwoda | −0.668 | **0.865** | **+1.533** | 예 |
| RWTH | **0.634** | 0.507 | −0.127 | 예, MLP보다 낮음 |
| MICH | −0.273 | −1.522 | −1.249 | 아니오 |
| MATR batch 2 | 0.447 | **0.465** | +0.018 | 예, 차이 작음 |
| XJTU | −1.565 | −1.308 | +0.256 | 아니오 |
| FEMTO | −0.902 | −1.378 | −0.476 | 아니오 |
| NASA milling | −0.376 | −4.826 | −4.450 | 아니오 |
| NASA battery | 0.272 | **0.495** | +0.223 | 예 |
| N-CMAPSS | **0.813** | 0.808 | −0.005 | 예, 사실상 동률 |

이 표는 PP-X paper-selected route가 아니라 **legacy 공통 PP backbone의 효용**을
분리한 ablation이다. PP-X에서는 typed contract와 validation approval를 거쳐
HUST transport, BQ/dual-scale, multiscale executor가 선택적으로 추가된다.

## 요청한 네 변수는 승패를 설명했는가

단변량 Spearman 상관은 모두 유의하지 않았다.

| 변수 | PP−MLP ΔR²와 Spearman ρ | p |
|---|---:|---:|
| Normalized horizon | −0.028 | 0.931 |
| Trajectory heterogeneity | −0.336 | 0.286 |
| Degradation SNR | −0.203 | 0.527 |
| Curvature gain | −0.063 | 0.846 |

네 변수를 함께 사용한 nested leave-one-domain-out ridge도 예측과 실제 gain의 Spearman이 `−0.916`으로 방향이 뒤집혔고 permutation p는 `0.060`이었다. One-variable stump gate의 balanced accuracy는 `0.10`, permutation p는 `0.945`였다. 즉 요청한 네 변수만으로 만든 사전 gate는 현재 데이터에서 **작동하지 않는다**.

이 실패는 중요한 이론적 결론이다. PP applicability는 정적 데이터셋 통계 하나보다 다음 두 단계를 구분해야 한다.

- backbone certificate: train에서 안정적인 affine tail과 충분한 unit support가 있는가
- executor certificate: validation과 test의 extrapolation ray/regime가 전달 가능한가

현재 네 descriptor는 첫 단계를 일부 나타내지만 두 번째인 relationship transportability를 관측하지 못한다. 그러므로 이를 성공한 사전 gate라고 주장하지 않는다.

## 탐색적으로 나타난 적용 영역

물질적으로 성공한 backbone을 `PP R²>0`이면서 `PP−MLP ΔR²>0.05`로 정의하면 Virkler, Sunwoda, NASA battery 세 도메인이다. 세 도메인은 모두 다음 영역에 있었다.

- median support distance > 0.5
- train-unit degradation-law heterogeneity < 0.05
- descriptor를 계산할 train unit ≥ 4

하지만 이 경계는 결과를 본 뒤 얻은 **retrospective certificate**다. NASA/UCF 외부 결과 이후 확정됐으므로 독립 검증으로 세지 않는다. 다음 외부 cohort에 그대로 고정할 후보 규칙이다.

## 새 외부 배터리 코호트 3개

NASA/UCF, CALCE, HNEI를 기존 PP 결과와 분리된 외부 자료로 추가했다. 다운로드 전에 데이터 적격성, ID 정렬 split, train 70%/held-out tail 30%, 5개 seed와 pooled R²를 protocol 파일에 고정했다. 감사 과정에서 각 test cell의 최종 수명을 시간 입력의 분모로 사용한 초기 구현을 발견해 폐기했다. 아래 수치는 모두 **train cell의 최대 관측 시간만으로 정규화해 다시 실행한 누출 수정 결과**다.

| 외부 코호트 | Test units | Plain MLP pooled R² | PP pooled R² | ΔR² | Unit RMSE wins | Exact sign-flip p |
|---|---:|---:|---:|---:|---:|---:|
| NASA/UCF | 3 | −0.785 | −3.212 | −2.427 | 0/3 | 0.250 |
| CALCE | 4 | **0.188** | −0.011 | −0.199 | 2/4 | 0.625 |
| HNEI | 4 | 0.961 | **0.994** | **+0.033** | 2/4 | 0.500 |

HNEI는 새 데이터에서 PP가 general MLP보다 높은 첫 외부 성공 사례다. 그러나 unit-mean RMSE reduction은 `+7.692 cycles`, bootstrap 95% CI `[−4.190, 20.250]`, p=`0.5`라서 **작은 4-unit 표본에서 통계적 우월성이 확정된 것은 아니다**. Pooled R² 향상과 unit별 승패를 분리해 보고해야 한다.

NASA/UCF에서는 PP가 0/3 unit에서 이겼고 mean RMSE reduction은 `−1.500 days` (95% CI `[−2.402, −0.491]`)였다. CALCE에서도 평균 차이는 `−8.856 cycles` (95% CI `[−38.042, 20.331]`)로 PP 열세였다. 따라서 배터리라는 도메인 이름만으로 PP applicability를 결정할 수 없다.

Validation pooled RMSE가 낮은 경로를 선택하는 단순 routing diagnostic도 세 코호트 모두 PP를 골랐다. HNEI에서는 맞았지만 NASA/UCF와 CALCE에서는 틀렸다. 이 분석은 외부 test 확인 후 추가한 retrospective diagnostic이며, 성공한 사전 gate의 증거로 사용하지 않는다. 핵심 미해결 변수는 validation unit과 test unit 사이의 **열화법칙 전달 가능성**이다.

### 실패 코호트의 후속 구조 개선

HNEI locked 결과는 건드리지 않고 NASA/UCF와 CALCE에 safe-continuation PP를 개발했다. 이 구조는 affine prior와 direct neural path를 하나의 네트워크 안에서 `λ`로 연결하며, `λ=0`은 동일 초기화의 plain NN이다. 양의 `λ∈{0.02,0.05,0.10,0.20}` 중 validation RMSE로 선택했다. NASA/UCF는 `λ=0.02`에서 R²가 `−0.785→−0.391`, CALCE는 `λ=0.20`에서 `0.188→0.218`로 개선됐다. NASA/UCF는 여전히 절대 R²가 음수다. 또한 두 test를 이미 본 뒤 수행한 개선이므로 독립 외부 확증이 아니라 development evidence다. 상세 결과는 `EXTERNAL_FAILURE_PP_RECOVERY_RESULTS_KO.md`에 분리했다.

## 불확실성 구간

90% support-scaled block-conformal interval을 12개 기존 도메인과 외부 3개 코호트에 계산했다. 외부 empirical coverage는 NASA/UCF `0.857`, CALCE `0.927`, HNEI `0.394`였다. NASA/UCF의 calibration unit은 1개, CALCE와 HNEI는 각각 2개뿐이다. 이 표본수로는 90% unit-level finite-sample 보장을 주장할 수 없으며, CALCE의 높은 coverage도 평균 폭이 약 416 cycles로 지나치게 넓다.

결론은 “PP가 언제 틀릴지 완전히 안다”가 아니다. **Calibration unit이 충분하고 validation–test regime가 교환 가능할 때만 coverage가 유지된다.** 논문에서는 calibration unit 수와 shell별 coverage를 반드시 함께 보고해야 한다.

## 논문 노벨티에 반영할 이론

현재 결과가 PP-X에 대해 지지하는 구조는
`typed contract → validation approval → executor/fallback → prediction`이다.

1. Geometric certificate: 실제 train-support 밖의 degradation-range extrapolation인지 확인한다.
2. Law-sharing certificate: train unit 간 열화법칙 이질성과 calibration unit 수가 허용 범위인지 확인한다.
3. Executor selection: 통과한 prior만 affine/BQ/history/transport executor로 조립한다. certificate가 부족하면 plain NN 또는 abstention으로 보낸다.

새 외부 결과는 이 구조의 필요성을 강화하지만, 고정 threshold의 외부 성공을 입증하지는 않는다. 실제로 HNEI는 사전 후보 certificate가 거부했음에도 PP가 성공한 false negative였고, validation routing은 두 실패 코호트를 승인했다. 다음 연구 단계는 정적 descriptor threshold가 아니라 train/validation에서 추정한 degradation-law posterior와 held-out unit의 짧은 history 사이의 compatibility score를 학습하는 것이다. 이 score는 새 확증 코호트에 들어가기 전에 고정해야 한다.

## 재현 파일

- 전체 matched 실험: `experiments/cross_domain_mechanism_study.py`
- meta-analysis: `experiments/meta_applicability_analysis.py`
- 외부 평가: `experiments/nasa_alt_external_eval.py`, `experiments/calce_external_eval.py`, `experiments/hnei_external_eval.py`
- 외부 사전 protocol: `NASA_ALT_EXTERNAL_LOCKED_PROTOCOL.md`, `CALCE_EXTERNAL_LOCKED_PROTOCOL.md`, `HNEI_EXTERNAL_LOCKED_PROTOCOL.md`
- 전체 원시 결과: `results/cross_domain_mechanism_v1/results.json`
- meta 결과: `results/meta_applicability_v1/results.json`
- 외부 결과: `results/nasa_alt_external_locked_v1/results.json`, `results/calce_external_locked_v1/results.json`, `results/hnei_external_locked_v1/results.json`
