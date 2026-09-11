# CTBF 메인 contract benchmark 승격 감사

작성자: 박진서  
판정: **PP-X 보편 대체 승격 실패; Virkler 전용 challenger 유지**

## 범위

PP-X 논문의 9개 메인 setting 중 scalar ordered state, 사전 선언된 failure
boundary, time-to-boundary target이 모두 성립하는 네 setting만 outcome 확인 전에
포함했다.

- Sunwoda
- RWTH
- NASA PCoE battery
- Virkler fatigue crack

HUST는 terminal-slope proxy target, MICH·MATR 계열은 공통 health boundary가
아닌 life/end-of-record target, N-CMAPSS는 scalar ordered state와 선언된
boundary가 없어 제외했다. 제외는 성능을 보기 전에 결정했다.

## 동일예산 결과

CTBF는 30개 validation candidate를 탐색하고 선택 설정을 seeds 42–46으로
재학습했다. 비교군은 기존 동일 split의 frozen PP-X와 각 30-candidate
benchmark artifact다.

| Setting | CTBF R² | PP-X R² | 최강 30c R² | direct MLP 30c R² |
|---|---:|---:|---:|---:|
| Sunwoda | 0.795 | **0.939** | 0.838 | -2.140 |
| RWTH | 0.792 | **0.878** | 0.732 | -0.261 |
| Virkler | **0.966** | 0.888 | 0.890 | 0.724 |
| NASA | 0.487 | **0.584** | 0.583 | 0.293 |

CTBF는 direct MLP보다 네 setting 모두에서 높은 pooled R²를 보였지만,
PP-X 대비 승리는 Virkler 1/4뿐이었다.

## PP-X 대비 physical-unit 통계

### Sunwoda

- mean unit log-RMSE improvement: -0.597
- bootstrap 95% CI: [-1.049, -0.105]
- unit wins: 2/9
- worst-unit CTBF/PP-X RMSE ratio: 5.595
- sign-flip p=0.0547, BH q=0.0938

### RWTH

- mean unit log-RMSE improvement: -0.366
- bootstrap 95% CI: [-0.588, -0.056]
- unit wins: 1/8
- worst-unit ratio: 2.114
- sign-flip p=0.0703, BH q=0.0938

### Virkler

- mean unit log-RMSE improvement: 1.229
- bootstrap 95% CI: [0.690, 1.780]
- unit wins: 9/10
- worst-unit ratio: 1.582
- sign-flip p=0.00391, BH q=0.0156

### NASA

- mean unit log-RMSE improvement: -0.169
- bootstrap 95% CI: [-0.409, -0.003]
- unit wins: 1/4
- worst-unit ratio: 1.693
- sign-flip p=0.25, BH q=0.25

Dataset-level CTBF 승리는 1/4이며 양측 exact sign test는 p=0.625다.

## 구조 ablation

Local observed-velocity loss의 효과는 일관되지 않았다.

- Sunwoda: 사용 시 R² 0.795, 제거 시 0.782로 소폭 도움
- RWTH: 사용 시 0.792, 제거 시 0.830으로 악화 요인
- Virkler: 사용 시 0.966, 제거 시 0.968로 차이 없음
- NASA: rate 입력이 없어 두 arm이 동일

따라서 local-rate supervision을 CTBF의 필수 구조적 기여로 주장할 수 없다.
반면 boundary-coordinate와 positive inverse-velocity integral은 direct MLP
대비 네 setting 모두에서 개선되어 별도 구조 신호가 있다.

## 사전 승격조건 판정

- PP-X 대비 최소 3/4 승리: **실패, 1/4**
- 평균 dataset effect 양수 및 최소 3/4 positive: **실패**
- 최대 pooled RMSE harm 10% 이하: **실패, 최대 84.1%**
- unit-bootstrap CI가 CTBF를 지지하는 setting 2개 이상: **실패, 1개**
- boundary exactness: **통과**
- local-rate loss 비의존성: **통과**

최종 판정은 `promote_over_ppx = false`다.

## 논문 해석

CTBF가 PP-X의 보편 후계모델이라는 주장은 기각한다. 그러나 다음 두 결과는
보존할 가치가 있다.

1. 명시적 boundary contract가 있는 네 setting 모두에서 direct MLP보다 높은
   pooled R²를 기록했다.
2. Virkler에서 PP-X와 최강 30-candidate baseline을 크게 이겼고,
   physical-unit 통계도 BH 보정 후 유의했다.

따라서 CTBF는 universal replacement가 아니라 **mechanism-aligned
time-to-boundary executor** 후보로 남긴다. 다음 실험은 전체 데이터에
강제하는 것이 아니라 validation에서 flow consistency가 확인되는 경우에만
승인하는 정책이어야 한다.

## 재현

```bash
PYTHONPATH=src /opt/anaconda3/bin/python \
  experiments/ctbf_main_contract_benchmark.py
```

원시 결과:

- `results/ctbf_main_contract_benchmark/results.json`
- `results/ctbf_main_contract_benchmark/*_predictions.npz`
