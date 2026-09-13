# PP-X Approval Threshold Tuning Protocol (Frozen)

작성자: 박진서  
상태: **공식 threshold 튜닝 규칙** (once → freeze)  
관련 실험: `experiments/ppx_oof_threshold_robustness.py`  
관련 결과: `PPX_OOF_THRESHOLD_ROBUSTNESS_RESULTS_KO.md`  
보조 sensitivity: `PPX_THRESHOLD_SENSITIVITY_RESULTS_KO.md`

## 한 줄

모델 실행은 고정 τ를 쓰고, **τ 자체는 development-unit OOF robustness로 한 번 고른 뒤 동결**한다.  
새 데이터셋마다 τ를 다시 맞추지 않는다.

## 무엇이 모델이고, 무엇이 튜닝인가

| 층 | 내용 | 매 데이터셋마다? |
|---|---|---|
| 모델 구조 (Algorithm 1) | contract → prior-residual → Val에서 executor 승인 → freeze | 예 (τ는 고정값으로 적용) |
| Threshold 튜닝 | 선언된 격자에서 OOF로 안정 영역 확인 → τ freeze | **아니오 (한 번만)** |

즉:

> 구조는 고정 τ로 돌아가고,  
> τ를 정하는 방법만 OOF robustness 튜닝이다.

## 선언된 후보 격자 (사전 고정)

$$
\tau_g \in \{1\%,2\%,5\%},\quad
\tau_w \in \{50\%,60\%,70\%},\quad
\tau_r \in \{1.05,1.10,1.20\}
$$

이 격자 밖에서 test를 보고 τ를 고르는 행위는 금지한다.

## 튜닝 절차 (Development only)

1. **Development units**만 사용 (validation unit leave-one-out).
2. 각 격자 셀에 대해 계산:
   - OOF executor win fraction
   - OOF mean / worst policy regret
   - Route stability (LOO 결정 = full-val 결정)
3. **선택 기준 (성능 최대화 아님):**
   - route stability가 높고
   - worst-case policy regret이 작으며
   - 합리적 격자 안에서 보수적으로 유지되는 점
4. 동결점:

$$
\boxed{\tau^\star = (2\%,\; 60\%,\; 1.10)}
$$

5. 이후 모든 새 데이터셋:
   - τ 재조정 금지
   - 그 데이터셋 validation으로 executor만 승인
   - test untouched

## 현재 증거에서의 동결 근거

OOF audit (`results/ppx_oof_threshold_robustness_v1/`):

- 후보 격자 27칸이 OOF 지표상 동일 (route stability 1.0)
- frozen τ는 **안정 영역의 operational point**
- retrospective FA/correct sensitivity도 27칸 동일 (8/12, FA=2)

따라서 “2/60/1.10만 test에 맞춰 고른 것”이 아니라,  
**안정 영역에서 운영점으로 동결한 것**으로 기술한다.

## 금지 문구

- “OOF에서 test R²가 가장 높은 τ를 찾았다”
- “데이터셋마다 threshold를 다시 튜닝한다”
- “τ는 이론적으로 유도된 최적값이다”

## 허용 문구

- “사전 선언 격자에서 unit-holdout OOF 안정성으로 τ를 고르고 동결했다”
- “배포 시 τ는 고정이며, 데이터셋별 validation은 executor 승인에만 쓰인다”
