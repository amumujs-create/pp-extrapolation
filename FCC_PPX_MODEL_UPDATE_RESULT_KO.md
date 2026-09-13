# FCC-PPX 예측구간 보조 실험 결과

> **메인 모델 승격 철회 (2026-09-13):** 이 실험은 coverage를 예측구간
> 포함률로 해석했다. 연구 목표의 coverage는 여러 seed와 dataset에서 안정적인
> 양의 R²를 유지하면서 pooled 성능이 높은 **predictive performance coverage**다.
> 따라서 FCC-PPX는 해당 목표를 개선한 새 모형으로 간주하지 않으며,
> uncertainty 보조 실험으로만 보존한다.

작성일: 2026-09-13  
모델명: **Falsification-Calibrated Credal Conformal PP-X (FCC-PPX)**

## 1. 목적

기존 PP-X의 validation-only 안전 라우팅을 유지하면서 다음 두 목표를 동시에
만족하는 모델 구조를 추가했다.

1. 단순 임계값 gate보다 강한 모형적 노벨티를 만든다.
2. 기존 support-scaled PP conformal interval의 empirical coverage를
   도메인별로 떨어뜨리지 않는다.

## 2. 구조 업데이트

직접 fallback을 `f0`, PP 후보를 `fPP`라고 한다. Validation physical-unit
log-regret의 one-sided 95% upper confidence bound로 PP를 falsification한다.

```text
certificate 통과: point = fPP, authority set A = {1}
certificate 실패: point = f0,  authority set A = [0, 1]
```

최종 예측구간은 다음 set-valued predictor다.

```text
{f0 + a(fPP - f0): a in A} + [-q s(x), q s(x)]
```

PP가 거절되면 점예측은 exact fallback이지만, 불확실성 표현에서는 `f0`와
`fPP` 사이를 credal set으로 유지한다. 모든 authority set에 `a=1`이 포함되기
때문에 새 구간은 기존 PP conformal interval을 row-wise로 포함한다. 따라서
동일한 평가 표본에서 coverage가 기존보다 낮아질 수 없다.

## 3. 기존 방식과 다른 점

- 기존 binary gate는 한 예측값만 고르고 버린 후보의 불확실성을 표현하지 않았다.
- FCC-PPX는 falsification 결과를 **set-valued residual authority**로 변환한다.
- 안전한 point routing과 coverage-preserving uncertainty set을 하나의 모델
  출력으로 결합한다.
- coverage 보존은 test tuning 결과가 아니라 interval nesting에서 나온다.

따라서 노벨티의 핵심은 새로운 scalar threshold가 아니라
`physical-unit falsification -> credal authority -> nested conformal set`의
결합 구조다.

## 4. 동결 실험

- 데이터: 이미 열린 common-backbone 12개 도메인
- nominal coverage: 90%
- falsification confidence: 95%
- bootstrap: physical-unit 20,000회
- conformal scale: 기존 PP-X와 같은 support-scaled residual scale
- threshold/width test tuning: 없음
- protocol: `protocols/FALSIFICATION_CREDAL_CONFORMAL_PPX_PROTOCOL.md`

## 5. 결과

| 지표 | 기존 PP-X | FCC-PPX |
|---|---:|---:|
| equal-domain coverage | 0.8706 | **0.9057** |
| pooled row coverage | 0.7267 | **0.7502** |
| 도메인별 coverage 비열화 | - | **12/12** |
| coverage 엄격 개선 | - | **4/12** |
| 최대 interval inclusion violation | - | **0.0** |
| mean domain width ratio | 1.000 | **1.165** |

점예측 결과:

- PP 승인 도메인: 4
- exact fallback 도메인: 8
- equal-domain mean unit log-RMSE improvement: **0.154252**
- domain bootstrap 95% CI: **[0.000854, 0.426799]**
- 개선 도메인: 4
- harm 도메인: **0**

## 6. 당시 판정과 수정 판정

예측구간 실험 기준은 통과했다.

- 기존 interval을 row-wise로 포함하므로 coverage 비열화가 구조적으로 차단됐다.
- 실제 12개 도메인에서도 coverage가 모두 비열화되지 않았고 4개에서 증가했다.
- point route는 4개 도메인에서 개선되고 harm 도메인은 없었다.
- coverage 개선 비용으로 평균 interval width가 약 **16.5%** 증가했다.

단, 점예측 이득 `0.154252` 자체는 기존 falsification certificate 정책과 같다.
이번 추가 기여는 점 RMSE의 추가 상승이 아니라, 해당 안전 라우팅을
coverage-preserving credal prediction model로 확장한 데 있다.

수정 판정은 **메인 PP-X 모델 기각**이다. 최종 PP-X의 seed/dataset R²
coverage와 pooled 성능을 개선하지 않았고, 더 약한 common-backbone 위에서
SUNWODA R²를 `0.865`에서 `-0.668`로 떨어뜨렸다. 논문에서는 point-performance
개선 또는 모형적 노벨티의 주 근거로 사용하지 않는다.

## 7. 주장 가능 범위와 한계

현재 주장 가능한 표현은 다음과 같다.

> FCC-PPX is a falsification-calibrated, set-valued residual-authority model
> whose prediction interval nests the original PP conformal interval while
> retaining exact point fallback under rejected prior corrections.

아직 주장하면 안 되는 내용:

- 미개봉 데이터에서 확인된 일반화
- arbitrary domain shift에서의 90% distribution-free coverage
- 최신 외부 SOTA 대비 우위
- 기존 certificate보다 높은 추가 point accuracy

특히 pooled row coverage는 0.7502로 nominal 0.90에 못 미친다. 이번 결과는
**현재 구간 대비 non-inferiority와 개선**이지, 모든 domain shift에서 nominal
coverage를 달성했다는 뜻이 아니다.

## 8. 재현 파일

- 모델: `src/pp_extrapolation/coverage_preserving_credal.py`
- 실험: `experiments/falsification_credal_conformal_ppx.py`
- 프로토콜: `protocols/FALSIFICATION_CREDAL_CONFORMAL_PPX_PROTOCOL.md`
- 테스트: `tests/test_coverage_preserving_credal.py`
- 결과: `results/falsification_credal_conformal_ppx_v2/results.json`
- 예측 배열: `results/falsification_credal_conformal_ppx_v2/predictions.npz`

검증 결과: 관련 테스트 **16 passed**.

## 9. 다음 승격 조건

모형적 노벨티를 논문의 확정 기여로 승격하려면 현재 코드를 동결한 뒤 미개봉
cohort에서 한 번 실행해야 한다. 확인할 핵심은 다음 세 가지다.

1. 도메인별 coverage non-inferiority 유지
2. point harm domain 0 유지
3. width inflation 대비 coverage gain의 효율 유지
