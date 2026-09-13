# OOF threshold-policy robustness 결과

작성자: 박진서  
상위 프로토콜: `protocols/PPX_THRESHOLD_TUNING_OOF_PROTOCOL.md` (공식 튜닝 규칙)  
실험: `experiments/ppx_oof_threshold_robustness.py`  
산출물: `results/ppx_oof_threshold_robustness_v1/results.json`

## 위치

이 결과는 **모델 구조 변경이 아니라 threshold 튜닝 근거**다.

- 실행: 고정 `τ=(2%,60%,1.10)`로 Val 승인
- 튜닝: 선언 격자 + development-unit OOF → 한 번 freeze


## 결론 (추가 가치)

**추가하는 편이 맞다.** 구조를 바꾸는 게 아니라,  
“왜 2%/60%/1.10인가?”에 대해 **test 튜닝이 아닌 development-unit OOF 안정성** 근거를 붙이는 audit다.

이번 실행에서는 선언된 27칸 grid가 OOF 지표상 **완전히 동일**했다.  
즉 frozen τ는 “유일하게 최적화된 점”이 아니라 **안정 영역 안의 operational freeze point**로 정당화된다.

## 범위

| | |
|---|---|
| LOO 대상 | validation unit ≥3인 **9개** setting |
| 제외 | sunwoda(2), femto(1), milling(1) — unit holdout 불가 |
| Test | threshold 선택에 **미사용** |

## Frozen τ=(2%, 60%, 1.10) OOF 요약 (9-setting macro)

| 지표 | 값 |
|---|---:|
| Route stability | **1.00** (LOO 결정 = full-val 결정) |
| OOF executor win fraction | **0.972** (승인 fold에서 held-out unit 개선 비율) |
| OOF mean policy regret | **−0.145** (음수 = 정책이 평균적으로 이득) |
| Max dataset worst policy regret | **0.033** |
| Full-val PP 승인 dataset 수 | 4/9 (`virkler, matr_batch2, nasa_battery, ncmapss`) |

설정별 (frozen):
- 거절 셋(hust/matr/rwth/mich/xjtu): stability 1.0, policy regret 0 (fallback 유지)
- 승인 셋: held-out unit이 대체로 이득; 최악은 `matr_batch2` worst regret ≈ +3.3%

## Grid 전체

Gain∈{1,2,5%} × UnitWin∈{50,60,70%} × Worst∈{1.05,1.10,1.20}  
→ **27/27 cells가 동일 OOF 지표** (route stability 1.0, 동일 accept pattern).

이전 retrospective sensitivity(FA/correct 동일)와 맞물려,  
이 archive에서는 근처 임계값이 **결정·OOF 안정성을 바꾸지 않는다.**

## 발표/논문에 쓸 논리

> 우리는 threshold를 test에서 최적화하지 않았습니다.  
> 미리 선언한 합리적 격자에서 validation-unit OOF로 route stability와 worst-case policy regret을 봤고,  
> 그 격자가 안정적임을 확인한 뒤 τ=(2%,60%,1.10)을 operational point로 동결했습니다.  
> 새 데이터셋에서는 τ를 다시 맞추지 않고, 그 셋의 validation으로 executor만 승인합니다.

배포 스토리:

$$ \text{Dev units} \xrightarrow{\text{OOF robustness}} \text{Freeze }\tau=(2\%,60\%,1.10) \xrightarrow{\text{no retune}} \text{Val approve} \to \text{Test untouched} $$

## 한계 (정직하게)

- 제외 3개 setting은 val unit이 적어 이 OOF audit에 못 들어감
- grid가 평평하다는 것은 “전 우주 무관”이 아니라 **현재 12-setting 후보가 임계 구간에 안 걸쳐 있음**을 뜻함
- 더 조밀한 격자나 train-unit OOF로 확장하면 일부 셀이 갈라질 수 있음
- 이건 policy robustness 보강이지, Algorithm 1 재동결이 아님
