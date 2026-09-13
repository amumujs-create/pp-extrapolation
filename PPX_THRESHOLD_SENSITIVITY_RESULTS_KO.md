# PP-X approval threshold sensitivity 결과

작성자: 박진서  
프로토콜: `protocols/PPX_THRESHOLD_SENSITIVITY_PROTOCOL.md`  
실험: `experiments/ppx_threshold_sensitivity.py`  
산출물: `results/ppx_threshold_sensitivity_v1/results.json`

## 한 줄 결론

**Gain×UnitWin×WorstRatio 27칸 전부가 frozen(2%/60%/1.10)과 동일한 승인 집합**이다.  
이 12-setting archive에서는 근처 임계값을 바꿔도 route 선택·FA/FR이 흔들리지 않는다.

## Baseline

| 정책 | correct | FA | FR | accepted |
|---|---:|---:|---:|---:|
| Validation RMSE only | 7/12 | **4** | 1 | 9 |
| Frozen 2% / 60% / 1.10 | **8/12** | **2** | 2 | 6 |

Frozen 승인: `virkler, matr_batch2, femto, milling, nasa_battery, ncmapss`  
Unit-risk가 RMSE-only에서 걸러낸 예: 승률·worst가 기준을 못 넘는 후보들 (예: sunwoda win=0.50·worst≈3.01, hust win≈0.44).

## Full grid (27 cells)

| Gain | UnitWin | WorstRatio | correct | FA | FR |
|---|---|---|---:|---:|---:|
| {1,2,5}% | {50,60,70}% | {1.05,1.10,1.20} | **전부 8** | **전부 2** | **전부 2** |

안정성 요약:
- `false_accepts_range = [2, 2]`
- `correct_range = [8, 8]`
- frozen accept set과 일치하는 cell: **27/27**

## 왜 안 흔들리나 (솔직한 해석)

이 archive의 후보는 임계값 사이에 아슬아슬하게 걸쳐 있지 않다.

- 통과군: win≈1.0, worst≲1.03, gain≫5% (또는 ncmapss gain≈14%)
- 탈락군: win≤0.50이거나 worst≫1.20이거나 gain 음수

그래서 1↔5%, 50↔70%, 1.05↔1.20 구간에서는 **같은 결정**이 나온다.  
“전 우주에서 임계값이 무관하다”가 아니라, **현재 12-setting retrospective 증거에서는 operational neighborhood가 안정적**이라는 뜻이다.

## 발표 Q&A

> 왜 2% / 60% / 1.10인가? 1·5%나 50·70%, 1.05·1.20이면?

**답:**
이론적 최적값이 아니라 frozen protocol의 보수적 operational threshold입니다.  
같은 12-setting archive에서 Gain∈{1,2,5%}, UnitWin∈{50,60,70%}, WorstRatio∈{1.05,1.10,1.20} sensitivity를 돌리면 **27칸 모두 8/12·FA=2로 동일**합니다.  
즉 이 근처에서 “2/60/1.10만 골라서 결과를 만든 것”으로 보이지 않습니다.  
반면 validation RMSE만 쓰면 FA=4로 더 위험합니다.

## 한계

- 더 조밀한 격자(예: win 45%, worst 1.5)까지 가면 일부 설정이 뒤집힐 수 있다.
- 이건 retrospective common-backbone audit이며 prospective 보장은 아니다.
- CI gate(≥0)를 켜면 별도 정책이 되어 FA를 더 줄일 수 있으나, 이번 grid의 대상이 아니다.
