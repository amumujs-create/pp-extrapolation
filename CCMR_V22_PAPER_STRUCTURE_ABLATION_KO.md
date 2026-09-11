# CCMR v2.2 논문용 구조·성능 Ablation

작성: 박진서  
스크립트: `experiments/ccmr_v22_paper_structure_ablation.py`  
산출: `results/ccmr_v22_paper_structure_ablation/results.json`  
범위: 개발 5도메인만 (Concrete, LG_M50T, SIT_LFP, RADAR_NMC, Luminosity)  
금지: Alloy A / MultiStage 로딩·재튜닝

## 질문

배포 CCMR v2.2의 **어느 구조가 필요한가?**

1. **안전**: false accept = 0, max raw regret ≤ 2%, exact fallback
2. **성능**: 문서화된 v2.2 이득(특히 SIT small-cohort, RADAR stable) 유지

## Full v2.2 기준선

| Domain | Route | Pooled ΔRMSE | Max regret | Coverage |
|---|---|---:|---:|---:|
| Concrete | exact_fallback | 0 | 0 | 0 |
| LG_M50T | cautious_causal | 0 | 0 | 0 |
| SIT_LFP | **small_crossfit_bank** | **+0.70%** | 0 | 0.468 |
| RADAR_NMC | **stable_bank** | **+5.91%** | 0 | 0.578 |
| Luminosity | exact_fallback | 0 | 0 | 0 |

- false accept **0**, max regret **0**, exact fallback OK

## Variant 요약

| Variant | 의미 | FA | Max regret | Mean ΔRMSE | SIT ΔRMSE | 안전 깨짐 | 핵심 |
|---|---|---:|---:|---:|---:|:---:|---|
| **full_v22** | 배포 기준 | 0 | 0% | +1.32% | +0.70% | 아니오 | 기준 |
| **v20_no_small_crossfit** | small route 제거 | 0 | 0% | +1.18% | **0%** | 아니오 | SIT 이득 소실 |
| **always_exact_fallback** | 항상 persistence | 0 | 0% | 0% | 0% | 아니오 | 하한 |
| **always_raw_bank** | route 무시·bank 강제 | 0 | **30.3%** | +2.22% | +0.70% | **예** | Luminosity/Concrete regret 폭발 |
| **cautious_only** | bank route 금지 | 0 | 0% | 0% | 0% | 아니오 | RADAR stable 이득 소실 |
| **no_risk_caps** | val risk cap 해제 | 0 | 0% | +0.14% | +0.70% | 아니오 | RADAR가 fallback으로 퇴행 |
| **no_residual_bound** | residual clip 제거 | 0 | 0% | +1.32% | +0.70% | 아니오 | 이 5도메인에선 차이 없음 |
| **relaxed_consensus** | consensus 완화 | 0 | 0% | +1.32% | +0.70% | 아니오 | 이 5도메인에선 차이 없음 |
| **ungated_bank** | consensus∧support 끔 | 0 | 0% | +19.0% | +27.0% | 아니오 | 평균은 오르지만 보수 gate와 다른 정책 |
| **only_linear_rate** | 단일 expert | 0 | **2.23%** | +0.48% | +0.85% | **예** | RADAR max regret > 2% |
| only_damped_acceleration | 단일 | 0 | 0% | +0.97% | +0.75% | 아니오 | RADAR 이득 축소 |
| only_monotone_hinge | 단일 | 0 | 0% | +1.13% | +0.81% | 아니오 | full 대비 약화 |
| only_nonlinear_residual | 단일 | 0 | 0% | +0.78% | +0.70% | 아니오 | full 대비 약화 |
| drop_* (각 1개 제거) | leave-one-expert | 0 | 0% | ≈full | 유지 | 아니오 | 단일 제거는 대체로 견딤 |

## 논문에서 주장 가능한 증거

### 1. Route는 필수 안전장치다
`always_raw_bank`는 평균 RMSE는 좋아 보이지만:
- Luminosity max regret **30.3%**
- Concrete max regret **5.7%**
→ **성능을 위해 bank를 상시 켜면 안 된다.** fallback/route가 위험을 막는다.

### 2. `small_crossfit_bank`는 SIT 성능에 필요하다
v2.0 route만 쓰면 SIT가 `exact_fallback`으로 돌아가고 pooled 이득 **+0.70% → 0**.  
RADAR는 유지. 즉 v2.2의 문서화된 개발 이득은 **소표본 route**에서 온다.

### 3. Bank route(stable/small)는 cautious만으로 대체 불가
`cautious_only`는 SIT·RADAR 이득을 모두 잃는다 (mean ΔRMSE 0).

### 4. 단일 expert는 부족하다
`only_linear_rate`는 RADAR에서 max regret **2.23% > 2% cap**으로 안전 깨짐.  
다른 단일 expert도 RADAR pooled 이득을 full(+5.91%)보다 낮춘다.  
반대로 expert 하나를 빼는 `drop_*`는 대체로 full과 비슷 → **은행(ensemble selection)** 이 단일 고정 expert보다 낫다.

### 5. Risk cap은 RADAR 승인 경로에 기여한다
`no_risk_caps`는 false accept/regret을 안 키우지만 RADAR가 `exact_fallback`으로 떨어져 pooled 이득을 잃는다.  
“안전 폭주”보다 **올바른 승인/거절**에 가깝다.

## 주장하면 안 되는 것

| 문장 | 이유 |
|---|---|
| consensus/support gate가 이 5도메인에서 필수 안전장치다 | `ungated_bank`/`relaxed_consensus`는 여기서 FA·maxreg를 안 깨고 평균은 오히려 큼 |
| residual bound가 여기서 필수다 | `no_residual_bound` ≈ full |
| ablation으로 Alloy/MultiStage를 재튜닝했다 | 로드하지 않음 |
| v2.2가 5도메인에서 통계적으로 유의하게 이겼다 | 기존 paired sign-flip p=1.0; 본 ablation은 구조 필수성 |

`ungated_bank`의 큰 평균 이득은 **개발 5도메인 retrospective**다. 보수 gate를 끄면 coverage가 1.0까지 올라가 평균은 오를 수 있으나, 이는 holdout 일반화·worst-case 안전을 대체하지 않는다. 논문에서는 “gate가 성능을 만든다”가 아니라 **“route가 위험을 가른다”**로 쓰는 것이 맞다.

## 권장 논문 문장

> On five non-holdout development domains, removing the deployment route and always applying the certified bank raises maximum raw unit regret to 30.3%. Disabling the v2.2 small-cohort route eliminates the SIT LFP gain while preserving RADAR. A single linear-rate expert violates the 2% max-regret cap on RADAR. Thus the paper’s structural claim is not a universal accuracy lift, but that **bank + risk-aware routing + exact fallback** is required to retain the documented gains without unsafe always-on correction.

## 재현

```bash
cd /Users/baghyeongbae/Desktop/연구/pp-extrapolation
python experiments/ccmr_v22_paper_structure_ablation.py
```

이미 `results.json`이 있으면 덮어쓰지 않는다.
