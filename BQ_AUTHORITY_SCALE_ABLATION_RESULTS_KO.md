# Residual authority scale ablation 결과

작성자: 박진서  
프로토콜: `protocols/BQ_AUTHORITY_SCALE_ABLATION_PROTOCOL.md`  
실험: `experiments/bq_authority_scale_ablation.py`  
산출물: `results/bq_authority_scale_ablation_v1/results.json`

## 한 줄 결론

**3-scale / continuous도 MICH를 올릴 수 있다. 하지만 dual이 “유일한 해”는 아니며, 그렇다고 3개 이상이 더 낫다고 말하기도 어렵다.**  
자유도를 늘리면 MICH ensemble은 더 오를 수 있지만 **seed 불안정**이 커지고, validation이 고른 continuous는 Sunwoda를 깎는다.

## Ensemble R² (seeds 42–46 평균 예측)

| Arm | Sunwoda | RWTH | MICH | min | mean |
|---|---:|---:|---:|---:|---:|
| Fixed \(B=2\) | **0.939** | **0.878** | 0.468 | 0.468 | 0.762 |
| Dual \(2+6\) | 0.934 | 0.842 | 0.751 | 0.751 | 0.842 |
| 3-scale \(2,4,6\) | 0.932 | 0.859 | **0.832** | **0.832** | **0.874** |
| Continuous \([2,6]\) | 0.869 | 0.857 | 0.828 | 0.828 | 0.852 |

## Validation으로 고르면?

Joint validation macro MSE 순위:

1. **continuous** (0.298) ← validation 선택
2. fixed (0.304)
3. dual (0.309)
4. tri_scale (0.316)

즉 이 matched 설정에서는 **validation이 dual을 1등으로 고르지 않는다.**  
Dual vs Fixed 2% 개선 규칙도 이 joint val에서는 통과 실패(상대 −1.6%).

## 안정성 (중요)

MICH seed별 R²:

| Arm | seed R² | seed SD |
|---|---|---:|
| Fixed | 0.47로 거의 고정 | ≈0 |
| Dual | 0.57–0.77 | 0.077 |
| 3-scale | 0.16–0.86 | **0.290** |
| Continuous | 0.13–0.89 | **0.288** |

3-scale·continuous는 ensemble만 보면 MICH가 좋아 보이지만, **나쁜 seed가 크게 무너진다.**

## 발표 Q&A용 답

> “왜 꼭 dual이야? 3개나 continuous면 안 돼?”

**답:**
가능합니다. 같은 residual network로 Fixed / Dual / 3-scale / Continuous를 matched ablation 했습니다.

- Fixed는 MICH를 0.468에 묶고, Dual은 0.751로 복구합니다.
- 3-scale·Continuous도 MICH ensemble을 더 올릴 수 있습니다 (0.83 근처).
- 다만 그건 **dual이 최적 개수**라는 뜻이 아니라, **권한 스케일을 늘리는 일반화가 가능하다**는 뜻입니다.
- 실제로 validation은 Continuous를 고르지만 Sunwoda가 떨어지고, 3-scale/Continuous는 MICH seed 분산이 Dual보다 훨씬 큽니다.
- 그래서 현재 논문은 “2가 최적”이 아니라 **Fixed가 부족한 반례에 대한 최소 이산 확장(Dual)을 조건부 허용**한 것이고, Multi-scale / Continuous는 후속 일반화로 열어 둡니다.

## 논문 포지션에 대한 함의

- Algorithm 1의 Dual executor를 지금 바꾸지 않아도 된다.
- “왜 하필 2와 6인가?”에는:
  - 현재 증거: MICH 복구에 필요한 **최소 추가 자유도**
  - 미주장: 전역 최적 scale 개수
- 후속: Continuous authority를 **unit-risk / seed-stability gate**와 함께 재설계하면 Q에 더 직접 답할 수 있다.
