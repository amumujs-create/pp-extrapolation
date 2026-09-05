# Prior 실패 시 PP fallback의 현재 증거

## 결론

`prior 실패 -> 무조건 PP`는 현재 증거로 지지되지 않는다. PP의 affine-tail 경로는
Virkler, NASA, C-MAPSS strict OP hull에서는 일반 NN보다 강하지만 HUST에서는 같은
정보의 plain MLP보다 낮다. 따라서 PP도 source의 unit-disjoint 외삽 validation에서
구조 이득을 확인한 뒤에만 선택해야 한다.

## 같은 정보의 plain NN ablation

HUST, Virkler, NASA 비교에서 두 모델은 같은 causal 입력, 같은 물리 unit 분할,
같은 strict-tail test 행, width 32의 tanh 은닉층 2개, AdamW, unit weighting,
출력 clipping, validation checkpoint, seeds 42–46을 사용했다. Plain MLP는 표준
PyTorch 초기화를 사용한다. PP는 여기에 validation으로 선택한 frozen affine path와
zero-start nonlinear correction을 추가한다.

| 데이터셋 | plain MLP pooled R² | PP pooled R² | PP−plain | seed 승리 |
|---|---:|---:|---:|---:|
| HUST | **0.758±0.032** | 0.724±0.039 | -0.035 | 0/5 |
| Virkler | -0.604±0.472 | **0.857±0.019** | +1.460 | 5/5 |
| NASA normalized health | 0.233±0.094 | **0.495±0.004** | +0.263 | 5/5 |

C-MAPSS FD002+FD004의 기존 동일 strict OP1–OP3 hull 개발 결과에서는 formula-free
shared NN v2가 `0.404±0.110`, staged PP가 `0.749±0.015`였다. 이 결과도 PP의
외삽 안정성을 지지하지만, 모두 이미 관찰한 데이터에 대한 retrospective 개발
근거다.

## 논문에서 사용할 라우팅 규칙

1. 컨셉 계약과 source validation에서 prior의 추가 이득을 확인한다.
2. prior가 통과하면 PAE structured head를 사용한다.
3. prior가 실패하거나 없으면 같은 source 외삽 validation에서 PP와 plain NN을
   비교한다.
4. PP가 사전 고정한 최소 이득과 seed 합의 기준을 통과할 때만 PP head를 사용한다.
5. PP가 통과하지 않으면 plain NN 또는 abstain으로 보낸다. final-test 결과로 route를
   바꾸지 않는다.

이 규칙의 논문상 의미는 특정 prior뿐 아니라 **fallback 구조도 전이 가능한지
검증한다**는 것이다. 다만 현재 표는 라우팅 규칙을 설계한 retrospective ablation이며,
규칙 자체의 독립 검증은 아직 아니다.
