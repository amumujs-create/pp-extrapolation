# 논문 노벨티 포지션

## 최종 방법의 중심

논문의 방법은 단순 residual NN이나 TTA가 아니라 다음 세 단계의 결합으로
정의한다.

1. **Typed prior compiler**: 데이터에서 관측 가능한 역할만 판별해 monotonic,
   boundary, rate, transport, regime contract를 조립한다.
2. **Support-aware neural executor**: affine prior tail과 NN correction을 함께
   학습하고, source anchor와 train-support 거리로 관측별 correction 신뢰도를
   조절한다.
3. **Verified abstention**: source 내부 검증을 통과하지 못한 prior는 켜지 않고
   formula-free PP 또는 plain NN 경로로 보낸다.

핵심 연구 질문은 “물리식을 NN에 넣으면 좋아지는가”가 아니라 **부분적으로만
관측 가능한 prior를 어떤 계약으로 컴파일하고, 외삽 위치에서 얼마만큼 신뢰하며,
검증 실패 시 어떻게 안전하게 사용하지 않을 것인가**다.

## TTA와의 차이

test 입력이나 test batch로 파라미터·통계량을 갱신하지 않는다. 모든 prior 선택,
anchor, support transform과 gate 강도는 source train/validation에서 고정된다.
test에서는 한 번의 forward inference만 수행한다.

## Counterfactual residual contract 결과

train unit의 마지막 두 관측으로 outward ray를 만들고, 거리와 함께 NN residual이
prior tail로 수렴하도록 하는 loss를 구현했다. HUST 단독 사용은 pooled R²
`0.856±0.027`로 soft-anchor `0.813`보다 높았지만 support-adaptive PP `0.910`보다
낮았다. 기존 최고 모델 위에서 validation은 5/5 seed 모두 counterfactual weight
0을 선택했다. 따라서 현 loss는 최종 핵심 기여나 성능 주장에 포함하지 않고
negative ablation으로 남긴다.

## 현재 방어 가능한 주장

- 하나의 보편 물리식을 모든 기계에 강제하지 않는다.
- prior 적용 가능성을 데이터 열 이름이 아니라 관측 역할과 source 검증으로 결정한다.
- 외삽 거리에서 NN correction의 신뢰도를 바꾸되 test-time adaptation은 하지 않는다.
- 동일 예산 로컬 PFN 대비 네 데이터에서 pooled R² 평균이 높다.

## 아직 필요한 증거

- 관련 prior selection, OOD gating, physics-guided residual 연구와 체계적 선행연구 비교
- compiler·anchor·support trust·abstention 각각의 고정 protocol ablation
- 기존에 보지 않은 데이터 또는 조건에서 prospective 실행
- exact convex-hull 거리와 안정적인 근사 거리의 계산량·성능 비교
- 실패 prior에서 abstention이 실제 regret를 줄이는지 검증

현재 단계에서는 **구성적 prior 컴파일과 검증 기반 사용 정책**이 가장 강한
노벨티 후보다. support decay 하나만 독립적인 강한 노벨티로 주장하지 않는다.
