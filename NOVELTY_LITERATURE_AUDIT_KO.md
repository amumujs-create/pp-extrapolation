# PP 연구 노벨티·선행연구 감사

검토일: 2026-09-07

## 결론

현재 PP는 **논문 소재가 될 수 있지만**, `affine + residual NN` 자체를 핵심 노벨티로
주장하기에는 약하다. Hybrid physics/data RUL, physics-informed loss, selective regression,
convex-hull applicability domain, 외삽용 validation은 각각 이미 연구돼 있다.

현재 가장 방어 가능한 중심 기여는 다음의 결합이다.

> **Unit-disjoint out-of-support RUL에서, 고정 affine tail과 학습형 regime residual을
> 사용하고, group-LOO validation과 다중 초기화의 일치가 확인될 때만 출력 scale 보정을
> 운반하며 그렇지 않으면 원 예측 또는 abstention으로 복귀하는 selective extrapolation
> protocol.**

즉 새로움은 단순한 혼합 네트워크보다 `외삽 구조 + 승인 조건 + 실패 시 fallback + 엄격한
평가 프로토콜`의 결합에 있다.

## 구성요소별 선행연구 중복

| PP 요소 | 관련 선행연구 | 단독 노벨티 판단 |
|---|---|---|
| affine/기초 모델 + NN correction | RUL의 model/data hybrid 및 corrective neural model이 이미 존재 | 약함 |
| physics/prior loss | PINN·physics-informed RUL 연구가 다수 존재 | 약함 |
| attention 또는 regime expert | attention, mixture/gating, degradation-stage 분할이 존재 | 약함~중간 |
| convex-hull 밖 평가 | applicability domain과 extrapolation validation 연구가 존재 | 약함 |
| uncertainty 기반 abstention | selective regression과 OOD rejection이 존재 | 약함 |
| 미래와 닮은 validation split | temporal extrapolation model selection에서 이미 제안 | 약함 |
| group-LOO scale transport | 일반 calibration 자체는 새롭지 않지만 RUL strict-tail 적용은 덜 흔함 | 중간 |
| 모든 seed가 scale 오류에 동의할 때만 calibration 승인 | 정확히 같은 조합은 이번 검색에서 찾지 못함 | 중간~강함 후보 |
| 승인 실패 시 identity를 보존하여 교차 데이터 악화를 막음 | selective prediction과 관련되지만 모델 보존 규칙은 차별 가능 | 중간 |
| unit-disjoint + convex-hull strict-tail 다중 도메인 benchmark | 일반 RUL random/ordinary holdout과 분명히 다름 | 강한 실험 기여 후보 |

## 가장 가까운 연구

1. Xu et al., *How Neural Networks Extrapolate*는 신경망의 외삽이 구조와 학습의 귀납편향에
   좌우된다는 이론적 배경을 제공한다. PP의 affine tail을 정당화하는 배경이지만 PP 구조와
   같은 방법은 아니다.
2. Krueger et al., *Risk Extrapolation*은 여러 학습 환경 사이 위험 차이를 줄여 더 극단적인
   분포 이동에 강건하게 한다. PP의 seed-consensus나 출력 보정과 직접 같지는 않지만
   `source evidence로 shift robustness를 제어한다`는 큰 목적이 겹친다.
3. Dirks and Poole은 미래 test와 닮은 시간 이동 validation과 ensemble을 사용해 외삽
   hyperparameter를 선택했다. 따라서 hull-out validation 선택 자체는 노벨티가 아니다.
4. *Extrapolation validation (EV)*은 convex hull 밖 표본의 외삽 위험을 평가하고 모델을
   선택한다. 따라서 convex-hull audit만으로 방법론 노벨티를 주장하면 안 된다.
5. Noskov et al.의 selective regression은 불확실할 때 회귀 예측을 거절하는 일반 이론을
   다룬다. PP의 abstention은 RUL strict-tail과 mechanism certificate에 특화해야 구별된다.
6. RUL 분야에는 Wiener degradation model과 corrective GRU의 interactive hybrid,
   model/data hybrid bearing RUL, self-attention PINN 등이 이미 있다. 물리/통계 모델과 NN을
   결합했다는 설명만으로는 신규성이 부족하다.

## 현재 증거가 지지하는 주장

- Seed-consensus calibration은 MATR `0.257→0.466`, HUST `0.773→0.899`, RWTH
  `0.507→0.743`으로 개선됐다.
- Virkler, Sunwoda, N-CMAPSS, MICH에서는 보정을 승인하지 않아 원 점수를 유지했다.
- 7개 평가에서 `3개 개선 / 4개 유지 / 0개 악화`라는 selective safety 결과가 있다.
- MATR에서 동일한 보정을 적용한 FT는 0.377이고 PP는 0.466이다.
- Latent regime PP는 원 PP보다 강하지만, test에서 gate가 late expert로 거의 포화된 사례가
  있어 두 expert의 지속적 혼합을 성능 원인으로 주장하면 안 된다.
- MICH의 음수 R²는 현재 시스템이 모든 도메인을 해결하지 못한다는 명확한 negative control이다.

## 논문에서 피해야 할 주장

- 최초의 physics-informed 또는 hybrid RUL network
- 최초의 affine + neural residual 구조
- 최초의 convex-hull 외삽 판정
- 모든 도메인에서 일반화하거나 모든 NN보다 우수함
- seed-consensus가 이론적으로 안전을 보장함
- 단순 output calibration 결과만으로 새로운 network architecture라고 표현함

## 권장 논문 포지셔닝

가장 응집력 있는 제목 방향은 다음과 같다.

**Consensus-Validated Prior-Residual Networks for Selective Out-of-Support RUL
Extrapolation**

기여는 세 개로 제한하는 것이 좋다.

1. **Strict-tail problem formulation:** physical unit을 먼저 분리하고 train convex hull 밖의
   후기 열화만 평가하는 RUL 외삽 정의.
2. **Prior-residual executor:** frozen affine tail과 학습형 nonlinear/latent-regime residual로
   외삽 귀납편향과 표현력을 분리.
3. **Consensus validation transport:** group-LOO와 seed unanimity로 scale correction을
   승인하고, 증거가 없으면 identity 또는 abstention으로 복귀.

이 구성은 각 부품의 최초성보다 **부품 간 책임 분리와 test-label-free 승인 규칙**을
노벨티로 삼는다. Consensus는 이제 exact binomial seed-vote test로 정의됐고, 승인된 세
데이터 모두 물리-unit paired bootstrap CI가 0 아래임을 확인했다. 현재 상태의 신규성은
중간 이상이다. 최상위 저널 수준으로 강화하려면 강한 selective/OOD baseline과 완전 고정된
추가 데이터 검증이 남아 있다. 통계 결과는 `STATISTICAL_NOVELTY_EVIDENCE_KO.md`에 정리했다.

## 논문 제출 전 우선 실험

1. `unanimity`, 4/5, 3/5, validation-score 단독 규칙의 risk–coverage 비교.
2. Deep ensemble variance, conformal/selective regression, 단순 validation affine calibration과
   동일 coverage에서 비교.
3. PP에서 affine path, regime expert, group-LOO, seed consensus를 하나씩 제거하는 ablation.
4. 데이터셋 단위 leave-one-domain-out으로 calibration 승인 threshold를 고정하고 마지막
   데이터셋에는 한 번만 적용.
5. 평균 점수뿐 아니라 calibration 승인 precision, 잘못된 승인 횟수, regret, seed SD,
   unit-paired bootstrap을 보고.

## 참고 문헌 링크

- Xu et al., 2021, How Neural Networks Extrapolate:
  https://openreview.net/forum?id=UH-cmocLJC
- Krueger et al., 2021, Out-of-Distribution Generalization via Risk Extrapolation:
  https://proceedings.mlr.press/v139/krueger21a.html
- Dirks and Poole, 2022, Automatic Neural Network Hyperparameter Optimization for Extrapolation:
  https://arxiv.org/abs/2210.01124
- Extrapolation validation (EV), 2024:
  https://doi.org/10.1039/D3DD00256J
- Noskov et al., 2024, Selective Nonparametric Regression via Testing:
  https://proceedings.mlr.press/v222/noskov24a.html
- Van Amersfoort et al., 2020, Deterministic Uncertainty Quantification:
  https://proceedings.mlr.press/v119/van-amersfoort20a.html
- Liao et al., 2023, Self-attention assisted PINN for RUL:
  https://doi.org/10.1016/j.aei.2023.102195
- Interactive Hybrid Model for RUL with UQ, 2024:
  https://doi.org/10.1109/TII.2023.3288225
- Hybrid drive of data and model for bearing RUL, 2022:
  https://doi.org/10.1109/JSEN.2022.3188646
