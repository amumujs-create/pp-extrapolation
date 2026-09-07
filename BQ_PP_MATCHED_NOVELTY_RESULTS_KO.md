# BQ-PP matched control과 노벨티 검증

## 핵심 판정

**Boundary-quotient parameterization과 prior-centered contraction은 지지됐다.**
그러나 bounded residual이 unbounded residual보다 범용적으로 우수하다는 주장은
지지되지 않았다. 논문은 이 두 결론을 분리해야 한다.

## matched 실험

모든 neural arm에 11개의 같은 causal-history feature, width 64, 5,005개 파라미터,
AdamW, 학습률 `1e-3`, weight decay `.01`, seed 42–46, validation epoch selection을
사용했다. validation으로 epoch를 선택한 뒤 train+validation에서 같은 epoch 수만큼
재학습했다. soft-boundary loss weight는 `.1, 1, 10`을 validation으로 비교해 `.1`을
선택했다.

| arm | Sunwoda pooled R² | RWTH pooled R² | MICH pooled R² | dataset-mean pooled R² | dataset-macro unit R² |
|---|---:|---:|---:|---:|---:|
| direct NN | -1.352 | 0.633 | 0.684 | -0.011 | -0.135 |
| soft-boundary NN | -3.107 | 0.483 | -0.057 | -0.894 | -1.137 |
| hard-boundary fully trainable NN | 0.900 | 0.855 | 0.319 | 0.691 | 0.491 |
| affine quotient only | 0.281 | 0.659 | -3.343 | -0.801 | -1.525 |
| frozen affine + unbounded residual | 0.718 | 0.788 | **0.759** | 0.755 | **0.697** |
| **frozen affine + bounded residual BQ-PP** | **0.939** | **0.878** | 0.468 | **0.762** | 0.684 |

direct NN은 단일 seed pooled R² SD가 Sunwoda `1.804`, RWTH `.554`, MICH `.359`로
크게 불안정했다. BQ-PP의 같은 SD는 `.020`, `.016`, `.0004`다. 결과는
prediction ensemble로 보고했지만, 단일 seed 안정성도 별도로 개선됐다.

## physical-unit paired 통계

| BQ-PP 대조군 | BQ-PP RMSE 우세 unit | 평균 상대 RMSE 차이 | unit bootstrap 95% CI | Wilcoxon p |
|---|---:|---:|---:|---:|
| direct NN | 17/25 | -33.3% | **[-53.9%, -11.8%]** | .0028 |
| soft-boundary NN | 24/25 | -56.2% | **[-68.7%, -42.5%]** | 1.13×10⁻⁶ |
| affine quotient only | 25/25 | -59.2% | **[-66.2%, -51.9%]** | 5.96×10⁻⁸ |
| hard-boundary trainable | 20/25 | -5.4% | [-25.4%, +19.0%] | .071 |
| frozen unbounded | 14/25 | -6.1% | [-27.6%, +16.1%] | .578 |

따라서 BQ-PP는 direct, soft-boundary, affine-only보다 유의하게 낫다. affine 동결과
residual bound의 개별 효과는 방향성은 있지만 현재 25 units로 유의하지 않다.

## prior-centered boundary contraction 증명

\[
\hat y=m\operatorname{softplus}(\ell(z)+c_\theta(z)),\qquad |c_\theta(z)|\le B,
\]

\[
\hat y_A=m\operatorname{softplus}(\ell(z)).
\]

softplus는 도함수가 `(0,1)` 범위이므로 1-Lipschitz이다. 따라서

\[
\begin{aligned}
|\hat y-\hat y_A|
&=m|\operatorname{softplus}(\ell+c_\theta)-\operatorname{softplus}(\ell)|\\
&\le m|c_\theta|\\
&\le mB.
\end{aligned}
\]

즉 학습된 residual이 어떤 OOD 입력을 받더라도 affine 외삽 경로를 바꿀 수
있는 절대량은 `mB`를 넘지 못한다. `m→0`이면 neural correction도 0으로
수축하고 RUL은 경계에서 정확히 0이다.

5 seed×3,529 source rows = **17,645개 실제 예측**에서 `|ŷ-ŷA|≤mB`를 검사했고
위반은 0건이었다. envelope 사용률의 최대는 `.99889`, 95백분위는 `.99828`이어서
제약이 실제 학습에서 활성화됐음도 확인했다.

## margin-adaptive contraction 후속 가설

늦은 tail에서 bound를 `B(m)=B[1+g(1-clip(m,0,1))]`로 늘리는 확장을
추가 검증했다. 이때도 `|ŷ-ŷA|≤mB(m)`이고 `m→0`에서 수축한다.

| growth g | Sunwoda | RWTH | MICH | 최저 dataset R² | 평균 R² |
|---:|---:|---:|---:|---:|---:|
| 0 | **.939** | **.878** | .468 | .468 | **.762** |
| .5 | .763 | .874 | .530 | .530 | .722 |
| 1 | .669 | .840 | .283 | .283 | .597 |
| 2 | .719 | .738 | **.746** | **.719** | .734 |
| 4 | -.493 | .455 | .616 | -.493 | .193 |

`g=2`는 최저 dataset R²를 크게 높였지만, mean validation은 `.5`, minimax
validation은 `1`을 선택했다. 따라서 `g=2`를 최종 모델로 채택하면 test
사후 선택이 된다. 이 확장은 **박사논문용 후속 가설**으로만 남기고 PP 저널
최종 모델에는 넣지 않는다.

## 논문용 노벨티 문장

> We propose a prior-centered boundary-contraction network for strict-tail RUL
> extrapolation. The model predicts a positive RUL-to-health-margin quotient around
> a frozen affine tail and restricts the neural deviation to a deterministic envelope
> that contracts linearly at the failure boundary. Unlike direct regression or a
> soft boundary penalty, the construction guarantees both exact boundary consistency
> and a pointwise bound on the neural alteration of the extrapolative path.

주장의 범위는 정확히 다음으로 제한한다.

- hard boundary 자체의 최초성을 주장하지 않는다.
- bounded residual의 평균 정확도 우월성을 주장하지 않는다.
- `mB` contraction 보장, direct/soft/affine-only 대비 개선, strict-tail 안정성을
  핵심 기여로 사용한다.
- 새 failure mechanism의 zero-shot 해결을 주장하지 않는다.

재현 코드는 `experiments/bq_pp_matched_controls.py`, 실험 결과는
`results/bq_pp_matched_controls_v1/results.json`이다. margin-adaptive 개발 실험은
`experiments/bq_margin_adaptive_development.py`와
`results/bq_margin_adaptive_development_v1/results.json`에 있다.
