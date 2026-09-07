# MICH PP 실패 원인 검증

## 결론

MICH는 PP의 단순 출력 scale 오류가 아니라 **baseline PP의 representation/shape failure**로
분류한다. 현재 PP에서 residual이 학습되지 않았고, validation-only affine transport도
명시적으로 거절됐다. 데이터 자체가 예측 불가능한 것은 아니다. 별도 PAE 연구의
boundary-gated history model은 양의 성능을 냈지만 PP 논문 모델로 합치지 않는다.

## 1. Strict extrapolation 확인

| 항목 | 값 |
|---|---:|
| validation hull-out | 100% |
| test hull-out | 100% |
| validation median hull distance | 0.730 |
| test median hull distance | 3.426 |

Test가 validation보다 약 4.7배 먼 strict health-tail이다. MICH 실패를 interpolation 결과로
설명할 수 없다.

## 2. PP residual collapse

| 진단 | 결과 |
|---|---:|
| validation에서 PP의 affine 대비 상대 MSE 이득 | `1.34×10⁻⁸` |
| normalized residual activity | `1.45×10⁻⁷` |
| 선택 epoch | `0, 0, 0, 0, 0` |
| PP pooled R² | -1.522189698 |
| Ridge pooled R² | -1.522189764 |
| 두 R²의 절댓값 차이 | `6.58×10⁻⁸` |

다섯 seed 모두 epoch 0을 선택했고 PP는 수치적으로 Ridge와 같다. Seed variance가 0인 것은
안정적인 성공이 아니라 nonlinear correction이 비활성화된 결과다.

## 3. Output transport 반증

다섯 seed 모두 `identity`를 선택했다. Validation 6 units 중 affine이 identity보다 나은
unit은 하나뿐이었다.

| 통계 | 값 |
|---|---:|
| mean validation MSE gain | -267.98 |
| paired-unit bootstrap 95% CI | **[-624.64, -12.95]** |
| transport 승인 | 거절 |

CI 전체가 음수이므로 scale/offset transport가 MICH를 해결한다는 가설은 validation에서
기각된다.

## 4. 다른 구조의 양성 대조

별도 PAE 연구의 shared boundary-gated NN 결과는 다음과 같다.

| 지표 | 결과 |
|---|---:|
| MICH macro-unit R² | `0.661±0.068` |
| MICH pooled R² | `0.798` |
| 저장 결과 verification | 모든 검사 통과 |

이 양성 대조는 MICH 데이터에 예측 신호가 존재하며 현재 PP의 실패가 단순한 데이터 불능은
아님을 보여준다. 그러나 PAE 모델은 boundary gate, causal history representation, 공유
cohort 학습과 전처리가 함께 달라졌다. 따라서 이 비교만으로 어느 한 요소가 성능 상승의
원인이라고 단정하지 않는다. 또한 PAE 모델은 PP 논문에 합치지 않는다.

## 논문에서 사용할 해석

MICH는 세 가지 역할을 한다.

1. hull 밖 거리만으로 PP 성공을 예측할 수 없다는 negative control;
2. seed disagreement가 0이어도 residual collapse이면 신뢰 근거가 아니라는 반례;
3. PP verifier가 보정 가능한 scale failure와 새 representation이 필요한 shape failure를
   구분한다는 사례.

PP 논문에서는 MICH를 실패로 그대로 보고하고 transport/identity 이후 applicability
abstention 대상으로 둔다. MICH 성능 개선은 PAE 또는 후속 history/boundary 모델 연구의
범위로 유지한다.

기계 판독 결과: `results/mich_failure_validation_v1/results.json`
