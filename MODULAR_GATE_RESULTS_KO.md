# Evidence-gated modular extrapolation 개발 결과

## 질문

외삽 모델을 plain NN, affine tail, PP 같은 부품으로 나누고, source validation에서
이득이 있는 부품만 조립하면 새로운 unit에서도 올바른 부품이 선택되는가?

후보는 결과를 보기 전에 `plain_nn`, `affine_tail`, `pp` 세 개로 고정했다. Plain이
아닌 head는 validation 평균 MSE를 2% 이상 줄이고 5개 seed 중 4개 이상에서 plain을
이겨야 선택 가능하게 했다. Final-test label은 route 선택에 사용하지 않았다.

## 단일 validation gate

| 데이터셋 | validation 선택 | plain NN final R² | affine final R² | PP final R² | 선택 final R² |
|---|---|---:|---:|---:|---:|
| HUST | PP | **0.758±0.032** | 0.601 | 0.724±0.039 | 0.724±0.039 |
| Virkler | PP | -0.604±0.472 | -0.823 | **0.857±0.019** | **0.857±0.019** |
| NASA | 모든 outer fold에서 PP | 0.233±0.094 | 0.424 | **0.495±0.004** | **0.495±0.004** |

Virkler와 NASA에서는 route가 맞았다. HUST validation에서는 PP가 plain보다 MSE를
22.1% 줄이고 4/5 seed에서 이겨 PP가 선택됐지만, final protocol 9–10에서는 plain이
더 높았다. 따라서 단일 source validation의 module 이득만으로 새 protocol 전이를
보장할 수 없다.

## HUST nested source-protocol 안정성

HUST source protocol 1–6을 차례로 통째로 제외했다. 제외한 protocol의 셀 절반은
checkpoint 선택, 나머지 절반은 route 감사에 사용했다. 감사 행은 inner-train
capacity support 아래에만 두었다.

| 제외 protocol | plain NN R² | PP R² | PP MSE 이득 | seed 승리 | PP 안정 |
|---:|---:|---:|---:|---:|:---:|
| 1 | 0.641±0.073 | 0.847±0.033 | +0.574 | 5/5 | 참 |
| 2 | 0.248±0.322 | 0.754±0.086 | +0.672 | 4/5 | 참 |
| 3 | 0.492±0.057 | 0.553±0.155 | +0.120 | 4/5 | 참 |
| 4 | 0.593±0.052 | 0.837±0.148 | +0.598 | 4/5 | 참 |
| 5 | 0.641±0.087 | 0.874±0.052 | +0.649 | 5/5 | 참 |
| 6 | 0.598±0.251 | 0.475±0.351 | -0.307 | 2/5 | 거짓 |

PP는 source protocol 5/6에서 안정성 문을 통과했지만 final protocol에서는 plain보다
낮았다. source 반복성도 새로운 protocol의 head 순위를 완전히 보장하지 못한다는
반례다.

## PAE 안에서의 해석

이 결과가 곧 전체 PAE route의 HUST 실패를 뜻하지는 않는다. HUST에서는
rate-to-boundary prior가 source에서 검증되므로 원래 순서대로라면 PP fallback에
도달하지 않는다. 실제 fallback 검증 대상은 prior가 없거나 꺼진 데이터다.

- FD004: load/CF prior가 꺼진 상태이고, 기존 strict hull에서 shared NN
  `0.383±0.164` 대비 PP `0.744±0.005`로 fallback 근거가 있다.
- TINY: load prior가 redundant 판정을 받았으므로 핵심 fallback 대상이다. 하지만
  공개 PAE Git의 T299 runner가 요구하는 TINY 생성 의존성 두 개가 저장소에 없어
  같은 입력의 PP 재학습을 현재 재현할 수 없다. 캐시 예측만으로 새 PP 결과를
  만들지 않았다.

## 현재 채택할 규칙

`prior 실패 -> PP`를 고정 규칙으로 사용하지 않는다. 정적 컨셉 계약으로 후보
부품을 제한하고 source validation으로 부품의 실패를 걸러내되, 이것을 전이 보증으로
표현하지 않는다. 논문용 최종 route는 후보 library와 선택 문턱을 잠근 뒤, 아직
선택에 사용하지 않은 concept-aligned cohort에서 coverage, 잘못된 head 선택,
성능 regret을 함께 평가해야 한다.

이번 결과의 중요한 기여 후보는 최고 점수만 선택하는 router가 아니라 **module
transportability의 실패를 명시적으로 측정하는 선택적 외삽**이다.

## 보수적 만장일치 gate 후향 점검

HUST 반례를 본 뒤 `모든 source 환경에서 안정성 통과 시에만 PP`라는 더 보수적인
규칙을 시험했다. HUST는 5/6이라 plain NN, Virkler는 3/3이라 PP, NASA는 네 outer
fold가 모두 PP를 선택해 PP route가 됐다.

| 데이터셋 | 선택 head | final pooled R² | final 최선 head | 관측 regret |
|---|---|---:|---|---:|
| HUST | plain NN | 0.758 | plain NN | 0.000 |
| Virkler | PP | 0.857 | PP | 0.000 |
| NASA | PP | 0.495 | PP | 0.000 |

선택 결과의 dataset-macro pooled R²는 `0.703`으로 always-PP `0.692`보다 `+0.012`
높았다. PP route coverage는 2/3, prediction coverage는 3/3이었다. 그러나 만장일치
문턱은 HUST final 순위를 본 뒤 만든 것이므로, 관측 regret 0은 새 gate의 검증
결과가 아니라 후향 수리 결과다.
