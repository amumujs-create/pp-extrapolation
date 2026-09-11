# Contract event-coordinate flow Stage 0 조기중단

작성자: 박진서  
판정: unified coverage challenger 기각

## 중단 이유

사용자가 정한 조기중단 규칙에 따라 PP-X보다 크게 악화되는 setting이 확인되면
남은 setting을 실행하지 않았다. HUST에서 큰 악화가 먼저 확인됐고,
중단 요청 시점까지 MICH를 포함한 6개 setting이 완료돼 있었다.

## 노출된 결과

| Setting | event-flow R² | PP-X R² | 판정 |
|---|---:|---:|---|
| HUST | 0.718 | 0.958 | 큰 악화 |
| Virkler | **0.970** | 0.888 | 개선 |
| NASA | 0.526 | 0.584 | 악화 |
| Sunwoda | 0.924 | 0.939 | 소폭 악화 |
| RWTH | 0.808 | 0.878 | 악화 |
| MICH | -0.090 | 0.751 | 큰 악화 |

MATR 2019, MATR batch 2, N-CMAPSS는 실행 전에 중단돼 결과가 없다.

## 해석

- observed/latent event coordinate를 하나의 hitting-time 구조로 통일하면
  형식적 coverage는 얻을 수 있다.
- 그러나 latent coordinate가 life-label 및 proxy-target 도메인의 PP-X
  구조를 대체하지 못했다.
- temporal-order loss 제거 여부도 HUST와 MICH 실패를 해결하지 못했다.
- Virkler에서 반복적으로 큰 개선이 확인된 것은 event-flow가 균열성장
  mechanism에는 잘 맞지만 범용 RUL operator는 아니라는 기존 결론을
  강화한다.

따라서 이 구조를 30-candidate 본실험으로 확장하지 않는다. 다음 challenger는
PP-X 예측을 버리고 latent flow로 전면 교체하지 않고, **PP-X의 exact coverage를
보존하면서 외삽 경로에서 residual derivative를 적분하는 구조**여야 한다.

원시 부분 결과:

- `results/contract_event_coordinate_flow_stage0/results.partial.json`
- 완료된 setting별 `*_predictions.npz`
