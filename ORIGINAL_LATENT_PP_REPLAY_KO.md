# 원래 latent PP 결과 재현 확인

당시 validation이 선택해 기록한 각 seed의 regularizer 설정을 그대로 사용해 처음부터 재학습했다. 300 epoch, patience70, width24, lr0.0005, weight_decay2. 모델/스케일링/분할 코드가 커밋60dca70 대비 변경되지 않았음을 git diff로 확인했다. 새로운 설정 탐색이나 test 기반 조정은 하지 않았다.

| 데이터셋 | 원래 ensemble pooled R² | 재실행 | 평균 ± SD |
|---|---:|---:|---:|
| hust | 0.810657642 | 0.810657642 | 0.709318 ± 0.071135 |
| virkler | -5.373762048 | -5.373762048 | -5.495880 ± 1.248462 |

10개 seed 모두 pooled R² 및 선택 epoch가 기존 기록과 정확히 일치했다. 저장된 원예측을 재계산한 ensemble 점수도 일치했다. 기존 기록에 row prediction이 없었으므로 과거 예측과의 행별 bitwise equality까지 입증한 것은 아니다.

앞선 150-epoch/seed42 단일 설정 선택/다른 width·optimizer grid의 비교는 원래 PP의 재현 실험이 아니었다. 그 점수를 원래 PP 0.811의 소실로 해석하면 안 된다. 원래 HUST 결과는 재현됐으며 원래 Virkler 실패도 재현됐다.

이 재현은 당시 선택된 모델을 다시 학습한 것이다. 9개 후보 전체의 재탐색을 반복한 것은 아니며 새 데이터 확증도 아니다. FT와 원래 PP의 공정한 비교를 위해서는 FT에도 동일한 학습·validation 선택 규칙을 적용한 별도 실험이 필요하다.
