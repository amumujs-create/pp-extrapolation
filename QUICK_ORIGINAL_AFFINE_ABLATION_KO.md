# 원래 PP affine 동결 비교 (MATR2019)

원래 PP의 seed별 선택 정규화 계수, 300 epoch, patience 70, lr 0.0005, weight decay 2, width 24, 데이터 분할과 초기화를 유지했다. affine requires_grad만 변경했다. 학습 가능한 affine에도 같은 optimizer의 weight decay가 적용된다. 원래 설정을 재사용한 paired ablation이며 개선안별 재튜닝 또는 새로운 확증 실험은 아니다.

| 모델 | 평균 예측 pooled R² | 단일 seed R² 평균 | seed SD |
|---|---:|---:|---:|
| original_frozen | 0.257392 | 0.171052 | 0.108738 |
| trainable_affine | -0.067811 | -0.393491 | 0.167580 |

원래 PP ensemble 수치가 정확히 재현됐다. 이번 고정 설정에서 affine 동결 해제는 5개 모든 seed에서 원래 PP보다 낮고 ensemble도 악화됐다. 이전 -0.160은 이 실험의 값이 아니며 이번 값은 -0.067811이다. FT+affine 두 구조의 재튜닝은 수행하지 않았다. 기존 FT 0.344는 150/25 프로토콜 결과이므로 현재 표의 엄밀한 대조군이 아니다.
