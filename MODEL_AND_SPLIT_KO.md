# PP 모델과 데이터 분할

PP의 모델식은 다음과 같다.

\[
\hat y_{norm}(x)=w^Tz+b+g_\theta(z), \qquad z_j=(x_j-\mu_j)/\sigma_j.
\]

`Linear(d, 1)` 경로와 `Linear(d,32)-tanh-Linear(32,32)-tanh-Linear(32,1)` 경로를 더한다. 선형 경로는 unit-balanced weighted Ridge 해로 초기화하고 고정하며, 비선형 경로만 학습한다. Ridge alpha와 epoch는 unit-disjoint hull-out validation으로 선택한다. 물리식이나 도메인별 열화식은 forward 함수에 넣지 않는다.

새 데이터에서는 다음 순서를 고정한다.

1. failure boundary, unit ID, 시간축, 인과 입력, 외삽 좌표를 먼저 정한다.
2. 같은 unit이 두 split에 들어가지 않도록 train/validation/test unit을 분리한다.
3. train unit에서는 관측 support 내부 행만 사용한다.
4. validation/test unit에서는 사전 지정한 외삽 좌표가 train convex hull 밖인 행만 남긴다.
5. feature 평균·표준편차와 RUL scale은 train에서만 계산한다.
6. alpha와 early-stopping epoch는 validation으로만 선택한다.
7. seed 42–46을 학습하고 고정 test의 raw pooled R²를 주 지표로 보고한다.

현재 CLI는 한 개의 순서형 외삽 좌표를 지원한다. 여러 좌표를 쓰는 경우 `audit_convex_hull_support()`로 결합 좌표의 hull을 검사하고 split 딕셔너리를 직접 전달한다.

모델 입력공간 전체가 아니라 별도로 지정한 좌표에서 hull을 검사했다면 논문에도 그 사실을 그대로 표기해야 한다. 같은 unit의 미래 관측, 최종 수명 또는 test 전체 궤적에서 계산한 통계량은 입력으로 사용하지 않는다.

