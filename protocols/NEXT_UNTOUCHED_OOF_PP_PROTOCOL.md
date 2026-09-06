# Frozen protocol for the next untouched cohort

이 파일 이후 선택되는 새 데이터에는 아래 규칙을 변경하지 않는다.

- Eligibility: 실제 run-to-failure 또는 명시적 EOL, 독립 train unit 12개 이상, validation/test unit 분리 가능.
- Split: unit-disjoint이며 validation과 test endpoint가 선언한 degradation coordinate의 train convex hull 밖에 있어야 한다.
- Base model: 공개 저장소의 PP 기본 설정, seeds 42–46.
- OOF target: train unit 3-fold outer split, 각 fold 내부 group validation, teacher seeds 101–103.
- Inference: PP 한 개와 uncertainty head 한 개. Test ensemble은 분석용으로만 계산한다.
- Gate grid: beta `(0,.05,.1,.25,.5,1,2,4,8)`, gamma `(0,1,2,4,8,16,32,64)`.
- Localization certificate: 관측별 head가 validation에서 최적 상수 shrinkage보다 mean-seed MSE를 상대 0.1% 이상 줄이고 gamma가 0보다 클 때만 활성화한다.
- Primary endpoint: raw pooled R². Secondary: pooled RMSE, unit-macro R², seed SD.
- Confirmatory success: certified PP가 base PP보다 pooled RMSE가 낮고 paired unit-bootstrap 95% CI 상한이 0 미만이어야 한다. 모든 결과와 coverage를 보고한다.
- Test label은 모델, gate, threshold, shell 또는 데이터 포함 여부 선택에 사용하지 않는다.

기존 HUST, Virkler, NASA, RWTH, MICH, Sunwoda, XJTU, MATR, Oxford, C-MAPSS 계열은 confirmatory untouched로 재사용하지 않는다.
