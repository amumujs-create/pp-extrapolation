# NASA Milling locked PP replay protocol

이 문서는 결과 실행 전에 고정한다.

- 목적: PP에 사용하지 않은 공구마모 도메인에서 소재·unit 이동과 후기 시간꼬리
  외삽을 동시에 검사한다.
- 데이터: NASA Milling. material 1 train cases 1,2,3,9,10,11; validation case 12;
  material 2 source cases 5,8,13,14,15,16 중 관측 가능한 case.
- target: 사전 유지보수 경계 VB=0.50 mm까지 RUL. 데이터셋의 공식 고장 정의로
  주장하지 않는다.
- 특징: 현재까지 관측된 wear, causal prefix rate, elapsed time, previous interval.
- split: train wear의 60% 분위수 이하만 학습하고, validation/source는 실제 train
  최대 wear보다 큰 행만 평가한다. unit은 세 partition에서 분리한다.
- PP: soft anchor 0.1, support decay 0.3, counterfactual loss 0. seeds 42–46.
- baseline: 동일 구조·optimizer의 plain NN, 동일 입력과 행.
- primary metric: raw pooled R². secondary: unit-macro R², RMSE, MAE.
- 성공: mean pooled R²가 0보다 크고 plain NN보다 높다.
- 실패 시: 이 도메인을 PP 적용 범위에서 제외하고 abstention/coverage 근거로 기록한다.
- 제한: 이 데이터는 이전 PAE 개발에서 관찰됐으므로 globally prospective 또는
  untouched 데이터라고 부르지 않는다. PP 구조에 대한 locked transfer replay다.
