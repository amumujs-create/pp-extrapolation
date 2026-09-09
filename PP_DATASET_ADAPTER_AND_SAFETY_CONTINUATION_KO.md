# PP 데이터셋 adapter와 safety continuation 개선

## 무엇이 문제였나

축류팬의 최초 실패는 열이나 RUL 정렬 오류가 아니었다. 완전수명의 초기·중기
시점을 모두 학습한 뒤, fan당 late endpoint 하나만 있는 공식 test에 적용해
selection distribution이 달라진 것이 가장 컸다. 위험구간을 맞추자 PP pooled
R²가 -0.746에서 0.591로 회복됐다.

MATWI는 다른 문제다. label-only 입력에서 공구별 총수명 척도가 53--152회로
달라지지만 절삭조건과 원시 센서가 빠져 같은 wear가 생애의 어느 위치인지
식별하기 어렵다. Misata는 입력은 충분해 PP R²가 0.833이었지만, 단일 latent
damage를 NN이 쉽게 학습해 affine prior가 오히려 작은 bias를 더했다.

## 개선된 adapter 원칙

1. 데이터 열을 읽는 schema adapter와 외삽 과제를 정하는 task adapter를
   분리한다.
2. task를 `known boundary`, `truncated endpoint`, `regime/relationship shift`로
   먼저 분류한다.
3. train unit 내부 가상 절단점으로 실제 test endpoint 분포를 재현한다.
4. tail horizon, feature history, executor와 최적화값을 grouped validation에서
   함께 선택한다.
5. test label은 이 선택 뒤 한 번만 사용한다.

## safety-continuation PP

새 preset은 PP 내부에 matched NN 경로를 정확히 포함한다. NN과 같은 seed로
direct residual을 초기화하고 affine trust를 0.02에서 시작한다. 하나의 learned
gate가 근거가 있을 때만 affine prior 비중을 키운다. 이는 완성된 두 모델의
사후 ensemble이 아니라 한 네트워크의 연속적인 submodel 경로다.

이 구조는 Misata test를 확인한 뒤 개발됐으므로 현재 Misata 결과를 확증으로
바꾸지 않는다. 다음 사전 고정 고호트에서 기존 PP, safety continuation PP,
matched MLP를 같은 nested tuning budget으로 비교해야 한다.

## 현재 근거

- 축류팬 posthoc repair: PP pooled R² 0.591, MLP 0.502, 세 설정 모두 PP RMSE 승.
- MATWI untouched label-only: PP -0.137, MLP -0.151, 둘 다 절대 실패.
- Misata untouched synthetic: PP 0.833, MLP 0.854, PP 우월성 실패.
- MEMSS, GaAs, Device-B: 사전 최소 event/history 수 미달로 inconclusive.

따라서 현 시점의 올바른 주장은 "adapter 수정으로 축류팬 실패를 구조적으로
복구했고, prior가 무익할 때 NN으로 연속 복귀할 모델 경로를 추가했다"이다.
새 실제 고호트 성공은 아직 확보되지 않았다.
