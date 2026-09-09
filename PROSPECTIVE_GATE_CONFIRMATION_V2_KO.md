# PP 사전 route 검증 v2 프로토콜

이 문서는 현 static descriptor gate의 실패를 숨기지 않고, 다음 외부 코호트에서 검증 가능한 protocol로 바꾸기 위한 사전등록 양식이다. 실행 전 이 파일, 코드 commit hash, 원시파일 SHA-256 manifest를 고정한다.

## 1. 데이터 고정

1. 최소 10개의 독립 run-to-failure unit을 확보한다. train/validation/test unit 수와 unit ID를 label 열기 전에 기록한다.
2. 센서 채널 매핑, 시간 순서, 실패 경계, causal prefix feature, train-only normalization을 schema 검사로 확정한다.
3. test target 파일과 feature 파일의 hash를 별도로 남긴다. 중복 unit과 byte-identical file은 test에서 제외한다.
4. strict support boundary와 direction을 train/validation 정보만으로 정한다. test row가 boundary 밖인지 label 없이 audit한다.

## 2. 허용되는 선택과 금지되는 선택

- 허용: train/validation R²·RMSE로 PP/MLP의 hyperparameter와 checkpoint를 고르는 것.
- 허용: validation disagreement 및 support distance로 **abstain**을 선택하는 것.
- 금지: test R²를 본 뒤 feature, normalization, boundary, route rule, eligible-unit 기준을 바꾸는 것.
- 금지: 관측 뒤 rule을 바꾸고 이를 prospective success로 부르는 것.

## 3. 사전 고정 route 출력

각 test unit에는 label을 열기 전에 아래 셋 중 하나를 남긴다.

| output | operational action | paper interpretation |
|---|---|---|
| `PP` | frozen PP prediction | PP route approved |
| `MLP` | frozen plain MLP prediction | PP route rejected |
| `ABSTAIN` | no automated RUL value | information insufficient |

`ABSTAIN`은 실패를 숨기기 위한 zero-coverage gate가 아니므로 coverage를 함께 보고한다. 승인·거절·abstain 각각 최소 3 unit이 없으면 route-accuracy claim은 exploratory로만 보고한다.

## 4. 성공 판정

주 결과는 test pooled R², unit-macro R², RMSE, MAE와 cluster bootstrap 95% CI다. route gate의 별도 주장은 다음을 모두 만족할 때만 한다.

1. pre-label route manifest가 존재할 것,
2. eligible independent test unit이 10개 이상일 것,
3. approved PP unit에서 PP−MLP paired unit-level error difference의 95% CI가 0보다 작을 것,
4. non-abstained coverage와 route confusion matrix를 모두 공개할 것.

이보다 약한 결과(한두 unit, 중복, protocol repair 후 재실행)는 외부 개발 증거로만 표기한다.
