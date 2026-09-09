# 30-candidate robust baseline coverage

각 robust baseline은 validation-only로 24개 architecture 후보와 6개 중복 없는
penalty 후보, 총 30개 설정을 평가한 뒤 선택 설정을 seed 42–46으로 재학습했다.

| Dataset | V-REx | GroupDRO | monotone NN |
|---|---:|---:|---:|
| HUST | .809 | .712 | .820 |
| Virkler | .601 | .724 | .552 |
| Sunwoda | -.926 | -.556 | -1.113 |
| RWTH | .645 | .602 | -.188 |
| MATR2019 | -.691 | -.578 | -.660 |
| MATR-b2 | .850 | .350 | .241 |
| N-CMAPSS | .883 | .881 | .888 |
| XJTU | -1.716 | -1.727 | -1.708 |
| FEMTO | -.520 | -.520 | .088 |
| MICH | -.697 | -.751 | -.690 |
| NASA Milling | -1.086 | -1.840 | -1.041 |

Milling은 validation group이 하나이므로 이 표의 수치를 비교 가능성 민감도 결과로만
해석한다. Engression과 full-train SVGP는 MICH에서 각각 -1.580, -2.247이었다.

MICH에서는 seed-average prediction을 기준으로 PP-X와 각 robust baseline의 exact
unit sign-flip 및 unit-cluster bootstrap을 수행했다. PP-X의 R² .826은 V-REx 대비
Δ=+1.523 (95% CI [1.296, 1.830], p=.0078), GroupDRO 대비 +1.577
([1.326, 1.871], p=.0078), monotone NN 대비 +1.516 ([1.294, 1.810], p=.0078)이다.

11개 데이터셋의 selection log와 pooled 성능은
`results/final_30_candidate_competitors_v1/results.json`에 저장했다. MICH는 seed-average
prediction artifact까지 보존해 `results/mich_cluster_inference_v1/`에서 통계검정을
재현할 수 있다. 나머지 데이터셋의 cluster bootstrap/sign-flip은 PP-X의 동일 test-row
prediction artifact를 검증한 뒤에만 추가한다.

## 최종 PP-X route의 paired inference

최종 route와 robust baseline의 `y` 및 unit ID가 완전히 일치한 HUST, XJTU,
MATR2019, MATR-b2, NASA Milling에는 별도 paired inference를 적용했다. HUST에서는
PP-X가 V-REx보다 ΔR²=+.149 (95% CI [.080, .213], exact p=.00003), GroupDRO보다
ΔR²=+.246 ([.158, .350], p=.00015), monotone NN보다 +.138 ([.080, .193], p=.00003)로
우세했다. MATR2019도 세 baseline 각각에 대해 CI가 0보다 컸고 exact p≤.0254였다.
MATR-b2에서는 GroupDRO 및 monotone NN에는 유의하게 우세했지만, V-REx와의 차이
ΔR²=+.013은 유의하지 않았다(p=.336). XJTU는 ΔR² 자체는 컸어도 5개 test unit의
이질성이 커 p≥.3125로 유의하지 않았다. Milling은 validation unit이 하나이고
bootstrap 일부 재표본에서 R² 분모가 0이어서 inferential claim에서 제외한다.

재현 결과는 `results/final_route_cluster_inference_v1/results.json`에 저장한다.
