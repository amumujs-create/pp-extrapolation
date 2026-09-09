# Misata machine-degradation untouched PP 결과

## 판정

**절대 예측 성공, PP 우월성 확증 실패.** 프로토콜 커밋 `23f3827` 뒤
526-KB 공식 archive를 다운로드했고 publisher의 80/20 machine split을 그대로
사용했다. 숨은 damage, RUL, failure flag는 입력에서 제외했다.

| 모델 | test pooled R² | RMSE | MAE |
|---|---:|---:|---:|
| tuned PP | 0.833 | 8.481 | 6.425 |
| tuned matched MLP | **0.854** | **7.922** | **6.005** |

20개 test machine의 paired bootstrap `MLP RMSE - PP RMSE`는 -0.553,
95% CI **[-1.049, -0.099]**였다. PP는 높은 양의 R²를 확보했지만 MLP보다
유의하게 낮으므로 사전 성공 규칙을 통과하지 못했다.

개발 validation은 PP에 tail 30%, width 16, learning rate 1e-3, weight decay
2.0을 골랐다. 이 controlled generator에서는 센서가 단일 latent damage에서
직접 생성되어 plain NN이 이미 매우 잘 맞고, frozen affine 경로가 추가 bias가
됐다. 다음 구조 개선은 prior가 무익할 때 동일 NN 경로를 정확히 포함하는
safety-continuation gate다.

이 자료는 100개 완전 run-to-failure를 가진 controlled synthetic evidence이며
실기계 외부 확증으로 표현하지 않는다.

재현 코드: `experiments/misata_machine_untouched.py`

결과: `results/misata_machine_untouched_v1/results.json`
