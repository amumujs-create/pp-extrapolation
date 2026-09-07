> 해석 정정: 이 표는 원래 PP의 개선 여부를 판정하는 실험이 아니다. `pp_adaptive`는 원래 PP의 loss와 seed별 탐색을 유지한 채 affine만 해제한 모델이 아니라, extended benchmark의 정규화 없는 PP를 바탕으로 한다. 모든 군은 150 epoch / patience 25이며, FT 보정 두 군은 기준 FT 설정을 재사용하고 각각의 구조에 맞춘 별도 탐색을 하지 않았다. 따라서 수치는 해당 제한된 설정의 관측 결과로 보존하되, “원래 PP의 affine 동결 해제가 실패했다” 또는 “충분히 튜닝한 개선안이 FT보다 열등하다”는 결론에는 사용하지 않는다.

# Affine 적응 개선 시도 결과

MATR 2019 사후 개발 실험. 이번 세 개선 시도는 FT-Transformer보다 좋은 성능을 얻지 못했다. 테스트 결과로 추가 설정을 고르지 않았고, 기존 모델 결과는 보존했다.

| 모델 | ensemble pooled R² | single mean ± SD |
|---|---:|---:|
| 기준 FT-Transformer | 0.344 | 0.303 ± 0.037 |
| pp_adaptive | -0.160 | -0.255 ± 0.070 |
| ft_affine_frozen | -0.337 | -0.468 ± 0.185 |
| ft_affine_adaptive | -0.254 | -0.497 ± 0.114 |

PP affine 동결 해제만으로는 개선되지 않았다. FT에 affine 기준 함수를 추가한 잔차 구성도 악화됐다. 따라서 이 설정에서 affine prior가 강한 NN의 외삽 성능을 개선한다는 가설은 지지되지 않는다. 모든 affine/residual 방법이 무효라는 증명은 아니다.

통제 범위: PP는 이전 공통 9개 설정 탐색을 반복했다. FT 두 군은 원 FT의 validation 선택 설정을 그대로 사용해 추가 탐색 없이 5 seed 재학습했다. 모든 군은 최대 150 epochs, patience 25이며 이전 확증 PP의 300 epochs 실험과 동일하지 않다.

FT 변형은 affine(x)+FT(x)-FT_initial(x)이다. 초기 함수 복사본은 동결되므로 affine에서 시작하지만, 단순 FT보다 forward 계산과 저장 공간이 추가된다. 이들은 기존 PP 모델과 구분되는 새 연구 변형이다.

다음 판단: 강한 NN 대비 우월성 주장으로 제출하기에는 근거가 부족하다. 원 PP·FT의 충분한 공통 300-epoch 재튜닝을 수행하려면 개발 cohort와 평가 cohort 및 탐색 예산을 먼저 재정의해야 한다. 이미 본 MATR 점수 목표를 맞출 때까지 구조를 바꾸는 것은 확증이 아니다.

검증: 3개 모델 초기 affine 일치/역전파 smoke check, 15개 저장 예측으로 ensemble 지표 재계산 일치.
