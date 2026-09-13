# PP-X vs Engression 승패 설명변수 (v1)

> Equal-budget ΔR² = R²_PP-X − R²_Engression 과 train/val-only 네 설명변수의
> Spearman 상관. test Y는 설명변수 계산에 사용하지 않았다.

## 판정 요약

단순 가설(`conditional_dispersion↑ ⇒ ΔR²↓`, `boundary_informativeness↑ ⇒ ΔR²↑`)은
**전 9-setting 순위상관으로는 확증되지 않았다.** 다만 패턴은 더 세분화된다.

### vs ΔR² (전체 9)

- `unit_heterogeneity`: Spearman ρ=0.767, p=0.0159 (가설: negative_with_delta_r2, n=9)
- `conditional_dispersion`: Spearman ρ=-0.050, p=0.8984 (가설: negative_with_delta_r2, n=9)
- `boundary_informativeness`: Spearman ρ=-0.250, p=0.5165 (가설: positive_with_delta_r2, n=9)
- `regime_heterogeneity`: Spearman ρ=0.048, p=0.9108 (가설: negative_with_delta_r2, n=8)

### vs Engression R² (전체 9)

- `unit_heterogeneity`: ρ=-0.733, p=0.0246, n=9
- `conditional_dispersion`: ρ=-0.183, p=0.6368, n=9
- `boundary_informativeness`: ρ=0.000, p=1.0000, n=9
- `regime_heterogeneity`: ρ=0.071, p=0.8665, n=8

### 민감도: Engression R²≥0 인 7-setting만 (MICH·MATR2019 제외)

- `unit_heterogeneity`: ρ=0.643, p=0.1194, n=7
- `conditional_dispersion`: ρ=-0.357, p=0.4316, n=7
- `boundary_informativeness`: ρ=-0.143, p=0.7599, n=7
- `regime_heterogeneity`: ρ=-0.086, p=0.8717, n=6

## 설정별 표

| Setting | PP-X R² | Engression R² | ΔR² | unit_het | cond_disp | bound_info | regime_het |
|---|---:|---:|---:|---:|---:|---:|---:|
| HUST | 0.958 | 0.871 | +0.087 | 0.439 | 0.126 | 0.745 | 0.095 |
| Virkler | 0.888 | 0.863 | +0.025 | 0.172 | 0.053 | 0.873 | 1.014 |
| NASA | 0.584 | 0.583 | +0.000 | 0.296 | 0.332 | 0.849 | 0.150 |
| SUNWODA | 0.939 | 0.566 | +0.373 | 0.348 | 0.204 | 0.826 | nan |
| RWTH | 0.878 | 0.532 | +0.347 | 0.561 | 0.331 | 0.600 | 0.540 |
| MICH | 0.751 | -1.580 | +2.332 | 0.516 | 0.309 | 0.423 | 0.780 |
| MATR2019 | 0.466 | -1.196 | +1.662 | 0.610 | 0.457 | 0.381 | 0.914 |
| MATR-b2 | 0.862 | 0.739 | +0.123 | 0.445 | 0.277 | 0.201 | 0.229 |
| N-CMAPSS | 0.937 | 0.932 | +0.005 | 0.244 | 0.819 | 0.000 | 9.020 |

## 정의 (사전 고정)

1. **unit_heterogeneity**: health 분위 구간 안에서 unit-mean RUL의 표준편차 중앙값 / 전역 std(Y).
2. **conditional_dispersion**: 표준화 X kNN(k=25)의 이웃 Y 표준편차 중앙값 / 전역 std(Y).
3. **boundary_informativeness**: 명시적 failure boundary 거리로 Y를 선형 설명한 R². N-CMAPSS는 스칼라 EOL이 없어 Y~x0(TRA) 대용이며 `has_explicit_boundary=false`.
4. **regime_heterogeneity**: regime별 Y~health 기울기의 MAD / |전역 기울기|. HUST=protocol, Sunwoda=온도, N-CMAPSS=TRA 삼분위, 그 외 health 삼분위/코호트.

## 해석

1. **N-CMAPSS 단독 패턴**은 가설과 맞다: conditional_dispersion이 9개 중 최대(0.82), 명시적 boundary informativeness≈0, ΔR²≈0(+0.005)로 Engression이 사실상 동률.
2. **전역 Spearman**에서는 unit_heterogeneity가 ΔR²와 **양의** 상관(ρ≈0.77). 즉 ‘이질성↑ → Engression 유리’가 아니라, 같은 health 근처 unit RUL이 흩어질수록 Engression이 크게 무너지고 PP-X 이득(ΔR²)이 커지는 쪽에 가깝다 (MICH·MATR2019 Engression 음수 R²가 순위을 당김).
3. boundary_informativeness의 전역 상관은 가설 반대 부호·비유의. Sunwoda·Virkler처럼 boundary R²가 높아도 ΔR² 크기는 제각각이고, MATR-b2는 boundary R²가 낮은데도 Engression이 양수라 ΔR²가 작다.
4. Engression R²≥0 인 7개만 보면 상관 부호·크기가 다시 흔들린다 → **n=9 정적 설명변수만으로 승패 법칙을 확정할 수 없다.**
5. 논문용 문장 후보: “분포적 이질성이 큰 N-CMAPSS에서는 Engression이 경쟁적이지만, setting-level 정적 descriptor만으로 PP-X−Engression 이득을 보편 예측하는 사전 gate는 아직 성립하지 않는다. 특히 unit-level health 근처 RUL 이질성은 Engression 붕괴와 함께 PP-X 상대이득을 키우는 방향으로 관측됐다.”

## 해석 경계

- n=9 setting이라 상관계수는 **탐색적**이다. p-value를 확증으로 쓰지 않는다.
- Sunwoda train/val이 동일 온도(25°C)라 regime_heterogeneity는 NaN.
- N-CMAPSS regime_het는 TRA 삼분위 기울기 차이로 매우 큼(≈9) — 스케일 이상치.

- Artifact: `results/ppx_engression_winloss_descriptors_v1/results.json`
- Runner: `experiments/ppx_engression_winloss_descriptors.py`

