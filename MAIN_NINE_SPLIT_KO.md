# PP-X Main 9-Setting Split 요약

작성: 박진서  
근거: `experiments/extrapolation_competitors_all.py`가 로드하는 **동일 split adapter**  
재현 감사: `reproducibility/main_nine_split_audit.json` (Git 추적; `results/`는 로컬 ignore)

---

## 공통 원칙

| 원칙 | 내용 |
|------|------|
| **unit-disjoint** | train / val / test **물리 unit ID 겹침 없음** |
| **train = support 안** | train unit에서 **외삽 좌표(1D)가 train 범위 안**인 행만 |
| **val / test = hull-out** | val·test unit에서 **train support 밖** (더 열화·더 극단 조건) |
| **val-only 선택** | executor·하이퍼·epoch는 validation만; test는 frozen forward |
| **1D hull-out** | 전체 feature-space convex hull 밖이 **아님** — **사전 선언 1D 좌표** 기준 |

**행(row)** = PP-X Final benchmark adapter가 실제 학습·평가에 쓰는 window/시점 수.  
배터리 3종(Sunwoda/RWTH/MICH)은 8-step causal window + stride 2, unit당 최대 250 window(seed 42).

---

## 1. HUST (화중과기대 배터리)

**외삽 좌표:** `capacity_ah`  
**소스:** `ca-css-ncmapss/run_affine_tail_external_three.py::prepare_hust`

| Split | Unit | 조건 | 행 수 |
|-------|------|------|------:|
| **Train** | protocol **1–6** (45 cells) | capacity **≥ 1.05 Ah** | **11,250** |
| **Val** | protocol **7–8** (16 cells) | capacity **< 1.05 Ah** (train_min) | **2,560** |
| **Test** | protocol **9–10** (16 cells) | capacity **< 1.05 Ah** | **7,775** |

- unit 분리 = **충전 프로토콜** (같은 protocol 내 셀은 train에만)
- Y = proxy EOL 기반 RUL (우측 검열; 실제 0.88 Ah 도달 전 종료 셀 다수)

---

## 2. Virkler (알루미늄 피로균열)

**외삽 좌표:** `crack_length_mm`  
**소스:** `ca-css-ncmapss/run_affine_tail_external_three.py::prepare_virkler`

| Split | Unit | 조건 | 행 수 |
|-------|------|------|------:|
| **Train** | 48 specimens (**seed 42**) | crack **≤ 33 mm** | **240** |
| **Val** | 10 specimens | crack **> 33 mm** | **20** |
| **Test** | 10 specimens | crack **> 33 mm** | **20** |

- 배터리가 아님; 균열 성장 discrete checkpoint
- Val/test = **처음 보는 시편** + train이 못 본 **긴 균열(39, 49.8 mm 쪽)**

---

## 3. NASA PCoE battery (4셀)

**외삽 좌표:** `health_phi = (capacity−1.4)/(initial−1.4)`  
**소스:** `ca-css-ncmapss/run_affine_tail_external_nasa_health_v2.py::prepare_folds`

| Split | Unit | 조건 | 행 수 |
|-------|------|------|------:|
| **Train** | **2 cells / fold** × 4 folds | health_phi **≥ 0.5** | **488** (fold 합) |
| **Val** | **1 cell / fold** × 4 folds | health_phi **< fold train_min** | **254** |
| **Test** | **1 cell / fold** × 4 folds | health_phi **< fold train_min** | **255** |

- Cells: **B0005, B0006, B0007, B0018** — 4-fold LOOCV
- 보고 R² = 4 fold test **pooled** (행 수는 fold 합계)

---

## 4. Sunwoda

**외삽 좌표:** 8-step window endpoint `health_last` (mAh)  
**소스:** `ca-css-ncmapss/pae_boundary_realdata.py` + `prepare_battery`

| Split | Unit | 조건 | 행 수 |
|-------|------|------|------:|
| **Train** | 25°C **7 cells** `[1025…7025]` | endpoint health **> train-q25**; early health ≥ threshold | **1,312** |
| **Val** | 25°C **2 cells** `[8025, 9025]` | endpoint health **< train-q25** | **141** |
| **Test** | **35°C 9 cells** `[10035…18035]` | health **< train support cutoff (980.7 mAh)** | **1,468** |

- EOL boundary: **880 mAh** first crossing
- Test = **unseen cell + 35°C domain shift + late health tail**

---

## 5. RWTH

**외삽 좌표:** 8-step window endpoint health (첫 관측 용량 정규화)  
**소스:** `pae_boundary_realdata.py` + `prepare_battery`

| Split | Unit | 조건 | 행 수 |
|-------|------|------|------:|
| **Train** | cell ID **2–24** (23) | endpoint health **> train-q25**; early ≥ threshold | **2,295** |
| **Val** | cell ID **25–32** (8) | endpoint health **< train-q25** | **231** |
| **Test** | cell ID **33–40** (8) | health **< train support cutoff (0.9000)** | **1,859** |

- nominal EOL: relative **0.8** crossing

---

## 6. MICH

**외삽 좌표:** 8-step window endpoint health (nominal capacity 정규화)  
**소스:** `pae_boundary_realdata.py` + `prepare_battery`

| Split | Unit | 조건 | 행 수 |
|-------|------|------|------:|
| **Train** | cell ID **1–18** (17 valid window) | endpoint health **> train-q25**; early ≥ threshold | **1,439** |
| **Val** | cell ID **19–24** (6) | endpoint health **< train-q25** | **259** |
| **Test** | cell ID **25–32** (8) | health **< train support cutoff (0.8708)** | **202** |

- Y = **공개 MICH life label** (단순 80% crossing 재탐색 아님)
- dual-scale executor 반례 setting

---

## 7. MATR 2019

**외삽 좌표:** 8-step window `Q_last`  
**소스:** `experiments/matr_2019_latent_confirmatory.py`

| Split | Unit | 조건 | 행 수 |
|-------|------|------|------:|
| **Train** | cells **0–25** (26) | Q_last **> q25 (0.9873)**; boundary=**0.9873** | **17,003** |
| **Val** | cells **26–30, 32–34** (8) | Q_last **< boundary** | **1,306** |
| **Test** | cells **35–44** (10) | Q_last **< boundary** | **2,235** |

- 유효 44 cells / 원본 45; cell 31 excluded
- Y = 각 셀 기록 **마지막 cycle − 현재 cycle**

---

## 8. MATR batch 2

**외삽 좌표:** 8-step window `Q_last`  
**소스:** `experiments/matr_batch2_confirmatory.py`

| Split | Unit | 조건 | 행 수 |
|-------|------|------|------:|
| **Train** | cells **0–29** (30) | Q_last **> q25 (0.9678)**; boundary=**0.9678** | **11,552** |
| **Val** | cells **30–38** (9) | Q_last **< boundary** | **673** |
| **Test** | cells **39–47** (9) | Q_last **< boundary** | **733** |

- locked 48-cell batch (`2017-06-30`)

---

## 9. N-CMAPSS DS02-006

**외삽 좌표:** **TRA** (throttle / operating condition) — **RUL tail 아님**  
**소스:** `ca-css-ncmapss/ncmapss_tra_quantile_split.py::make_tra_hard_split`  
**PP-X route:** validation-selected multiscale **history** executor

| Split | Unit (engine) | 조건 (TRA) | 행 수 |
|-------|---------------|------------|------:|
| **Train** | **2, 5, 10, 16, 18** | **TRA ≤ 79.10** (train-pool q70) | **5,239** |
| **Val** | **2, 5, 10, 16, 18, 20** | **79.10 < TRA ≤ 81.74** (q70–q90) | **1,810** |
| **Test** | **11, 14, 15** (unseen) | **TRA > 81.74** (q90) | **159** |

- window: **30-step** causal sequence; TRA = **창 마지막 timestep**
- **NOT main test:** `RUL ≤ 50` band (`hard_extrap_late`, 130 rows) — contrast만, **0.937 근거 아님**
- **DS03** prospective = **별도 protocol** (이 표와 다름)

---

## 한 장 비교 (Test만)

| Dataset | Test unit 요약 | Test 조건 (1D) | Test rows |
|---------|----------------|----------------|----------:|
| HUST | protocol 9–10, 16 cells | capacity < train_min | 7,775 |
| Virkler | 10 specimens | crack > 33 mm | 20 |
| NASA battery | 1 cell × 4 folds pooled | health_phi < fold train_min | 255 |
| Sunwoda | 9 cells @ **35°C** | health < 980.7 mAh | 1,468 |
| RWTH | cells 33–40 | health < 0.900 | 1,859 |
| MICH | cells 25–32 | health < 0.871 | 202 |
| MATR 2019 | cells 35–44 | Q_last < 0.987 | 2,235 |
| MATR b2 | cells 39–47 | Q_last < 0.968 | 733 |
| N-CMAPSS | engines 11, 14, 15 | TRA > 81.74 | 159 |

---

## 발표용 한 줄

> **Train** = 고른 unit의 support **안**. **Val/Test** = **다른 unit** + train이 **못 본 1D 좌표 바깥**.  
> 배터리·균열 = **health/crack/capacity tail**; N-CMAPSS = **unseen engine + high TRA**.

---

## 감사 JSON 재생성

```bash
cd pp-extrapolation
/opt/anaconda3/bin/python3.12 - <<'PY'
# append_executor_formula_appendix.py 상단과 동일 경로 설정 후
# results/main_nine_split_audit.json 생성 스크립트 — 저장소 내 최신 audit 참조
PY
```

또는 `results/main_nine_split_audit.json` 직접 확인.

**관련 문서:** `MODEL_AND_SPLIT_KO.md`, `docs/build_ppx_dataset_complete_guide.py`
