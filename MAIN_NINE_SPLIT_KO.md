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
| **Train** | 25°C **7 cells** `[1025…7025]` | endpoint health **> 1011.6 mAh** (train-q25) | **1,312** |
| **Val** | 25°C **2 cells** `[8025, 9025]` | endpoint health **< 1011.6 mAh** (train-q25) | **141** |
| **Test** | **35°C 9 cells** `[10035…18035]` | point health **< 980.7 mAh** (early_health_min) | **1,468** |

- **Cutoff map:** train/val q25 = **1011.625 mAh**; test support = **980.7 mAh** (서로 다른 규칙 — test는 `prepare_dataset` frame cutoff)
- EOL boundary: **880 mAh** first crossing
- Test = **unseen cell + 35°C domain shift + late health tail**

---

## 5. RWTH

**외삽 좌표:** 8-step window endpoint health (첫 관측 용량 정규화)  
**소스:** `pae_boundary_realdata.py` + `prepare_battery`

| Split | Unit | 조건 | 행 수 |
|-------|------|------|------:|
| **Train** | cell ID **2–24** (23) | endpoint health **> 0.9134** (train-q25) | **2,295** |
| **Val** | cell ID **25–32** (8) | endpoint health **< 0.9134** (train-q25) | **231** |
| **Test** | cell ID **33–40** (8) | point health **< 0.9000** (early_health_min) | **1,859** |

- nominal EOL: relative **0.8** crossing

---

## 6. MICH

**외삽 좌표:** 8-step window endpoint health (nominal capacity 정규화)  
**소스:** `pae_boundary_realdata.py` + `prepare_battery`

| Split | Unit | 조건 | 행 수 |
|-------|------|------|------:|
| **Train** | cell ID **1–18** (17 valid window) | endpoint health **> 0.9233** (train-q25) | **1,439** |
| **Val** | cell ID **19–24** (6) | endpoint health **< 0.9233** (train-q25) | **259** |
| **Test** | cell ID **25–32** (8) | point health **< 0.8708** (early_health_min) | **202** |

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
| Sunwoda | 9 cells @ **35°C** | point health < **980.7 mAh** | 1,468 |
| RWTH | cells 33–40 | point health < **0.9000** | 1,859 |
| MICH | cells 25–32 | point health < **0.8708** | 202 |
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
PYTHONPATH=src:experiments:../ca-css-ncmapss \
  /opt/anaconda3/bin/python3.12 experiments/build_main_nine_split_audit_v1.py
```

Git 정본: `reproducibility/main_nine_split_audit.json` (cutoff 실수 + q-label 매핑 + rationale 포함).

**관련 문서:** `MODEL_AND_SPLIT_KO.md`, `docs/build_ppx_dataset_complete_guide.py`
## Cutoff map (Git audit, unified units)

작성: 박진서 · 재생성: `python experiments/build_main_nine_split_audit_v1.py`

배터리 3종은 **train/val = train-q25 (endpoint)** 와 **test = early_health_min** 이 다른 두 cutoff를 쓴다. 표에는 둘 다 실수로 적었다.

| Dataset | Coordinate | Train | Val | Test | q / label → value |
|---------|------------|-------|-----|------|-------------------|
| HUST | capacity_ah (Ah) | protocol 1–6, capacity ≥ 1.05 Ah; hull axis capacity ≥ 1.0500 Ah | protocol 7–8, capacity < 1.0500 Ah | protocol 9–10, capacity < 1.0500 Ah | train_support_min={"value": 1.050008069444443, "unit": "Ah", "train": "≥ 1.0500", "val": "< 1.0500", "test": "< 1.0500", "note": "train_min = min capacity among train rows after protocol filter"} |
| Virkler | crack_length_mm (mm) | crack length ≤ 33.0 mm | crack length > 33.0 mm | crack length > 33.0 mm | train_support_max={"value": 33.0, "unit": "mm", "train": "≤ 33.0", "val": "> 33.0", "test": "> 33.0"} |
| NASA PCoE battery | health_phi = (capacity−1.4)/(initial−1.4) (dimensionless) | health_phi ≥ 0.5 | health_phi < fold_train_min (see cutoffs.fold_train_min) | health_phi < fold_train_min (4 folds pooled) | train_floor={"value": 0.5, "unit": "health_phi", "train": "≥ 0.5"}; fold_train_min={"unit": "health_phi", "per_fold": [{"test_cell": "B0005", "validation_cell": "B0006", "fold_train_min_health_phi": 0.5003218054771423}, {"test_cell": "B0006", "validation_cell": "B0007", "fold_train_min_health_phi": 0.5003218054771423}, {"test_cell": "B0007", "validation_cell": "B0018", "fold_train_min_health_phi": 0.509838879108429}, {"test_cell": "B0018", "validation_cell": "B0005", "fold_train_min_health_phi": 0.503444254398346}], "val": "health_phi < fold_train_min", "test": "health_phi < fold_train_min"} |
| Sunwoda | 8-step window endpoint health (feature x[:,−1,0]) (mAh) | endpoint health > 1011.6250 mAh (train-q25 (train endpoint health)) | endpoint health < 1011.6250 mAh (train-q25 (train endpoint health)) | point health < 980.7000 mAh (early_health_min (support cutoff on source cells)) | train_val_endpoint_q25={"quantile": 0.25, "population": "train split window endpoints", "value": 1011.6249847412109, "unit": "mAh", "train": "> 1011.6249847412109", "val": "< 1011.6249847412109"}; test_support_early_health_min={"value": 980.7, "unit": "mAh", "test": "< 980.7", "derivation": "min health among eligible early-life windows on train∪val units; boundary=880.0, threshold=980.6500"} |
| RWTH | 8-step window endpoint health (feature x[:,−1,0]) (normalized health) | endpoint health > 0.9134 (train-q25 (train endpoint health)) | endpoint health < 0.9134 (train-q25 (train endpoint health)) | point health < 0.9000 (early_health_min (support cutoff on source cells)) | train_val_endpoint_q25={"quantile": 0.25, "population": "train split window endpoints", "value": 0.9133632481098175, "unit": "normalized health", "train": "> 0.9133632481098175", "val": "< 0.9133632481098175"}; test_support_early_health_min={"value": 0.9000159890976354, "unit": "normalized health", "test": "< 0.9000159890976354", "derivation": "min health among eligible early-life windows on train∪val units; boundary=0.8, threshold=0.9000"} |
| MICH | 8-step window endpoint health (feature x[:,−1,0]) (normalized health) | endpoint health > 0.9233 (train-q25 (train endpoint health)) | endpoint health < 0.9233 (train-q25 (train endpoint health)) | point health < 0.8708 (early_health_min (support cutoff on source cells)) | train_val_endpoint_q25={"quantile": 0.25, "population": "train split window endpoints", "value": 0.923305094242096, "unit": "normalized health", "train": "> 0.923305094242096", "val": "< 0.923305094242096"}; test_support_early_health_min={"value": 0.8707627118644069, "unit": "normalized health", "test": "< 0.8707627118644069", "derivation": "min health among eligible early-life windows on train∪val units; boundary=0.8, threshold=0.8708"} |
| MATR 2019 | Q_last (8-step window endpoint) (normalized Q) | Q_last > 0.9873 (train-q25) | Q_last < 0.9873 | Q_last < 0.9873 | train_q25_boundary={"quantile": 0.25, "population": "train cells 0–25 endpoint Q", "value": 0.9873319864273071, "unit": "Q_last", "train": "> 0.9873", "val": "< 0.9873", "test": "< 0.9873"} |
| MATR batch 2 | Q_last (normalized Q) | Q_last > 0.9678 (train-q25) | Q_last < 0.9678 | Q_last < 0.9678 | train_q25_boundary={"quantile": 0.25, "value": 0.9678474068641663, "unit": "Q_last", "train": "> 0.9678", "val": "< 0.9678", "test": "< 0.9678", "note": "q25 raw=0.9678; boundary=min train coordinate=0.9678"} |
| N-CMAPSS DS02-006 | TRA at window end (percent throttle) | TRA ≤ 79.10 | 79.10 < TRA ≤ 81.74 | TRA > 81.74 | train_q70={"value": 79.1009979248047, "train": "TRA ≤ 79.10"}; val_band={"low": 79.1009979248047, "high": 81.73770141601562, "val": "79.10 < TRA ≤ 81.74"}; test_q90={"value": 81.73770141601562, "test": "TRA > 81.74"} |

### Split rationale (one line each)

- **HUST:** Physical units are charging protocols. Train keeps high-capacity regimes; val/test use later protocols with capacity below the train support floor (1.0500 Ah), giving 100% 1D hull-out on capacity_ah.
- **Virkler:** Specimens are unit-disjoint (seed 42). Train stays inside the declared crack support; val/test specimens extrapolate beyond 33 mm, the train maximum.
- **NASA PCoE battery:** Four-fold LOOCV over cells B0005–B0018. Train keeps phi≥0.5; val/test cells use health below that fold's train minimum on the 1D phi axis.
- **Sunwoda:** 25°C train/val cells vs 35°C test cells. Train/val split by train-q25 on 8-step endpoint health; test cells use early_health_min (980.7 mAh) for deeper tail + temperature shift.
- **RWTH:** Unit IDs 2–24 / 25–32 / 33–40. Train/val use train-q25 on normalized endpoint health; test uses early_health_min 0.9000 on unseen IDs.
- **MICH:** Unit IDs 1–18 / 19–24 / 25–32 with published MICH life labels. Same q25 vs early_health_min split as RWTH; test cutoff 0.8708.
- **MATR 2019:** Locked 45-cell batch: q25 on train endpoints defines one boundary for all splits.
- **MATR batch 2:** Same q25 boundary logic as MATR 2019 on the locked batch-2 cohort.
- **N-CMAPSS DS02-006:** Unseen engines at high TRA; quantiles q70/q90 fit on train-pool TRA only. Main test is hard_extrap (159 rows), not RUL≤50 contrast band.

