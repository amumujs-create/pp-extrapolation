# Main 9 — Extrapolation distance appendix

작성: 박진서

PPT Fig. 7 schematic의 수치 근거. **Test split** 기준.

- **Outside-support %** = `outside_fraction` × 100 (1D 선언 좌표, train convex hull 밖)
- **Extrapolation distance** = `distance_median` (train SD 단위 hull 밖 거리)
- **Full-feature distance** = train 표준화 full feature 공간 NN median 거리

좌표 정의: `predeclared first PP extrapolation coordinate, standardized by train SD`

재생성:

```bash
cd pp-extrapolation
PYTHONPATH=src:experiments:../ca-css-ncmapss python experiments/all_dataset_hull_audit.py
PYTHONPATH=src:experiments:../ca-css-ncmapss python experiments/export_main_nine_extrapolation_appendix_v1.py
```

Git-tracked JSON: `reproducibility/all_dataset_hull_audit_v1/results.json`

| Domain | Dataset | Axis (1D hull) | Train | Val | Test | Outside % | Hull dist. (med.) | Full-NN dist. |
|--------|---------|----------------|-------|-----|------|----------:|------------------:|--------------:|
| Battery (protocol shift) | HUST | capacity_ah (1D hull-out) | 45 units · 11250 rows · protocol 1–6 (45 units), capacity ≥ 1.05 Ah; train_min=1.0500 Ah | 16 units · 2560 rows · protocol 7–8 (16 units), capacity < 1.0500 Ah | 16 units · 7775 rows · protocol 9–10 (16 units), capacity < 1.0500 Ah | 100.0 | 1.608 | 1.953 |
| Fatigue crack growth | Virkler | crack_length_mm (1D hull-out) | 48 units · 240 rows · 48 specimens (seed 42), crack ≤ 33 mm | 10 units · 20 rows · 10 specimens, crack > 33 mm | 10 units · 20 rows · 10 specimens, crack > 33 mm | 100.0 | 1.623 | 3.190 |
| Battery (cell LOO) | NASA PCoE battery | health_phi = (capacity−1.4)/(initial−1.4) | 2/fold×4 folds units · 488 rows · LOOCV: 2 cells/fold, health_phi ≥ 0.5 | 1/fold×4 folds units · 254 rows · 1 cell/fold, health_phi < fold train_min | 1/fold×4 folds units · 255 rows · 1 test cell/fold, health_phi < fold train_min; 4 folds pooled for reporting | 100.0 | 2.009 | 2.009 |
| Battery (unseen cell) | Sunwoda | health_last on 8-step window (1D hull-out) | 7 units · 1312 rows · 25°C cells [1025, 2025, 3025, 4025, 5025, 6025, 7025] (7 units); 8-step endpoint health > train-q25; early health ≥ threshold | 2 units · 141 rows · 25°C cells [8025, 9025] (2 units); endpoint health < train-q25 | 9 units · 1468 rows · 35°C cells [10035, 11035, 12035, 13035, 14035, 15035, 16035, 17035, 18035] (9 units); health < train support cutoff (980.7000) | 100.0 | 4.421 | 6.827 |
| Battery (unseen cell) | RWTH | health_last on 8-step window (1D hull-out) | 23 units · 2295 rows · cell ID 2–24 (23 units); endpoint health > train-q25; early health ≥ threshold | 8 units · 231 rows · cell ID 25–32 (8 units); endpoint health < train-q25 | 8 units · 1859 rows · cell ID 33–40 (8 units); health < train support cutoff (0.9000) | 100.0 | 2.870 | 5.238 |
| Battery (unseen cell) | MICH | health_last on 8-step window (1D hull-out) | 17 units · 1439 rows · cell ID 1–18 (18 units); endpoint health > train-q25; early health ≥ threshold | 6 units · 259 rows · cell ID 19–24 (6 units); endpoint health < train-q25 | 8 units · 202 rows · cell ID 25–32 (8 units); health < train support cutoff (0.8708) | 100.0 | 3.426 | 5.929 |
| Battery (capacity fade) | MATR 2019 | Q_last (1D hull-out) | 26 units · 17003 rows · cells [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] (26); Q_last > q25 (0.9873); boundary=0.9873 | 8 units · 1306 rows · cells [26, 27, 28, 29, 30, 32, 33, 34] (8); Q_last < boundary | 10 units · 2235 rows · cells [35, 36, 37, 38, 39, 40, 41, 42, 43, 44] (10); Q_last < boundary | 100.0 | 5.554 | 9.114 |
| Battery (capacity fade) | MATR batch 2 | Q_last (1D hull-out) | 30 units · 11552 rows · cells 0–29 (30); Q_last > q25 (0.9678); boundary=0.9678 | 9 units · 673 rows · cells 30–38 (9); Q_last < boundary | 9 units · 733 rows · cells 39–47 (9); Q_last < boundary | 100.0 | 1.898 | 2.757 |
| Aero engine (unseen engine × TRA) | N-CMAPSS DS02-006 | TRA (throttle / operating condition) | [2, 5, 10, 16, 18] units · 5239 rows · unit∈[2, 5, 10, 16, 18], TRA <= 79.10 | [2, 5, 10, 16, 18, 20] units · 1810 rows · unit∈[2, 5, 10, 16, 18, 20], 79.10 < TRA <= 81.74 | [11, 14, 15] units · 159 rows · unit∈[11, 14, 15], TRA > 81.74 | 100.0 | 0.228 | 1.418 |

## Compact (PPT right panel)

| Dataset | Outside % | Extrapolation distance | Full-feature distance |
|---------|----------:|-----------------------:|----------------------:|
| HUST | 100.0 | 1.608 | 1.953 |
| Virkler | 100.0 | 1.623 | 3.190 |
| NASA PCoE battery | 100.0 | 2.009 | 2.009 |
| Sunwoda | 100.0 | 4.421 | 6.827 |
| RWTH | 100.0 | 2.870 | 5.238 |
| MICH | 100.0 | 3.426 | 5.929 |
| MATR 2019 | 100.0 | 5.554 | 9.114 |
| MATR batch 2 | 100.0 | 1.898 | 2.757 |
| N-CMAPSS DS02-006 | 100.0 | 0.228 | 1.418 |
