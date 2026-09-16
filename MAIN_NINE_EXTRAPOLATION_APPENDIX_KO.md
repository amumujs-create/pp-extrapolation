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
| Battery (protocol shift) | HUST | capacity_ah (Ah) | 45 units · 11250 rows · protocol 1–6, capacity ≥ 1.05 Ah; hull axis capacity ≥ 1.0500 Ah | 16 units · 2560 rows · protocol 7–8, capacity < 1.0500 Ah | 16 units · 7775 rows · protocol 9–10, capacity < 1.0500 Ah | 100.0 | 1.608 | 1.953 |
| Fatigue crack growth | Virkler | crack_length_mm (mm) | 48 units · 240 rows · crack length ≤ 33.0 mm | 10 units · 20 rows · crack length > 33.0 mm | 10 units · 20 rows · crack length > 33.0 mm | 100.0 | 1.623 | 3.190 |
| Battery (cell LOO) | NASA PCoE battery | health_phi = (capacity−1.4)/(initial−1.4) (dimensionless) | 2/fold×4 units · 488 rows · health_phi ≥ 0.5 | 1/fold×4 units · 254 rows · health_phi < fold_train_min (see cutoffs.fold_train_min) | 1/fold×4 units · 255 rows · health_phi < fold_train_min (4 folds pooled) | 100.0 | 2.009 | 2.009 |
| Battery (unseen cell) | Sunwoda | 8-step window endpoint health (feature x[:,−1,0]) (mAh) | 7 units · 1312 rows · endpoint health > 1011.6250 mAh (train-q25 (train endpoint health)) | 2 units · 141 rows · endpoint health < 1011.6250 mAh (train-q25 (train endpoint health)) | 9 units · 1468 rows · point health < 980.7000 mAh (early_health_min (support cutoff on source cells)) | 100.0 | 4.421 | 6.827 |
| Battery (unseen cell) | RWTH | 8-step window endpoint health (feature x[:,−1,0]) (normalized health) | 23 units · 2295 rows · endpoint health > 0.9134 (train-q25 (train endpoint health)) | 8 units · 231 rows · endpoint health < 0.9134 (train-q25 (train endpoint health)) | 8 units · 1859 rows · point health < 0.9000 (early_health_min (support cutoff on source cells)) | 100.0 | 2.870 | 5.238 |
| Battery (unseen cell) | MICH | 8-step window endpoint health (feature x[:,−1,0]) (normalized health) | 18 units · 1439 rows · endpoint health > 0.9233 (train-q25 (train endpoint health)) | 6 units · 259 rows · endpoint health < 0.9233 (train-q25 (train endpoint health)) | 8 units · 202 rows · point health < 0.8708 (early_health_min (support cutoff on source cells)) | 100.0 | 3.426 | 5.929 |
| Battery (capacity fade) | MATR 2019 | Q_last (8-step window endpoint) (normalized Q) | 26 units · 17003 rows · Q_last > 0.9873 (train-q25) | 8 units · 1306 rows · Q_last < 0.9873 | 10 units · 2235 rows · Q_last < 0.9873 | 100.0 | 5.554 | 9.114 |
| Battery (capacity fade) | MATR batch 2 | Q_last (normalized Q) | 30 units · 11552 rows · Q_last > 0.9678 (train-q25) | 9 units · 673 rows · Q_last < 0.9678 | 9 units · 733 rows · Q_last < 0.9678 | 100.0 | 1.898 | 2.757 |
| Aero engine (unseen engine × TRA) | N-CMAPSS DS02-006 | TRA at window end (percent throttle) | [2, 5, 10, 16, 18] units · 5239 rows · TRA ≤ 79.10 | [2, 5, 10, 16, 18, 20] units · 1810 rows · 79.10 < TRA ≤ 81.74 | [11, 14, 15] units · 159 rows · TRA > 81.74 | 100.0 | 0.228 | 1.418 |

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
