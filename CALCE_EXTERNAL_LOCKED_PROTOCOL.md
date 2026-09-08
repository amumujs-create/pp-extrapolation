# Locked external evaluation: CALCE CS2/CX2

Status: frozen before downloading `CALCE.zip` or inspecting processed outcomes.  
Primary source: CALCE Battery Research Group.  
Distribution used: BatteryLife processed release on Zenodo record 17958489.

1. Use every readable CS2/CX2 cell containing at least 20 finite cycle-capacity observations.
2. Sort cell identifiers lexicographically; allocate first 60% train, next 20% validation, remainder test. Do not reorder by lifetime or outcome.
3. Restrict training cell labels to the first 70% of each observed lifetime. Score validation/test cells on their final 30%; prior observations only construct causal features.
4. RUL is cycles remaining to the final valid capacity observation. Report this as dataset-end RUL unless the processed record exposes the documented CALCE cutoff endpoint.
5. Inputs are normalized capacity, normalized elapsed cycle, causal slopes and multiscale capacity history. Cell identity and future values are forbidden.
6. Compare matched plain MLP and affine-tail PP over seeds 42–46. No architecture or hyperparameter changes after outcome materialization.
7. Primary success: PP pooled ensemble R² > 0 and no lower than plain MLP. Secondary: unit metrics, paired RMSE, hull distance and 90% block-conformal coverage.
8. Parser-only repairs based on schema are logged. Any split, eligibility, feature or model change after outcome materialization makes the result developmental.
