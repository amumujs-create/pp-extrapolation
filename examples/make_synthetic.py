"""Create a small run-to-failure CSV for checking the PP pipeline."""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
rows = []
for unit in range(1, 41):
    life = int(rng.integers(80, 121))
    rate = rng.uniform(0.008, 0.013)
    for step in range(life + 1):
        health = 1.2 - rate * step + rng.normal(0.0, 0.008)
        recent_rate = rate + rng.normal(0.0, 0.0005)
        rows.append(
            {
                "unit": unit,
                "step": step,
                "health": health,
                "recent_rate": recent_rate,
                "RUL": life - step,
            }
        )
path = Path(__file__).with_name("synthetic_rul.csv")
pd.DataFrame(rows).to_csv(path, index=False)
print(path.resolve())

