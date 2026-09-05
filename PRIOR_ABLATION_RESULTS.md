# Synthetic prior ablation — development only

Fixed seed split, model seeds 42–44, 150 epochs maximum. All tested configurations are reported. No real-data performance claim.

| Scenario | Variant | Pooled R² mean | Seed SD |
|---|---|---:|---:|
| smooth | PP | -1.191274 | 0.078185 |
| smooth | direction | -1.191274 | 0.078185 |
| smooth | transport | -1.191274 | 0.078185 |
| smooth | both | -1.191274 | 0.078185 |
| smooth | wrong_direction | -1.945470 | 0.000000 |
| turning | PP | -23.755170 | 0.519064 |
| turning | direction | -23.755170 | 0.519064 |
| turning | transport | -23.755170 | 0.519064 |
| turning | both | -23.755170 | 0.519064 |
| turning | wrong_direction | -10.852713 | 0.000000 |

No positive improvement was observed for direction/transport/both in these runs. The enabled losses can be inactive inside their allowed margins; equal scores alone do not establish the cause. Both scenarios have negative test R², so this is a failed extrapolation benchmark, not supporting evidence for accuracy.

`wrong_direction` means reversed relative to the declared increasing-health prior. It is not globally wrong for the turning function, whose derivative changes sign. Its partial improvement on that scenario remains negative and cannot validate the method.

Existing PP fixes the affine slope. A bounded correction cannot globally repair an incompatible affine tail. This motivates investigating domain-of-validity estimation and train-only validation of prior reliability before making stronger claims.
