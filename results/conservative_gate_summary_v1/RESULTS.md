# Conservative unanimous-stability gate

PP is selected only when source environments unanimously pass the stability gate; otherwise the route falls back to plain NN.

| dataset | selected route | selected pooled R² | audit oracle | regret |
|---|---|---:|---|---:|
| hust | **plain_nn** | 0.758 | plain_nn (0.758) | 0.000 |
| virkler | **pp** | 0.857 | pp (0.857) | 0.000 |
| nasa | **pp** | 0.495 | pp (0.495) | 0.000 |

- PP route coverage: **66.7%**
- prediction coverage: **100.0%**
- dataset-macro selected pooled R²: **0.703**
- always-PP dataset macro: **0.692**
- observed gain over always PP: **+0.012**
- worst selected dataset R²: **0.495**
- mean observed head regret: **0.000**

This threshold was created after the HUST final ranking was known. Zero observed regret is a retrospective repair result, not validation of the gate.
