# CIST-PPX external development protocol

The model and interpretation are defined in
`PPX_ENGRESSION_MECHANISM_AND_CIST_ARCHITECTURE_KO.md`.

Search width `{32,64}`, slope-integration steps `{4,8}`, energy beta `{0.5,1}`,
and point-mean weight `{0.1,0.5}` with seed 42. Select by equal-unit validation
MSE; MATWI uses its original four unit folds. Axial validation is reduced to
one final pseudo-endpoint per validation fan to match the test observation
contract. Refit the selected configuration for seeds 42--46.

Strict development success requires pooled R2 above official Engression on all
five opened external settings. Engression predictions and parameters are not
used by CIST-PPX. This is retrospective development and cannot replace a new
untouched confirmation.
