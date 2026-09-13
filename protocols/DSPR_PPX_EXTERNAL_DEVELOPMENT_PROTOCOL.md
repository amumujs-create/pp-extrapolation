# DSPR-PPX external development protocol

## Requirement

The strict development target is pooled R2 greater than the matched official
Engression result on each of five opened external settings: three Axial-fan
configurations, MATWI tool life, and Misata machine degradation. Coverage and
seed stability must also be reported without deleting failed settings.

## Architecture

DSPR-PPX is a single distributional structural prior-residual network. It has
a frozen validation-selected affine path, a bounded nonlinear residual
location, a positive residual scale, and a learned input-dependent authority
gate. It is trained by an equal-unit weighted energy score plus predictive-mean
loss. Engression predictions or parameters are not model inputs.

## Selection

Search width `{32,64}`, normalized residual bound `{0.5,1.0}`, and mean-loss
weight `{0.1,0.5}`. Learning rate is `0.001`, weight decay is `0.1`, and seed
42 performs selection. Axial-fan and Misata use their existing validation
units. MATWI uses the original four development-unit folds. Refit the selected
configuration on all development units for seeds 42--46 using the median
selected epoch.

## Evidence status

All five outcomes and Engression scores were opened before this model was
created. Passing the strict target would be retrospective development success,
not external confirmation. A new untouched cohort remains mandatory.
