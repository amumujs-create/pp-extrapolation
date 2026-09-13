# ECAT-PPX external Engression challenge protocol

## Frozen target

ECAT-PPX must exceed the already recorded official Engression pooled R2 on
each of Axial-fan `1P_8F`, `4P_1F`, `4P_8F`, MATWI, and Misata. All failures
remain in the table. The five outcomes are already open, so this is
retrospective development only.

## Independent architecture

The model does not consume Engression predictions or parameters. It combines
four endpoint experts: a global ridge slope, rank-weighted local analog
transport, direct extra-tree transport, and ridge-residual tree transport.
Histogram gradient transport is also a candidate. A group-risk validation
router selects at most four experts and simplex weights. The global affine
expert can receive exact weight zero, implementing affine-prior rejection.

## Selection and stability

Selection minimizes equal-unit validation MSE, followed by CVaR20 and maximum
unit MSE. Axial and Misata reuse their original validation units; MATWI reuses
the original four unit folds. Selected stochastic experts are refit with seeds
42--46. Test labels are used only after prediction.

## Claim firewall

Even a 5/5 win is not external confirmation because the architecture was
created after these outcomes were observed. A new untouched cohort is required
for a general superiority statement.
