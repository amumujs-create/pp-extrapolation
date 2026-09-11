# CCMR v2.1 stochastic predictor firewall

Frozen 2026-09-11 by 박진서 before v2.1 implementation.

Alloy A and Multi-Stage RPT test labels, errors, competitor scores, and feature
descriptors are excluded from v2.1 architecture, hyperparameter, threshold,
and stopping decisions. Their train and validation partitions may be used only
after freezing, as ordinary within-cohort fitting data; their test partitions
may then be replayed once.

Development cohorts are Concrete, LG M50T, SIT LFP, RADAR NMC, and Luminosity.
The deterministic center is frozen CCMR v2.0. The added head is fixed as:

- two-layer width-16 tanh network;
- Gaussian noise input dimension 4;
- eight Monte Carlo samples per training row;
- group-balanced energy score plus 0.1 mean-square stabilization;
- at most 200 epochs, validation early stopping patience 30;
- five train-group cross-fit buckets;
- seeds 42--46;
- train-only robust feature and target scaling.

The validation conformal interval uses the 90th percentile standardized
absolute error. A correction is eligible only if its interval excludes zero,
cross-fit sign agreement is at least 80%, relative dispersion is at most 75%,
and the deterministic v2.0 support gate accepts it.

Deployment mass is selected from 0 to 1 in increments of 0.01 under raw
physical-unit regret caps mean/CVaR20/maximum = 0%/1%/2%. If no candidate
strictly improves unit-macro MSE, mass is zero. Non-finite inputs or outputs,
failed conformal evidence, disagreement, support rejection, and zero mass must
return the supplied persistence anchor bit-exactly.

Promotion requires non-holdout false accepts equal to zero, maximum raw test
regret no greater than 2%, and at least one development cohort with a strict
pooled improvement over frozen v2.0.
