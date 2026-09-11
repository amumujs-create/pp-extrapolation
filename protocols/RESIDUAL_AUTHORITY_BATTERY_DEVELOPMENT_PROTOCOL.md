# Boundary-quotient residual-authority development protocol

**Frozen before execution:** 2026-09-12, 박진서

## Status

Sunwoda, RWTH, and MICH are opened retrospective development datasets. This
experiment tests mechanism plausibility only and cannot be called prospective
confirmation.

## Purpose

Test residual authority on the three battery cohorts where the frozen
boundary-quotient prior, its nonlinear residual, and a matched direct NN can be
separated explicitly.

## Separation

The existing train, validation, and source/test unit assignments are retained.
Within the original train partition, every fifth sorted physical unit in each
dataset is reserved for checkpoint selection. The remaining train units fit
the direct NN and boundary-quotient model.

The untouched outer validation partition calibrates residual authority with
leave-one-unit-out cross-fitting. After the policy is frozen, each model is
refitted on the original train plus validation rows for its inner-selected
number of epochs. Source/test labels are read only for final reporting.

## Shared model budget

- architecture and feature contract: matched boundary-quotient control
- width: 64
- affine ridge alpha: 1000
- learning rate: 0.001
- weight decay: 0.01
- residual bound: 2.0
- seeds: 42, 43, 44
- maximum selection epochs: 500
- patience: 70

## Authority

The strict-tail health margin is the distance coordinate. It is normalized by
the median positive outer-validation margin separately per dataset.

- shell quantiles: 0%, 50%, 80%, 100%
- authority grid: 0, .25, .50, .75, 1
- primary constraint: non-increasing authority with distance
- active-shell requirement: at least two physical validation units
- cross-fitted authority requirement: at least five physical validation units;
  a smaller cohort uses the exact direct-NN fallback
- mean and worst-20% unit excess MSE versus matched direct NN: at most 2%
- approval: strictly positive leave-one-unit-out gain and both harm constraints

An unsupported shell uses the exact direct-NN fallback. After the first
fallback shell, all farther shells also use fallback.

## Arms

1. matched direct NN fallback
2. frozen affine quotient only
3. full bounded boundary-quotient residual
4. global authority
5. unconstrained shell authority
6. monotone shell authority

## Decision

The challenger is not promoted from this retrospective run. Promotion requires
improvement over the existing PP-X executor under a frozen policy on a new
untouched cohort.
