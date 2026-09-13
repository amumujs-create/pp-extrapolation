# Weak-prior projection authority synthetic protocol

**Frozen before execution:** 2026-09-12, 박진서

## Purpose

Test whether a learned authority head can exploit an incomplete monotonicity
and nonnegativity prior when it is useful, while relaxing it when the weak
prior becomes false outside source support.

## Synthetic tasks

Two hundred forty regression tasks are generated from fixed seeds. Each task
has observations on \(x\in[0,0.60]\) and an unseen extrapolation tail on
\(x\in(0.60,1]\).

The task function combines an affine trend, curvature, smooth oscillation,
noise, and an optional post-support tail reversal. Approximately half of the
tasks satisfy the weak nondecreasing prior over the full interval; the
remaining tasks violate it only partially or primarily in the unseen tail.

- authority-development tasks: 180
- untouched synthetic confirmation tasks: 60
- source rows: 64
- extrapolation rows: 48
- base predictor: identical two-layer tanh MLP and training budget in all arms

## Model

The base MLP produces a raw prediction vector \(z_\theta\) on an ordered
extrapolation ray. A differentiable enforcement layer produces

\[
\Pi_{\mathcal C}(z_\theta)
=
\operatorname{cummax}\{\max(z_\theta,0)\},
\]

which satisfies the weak nonnegative and nondecreasing constraints. The
learned authority head outputs \(a\in[0,1]\):

\[
\hat y
=
z_\theta
+a\left[
\Pi_{\mathcal C}(z_\theta)-z_\theta
\right].
\]

Thus \(a=0\) is exact unconstrained fallback and \(a=1\) is hard prior
enforcement.

## Outcome-free authority features

For each task the head receives only source and source-pseudo-tail evidence:

- direct pseudo-tail RMSE
- hard-projection pseudo-tail RMSE
- relative pseudo-tail projection gain
- raw monotonicity violation rate
- raw negative-output violation rate
- source curvature
- source residual scale
- source-to-pseudo-tail distance
- correction magnitude induced by projection

The last 25% of source support, \(x\in(0.45,0.60]\), is used as a causal
pseudo-tail. No confirmation-tail target enters authority features.

## Authority learning

On the 180 development tasks, the target authority is the value in
\(\{0,.1,\ldots,1\}\) minimizing true extrapolation RMSE. This is meta-training
across tasks, not adaptation on confirmation outcomes. A two-layer authority
MLP is trained with development tasks only.

All base models and authority heads are frozen before the 60 confirmation
tails are scored.

## Arms

1. raw direct MLP
2. hard weak-prior projection
3. pseudo-tail-selected authority
4. learned authority without projection-gain features
5. full learned weak-prior projection authority (primary)
6. tail oracle authority, explicitly nondeployable

## Evaluation

- equal-task mean RMSE
- mean relative RMSE change from direct
- task win fraction
- worst-20% relative harm
- results stratified by globally valid versus violated weak prior
- authority calibration against oracle authority

## Promotion

The primary model must:

- improve mean confirmation RMSE over direct MLP
- outperform hard projection
- reduce harm on violated-prior tasks relative to hard projection
- use nonzero and nonunit authority on confirmation tasks
- outperform the pseudo-tail-selected scalar authority

This synthetic result establishes mechanism plausibility only. Real
physical-unit and prospective experiments remain mandatory.
