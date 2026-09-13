# RAVEN-X v2 sticky-duration remediation

Status: recorded after v1 revealed transition-everywhere collapse and before v2 training.

This is post-v1 TRAIN development, not an independent confirmation. V1 remains preserved.

## Fixed change

- Initialize both transition-type logits to -3.
- Add a fixed early-to-late bin offset from -0.5 to +0.5.
- Initialize extrapolation-distance effects at +0.1.
- Add `0.01 * -log P(no switch)` to the inner source objective.
- Change no other architecture, split, input, decoder, seed, step count, sampling, or control definition.

## Evaluation

Reuse the six v1 prepared outer/inner TRAIN-only episodes byte-for-byte. Refit all eight learned arms for seeds 42--46. Use the same exact seed-mixture scoring and the original v1 real-data gate. Synthetic relation-mask success is inherited only as a code-structure check; it is not a real-data gate substitute.

V2 advances only if the original conjunctive gate passes. A better average than v1 alone is insufficient. Validation and test remain unopened after failure.
