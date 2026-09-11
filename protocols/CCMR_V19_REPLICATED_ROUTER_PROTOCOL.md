# CCMR v1.9 replicated validation router

**Frozen:** 2026-09-11, 박진서, before downloading Multi-Stage RPT outcomes.

This is post-test development from the opened RADAR NMC result. RADAR showed
that the v1.8 stable certificate was stricter than the final deployment risk
contract and rejected a candidate that subsequently passed every test risk
and performance cap. It is not independent v1.9 evidence.

## Frozen route

Fit the unchanged CCMR-L v1.6.1 candidate. Select `stable_base` only when:

1. validation contains at least 10 independent physical units;
2. base active fraction is at least 50%;
3. unit-macro RMSE improvement is at least 5%;
4. raw unit mean/CVaR20/maximum regret are at most 0%/1%/2%.

This replaces the arbitrary v1.8 requirement that all three validation
regrets be below -5% and aligns routing with the deployment risk contract.
The independent-unit floor prevents a small cohort from establishing the
stable regime.

If the certificate passes, deploy base CCMR with its cross-fit consensus and
context-support row fallback. Otherwise use the frozen v1.7 causal route and
global validation veto. Route selection cannot inspect test outcomes.
