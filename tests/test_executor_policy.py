from pp_extrapolation.executor_policy import ExecutorEvidence, select_residual_executor


def test_defaults_to_unbounded_without_validation_gain():
    decision = select_residual_executor(ExecutorEvidence(10.0, bounded_loss=9.9, dual_scale_loss=9.0))
    assert decision.executor == "unbounded"


def test_approves_bounded_only_after_margin():
    decision = select_residual_executor(ExecutorEvidence(10.0, bounded_loss=9.0))
    assert decision.executor == "bounded"


def test_dual_scale_requires_heterogeneous_support_and_beats_bound():
    rejected = select_residual_executor(ExecutorEvidence(10.0, bounded_loss=9.0, dual_scale_loss=8.0, support_heterogeneity=0.1))
    approved = select_residual_executor(ExecutorEvidence(10.0, bounded_loss=9.0, dual_scale_loss=8.0, support_heterogeneity=0.8))
    assert rejected.executor == "bounded"
    assert approved.executor == "dual_scale"
