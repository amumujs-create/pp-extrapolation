"""Coherent positive one-switch health paths for RAVEN-X."""
import torch
from .function_prior_sharing import integral


def one_switch_rul(theta_before,theta_after,margin,switch_health):
    """Integrate before regime from margin to switch, then after to boundary.

    theta tensors are [batch, sample, 3]; margin and switch_health are [batch].
    """
    if theta_before.shape!=theta_after.shape or theta_before.ndim!=3 or theta_before.shape[-1]!=3:
        raise ValueError('aligned batch x sample x 3 coefficients required')
    if margin.ndim!=1 or switch_health.shape!=margin.shape or len(margin)!=len(theta_before):
        raise ValueError('aligned one-dimensional health coordinates required')
    if torch.any(switch_health<0) or torch.any(switch_health>margin):
        raise ValueError('switch health must lie between boundary and current margin')
    before=integral(theta_before,margin,target=switch_health)
    after=integral(theta_after,switch_health)
    return before+after


def no_switch_rul(theta,margin):
    return integral(theta,margin)
