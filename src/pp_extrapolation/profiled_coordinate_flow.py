"""Experimental differentiable identification of an observed-coordinate flow.

An ordered coordinate may be time or a declared physical progression variable.
No physics discovery, global monotonicity, or coverage of unordered covariate
shifts is implied. Source training learns a shared relaxation dictionary and
coefficient regularization. Each query prefix identifies local coefficients by
one differentiable linear solve, with no test-time optimizer or label access.

This is related to variable projection / dynamic mode decomposition; a distinct
novelty claim requires more than this implementation.
"""
import math

import torch
from torch import nn


class ProfiledCoordinateFlow(nn.Module):
    """Exact observed-boundary continuation with stable relaxation modes.

    phi(s) = [s, tau_k (1-exp(-s/tau_k))]. A centred prefix solve estimates
    coefficients c. Prediction is y_last + (phi(s_last+h)-phi(s_last)) c.
    Signed coefficients allow either increasing or decreasing outputs. The
    relaxation components stay bounded, but the linear component is unbounded.
    """

    def __init__(self, *, learn_dictionary=True):
        super().__init__()
        self.log_tau = nn.Parameter(torch.tensor([math.log(v) for v in (.125, .5, 2.)]),
                                    requires_grad=learn_dictionary)
        self.log_ridge = nn.Parameter(torch.full((4,), -4.), requires_grad=learn_dictionary)
        self.prior = nn.Parameter(torch.zeros(4), requires_grad=learn_dictionary)

    def basis(self, coordinate):
        tau = self.log_tau.clamp(-5., 5.).exp()
        relax = -tau * torch.expm1(-coordinate[..., None] / tau)
        return torch.cat((coordinate[..., None], relax), dim=-1)

    def forecast(self, x, coordinate, mask, horizons):
        if x.ndim != 3 or x.shape[2] != 1:
            raise ValueError('one observed output channel is required')
        if coordinate.shape != x.shape[:2] or mask.shape != x.shape[:2]:
            raise ValueError('unaligned prefix arrays')
        if horizons.ndim != 2 or len(horizons) != len(x):
            raise ValueError('horizons must have shape (batch, queries)')
        if mask.dtype != torch.bool:
            raise ValueError('mask must be boolean')
        lengths = mask.long().sum(1)
        expected = torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None]
        if torch.any(lengths < 2) or not torch.equal(mask, expected):
            raise ValueError('at least two right-padded observations per prefix required')
        if (not torch.isfinite(x[mask]).all() or not torch.isfinite(coordinate[mask]).all()
                or not torch.isfinite(horizons).all() or torch.any(horizons < 0)):
            raise ValueError('finite observations and nonnegative horizons required')
        adjacent = mask[:, 1:] & mask[:, :-1]
        if torch.any((coordinate[:, 1:] - coordinate[:, :-1])[adjacent] <= 0):
            raise ValueError('observed coordinates must be strictly increasing')
        # Sanitize padding before exponentiation and arithmetic; zero times NaN
        # would not suppress invalid padding in a masked multiplication.
        s = torch.where(mask, coordinate - coordinate[:, :1], 0.)
        y = torch.where(mask, x[..., 0], 0.)
        phi = self.basis(s)
        w = mask.to(x.dtype)
        count = lengths.to(x.dtype)
        phi_mean = (phi*w[..., None]).sum(1)/count[:, None]
        y_mean = (y*w).sum(1)/count
        design = (phi-phi_mean[:, None])*w[..., None]
        centered_y = (y-y_mean[:, None])*w
        gram = torch.einsum('nli,nlj->nij', design, design)/count[:, None, None]
        rhs = torch.einsum('nli,nl->ni', design, centered_y)/count[:, None]
        ridge = self.log_ridge.clamp(-12., 6.).exp() + 1e-8
        matrix = gram + torch.diag(ridge)[None]
        coefficients = torch.linalg.solve(matrix, (rhs + ridge*self.prior)[..., None]).squeeze(-1)
        index = torch.arange(len(x), device=x.device)
        last = lengths - 1
        endpoint = s[index, last]
        future = self.basis(endpoint[:, None] + horizons)
        delta_basis = future - self.basis(endpoint)[:, None]
        prediction = y[index, last, None] + torch.einsum('nhi,ni->nh', delta_basis, coefficients)
        return prediction[..., None]
