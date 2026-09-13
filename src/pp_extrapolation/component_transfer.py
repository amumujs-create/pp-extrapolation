"""Experimental base-preserving, cross-fitted component residual transport.

The caller must generate honest held-out source episodes. This module does not
claim causal identification, distribution-free safety, or established novelty.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn

from .relation_local_transport import balanced_bank, group_weights


@dataclass
class ComponentBank:
    center: np.ndarray
    scale: np.ndarray
    x: np.ndarray
    values: np.ndarray
    weights: np.ndarray
    value_scale: float
    progress_index: int
    boundary_index: int | None


def make_bank(rows, base, *, progress_index=0, boundary_index=None, maximum=256):
    x, y, groups = np.asarray(rows['x'], float), np.asarray(rows['y'], float), np.asarray(rows['groups'])
    base = np.asarray(base, float)
    if x.ndim != 2 or not len(x) or y.shape != (len(x),) or base.shape != y.shape or groups.shape != y.shape:
        raise ValueError('aligned nonempty source rows/base required')
    if any(not np.isfinite(v).all() for v in (x, y, base)) or not 0 <= progress_index < x.shape[1]:
        raise ValueError('finite source data and valid progress index required')
    w = group_weights(groups)
    center = w @ x
    scale = np.sqrt(w @ (x-center)**2)
    scale[scale < 1e-8] = 1.
    value_scale = max(float(np.sqrt(w @ (y-w@y)**2)), 1e-6)
    residual = y-base
    if boundary_index is not None:
        if not 0 <= boundary_index < x.shape[1] or np.any(x[:, boundary_index] <= 0):
            raise ValueError('positive source boundary factor required')
        residual /= x[:, boundary_index]
    ix = balanced_bank(groups, maximum)
    return ComponentBank(center, scale, ((x-center)/scale)[ix], residual[ix]/value_scale,
                         group_weights(groups[ix]), value_scale, progress_index, boundary_index)


def component_features(bank, x, base):
    """Level, progress slope, remaining slopes; context needs no query outcomes."""
    x, base = np.asarray(x, float), np.asarray(base, float)
    if x.ndim != 2 or x.shape[1] != len(bank.center) or base.shape != (len(x),):
        raise ValueError('aligned queries/base required')
    if not np.isfinite(x).all() or not np.isfinite(base).all():
        raise ValueError('finite query observations required')
    factor = np.ones(len(x)) if bank.boundary_index is None else x[:, bank.boundary_index]
    if np.any(factor < 0) or (bank.boundary_index is not None and np.any(abs(base[factor == 0]) > 1e-10)):
        raise ValueError('boundary must be nonnegative with exact-zero boundary base')
    components, contexts = [], []
    for start in range(0, len(x), 64):
        q = (x[start:start+64]-bank.center)/bank.scale
        distance = ((q[:, None]-bank.x[None])**2).mean(-1)
        logits = -.5*distance+np.log(bank.weights)[None]
        w = np.exp(logits-logits.max(1, keepdims=True))
        w /= w.sum(1, keepdims=True)
        mx, my = w @ bank.x, w @ bank.values
        centered = bank.x[None]-mx[:, None]
        gram = np.einsum('bn,bni,bnj->bij', w, centered, centered)+.03*np.eye(bank.x.shape[1])[None]
        rhs = np.einsum('bn,bni,bn->bi', w, centered, bank.values[None]-my[:, None])
        slope = np.linalg.solve(gram, rhs[..., None]).squeeze(-1)
        pieces = (q-mx)*slope
        progress = pieces[:, bank.progress_index]
        other = pieces.sum(1)-progress
        c = np.column_stack((my, progress, other))
        std = np.sqrt(np.sum(w*(bank.values[None]-my[:, None])**2, axis=1))
        context = np.column_stack((c, np.log1p(np.sum(w*distance, axis=1)),
            1/(np.sum(w*w, axis=1)*len(bank.x)), std,
            q[:, bank.progress_index]-mx[:, bank.progress_index],
            base[start:start+64]/bank.value_scale))
        components.append(c*bank.value_scale*factor[start:start+64, None])
        contexts.append(context)
    return (np.concatenate(components), np.concatenate(contexts)) if components else (np.empty((0, 3)), np.empty((0, 8)))


class ComponentGate(nn.Module):
    def __init__(self, conditional=True):
        super().__init__()
        self.conditional = conditional
        if conditional:
            self.net = nn.Sequential(nn.Linear(8, 12), nn.Tanh(), nn.Linear(12, 3))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.constant_(self.net[-1].bias, -2.)
        else:
            self.logits = nn.Parameter(torch.full((3,), -2.))

    def forward(self, context):
        return torch.sigmoid(self.net(context) if self.conditional else self.logits.expand(len(context), 3))


@dataclass
class ComponentFit:
    model: ComponentGate
    context_center: np.ndarray
    context_scale: np.ndarray
    enabled: bool
    cap: float | None
    selection: dict


def predict_components(fit, components, context, base, *, return_gates=False):
    components, context, base = np.asarray(components, float), np.asarray(context, float), np.asarray(base, float)
    if components.shape != (len(base), 3) or context.shape != (len(base), 8):
        raise ValueError('unaligned component prediction inputs')
    if any(not np.isfinite(v).all() for v in (components, context, base)):
        raise ValueError('finite component prediction inputs required')
    with torch.no_grad():
        z = np.clip((context-fit.context_center)/fit.context_scale, -8., 8.)
        gate = fit.model(torch.tensor(z, dtype=torch.float32)).numpy() if fit.enabled else np.zeros_like(components)
    pred = base+(components*gate).sum(1)
    pred = np.clip(pred, 0, fit.cap) if fit.cap is not None else np.maximum(pred, 0)
    return (pred, gate) if return_gates else pred


def fit_components(episodes, validation, *, seed=42, conditional=True, robust_weight=.25, cap=None,
                   max_epochs=200, patience=40):
    """Train gates on source cross-fit residuals; validation selects checkpoint.

    Each episode contains c, z, base, y, groups; group identifiers must include
    the source fold/cutoff so reused source units are not treated as independent
    test cohorts. Epoch -1 is the exact disabled correction candidate.
    """
    if not episodes or robust_weight < 0 or max_epochs < 1 or (cap is not None and cap <= 0):
        raise ValueError('invalid component training configuration')
    data = {k: np.concatenate([e[k] for e in episodes]) for k in ('c', 'z', 'base', 'y', 'groups')}
    for d in (data, validation):
        if (d['c'].shape != (len(d['y']), 3) or d['z'].shape != (len(d['y']), 8)
                or d['base'].shape != d['y'].shape or d['groups'].shape != d['y'].shape
                or any(not np.isfinite(d[k]).all() for k in ('c', 'z', 'base', 'y'))):
            raise ValueError('finite aligned episode/validation arrays required')
    w = group_weights(data['groups'])
    center = w @ data['z']
    scale = np.sqrt(w @ (data['z']-center)**2)
    scale[scale < 1e-6] = 1.
    torch.manual_seed(seed)
    model = ComponentGate(conditional)
    fit = ComponentFit(model, center, scale, False, cap, {})
    vw = group_weights(validation['groups'])
    best = float(vw @ (validation['base']-validation['y'])**2)
    best_epoch, best_enabled, state = -1, False, copy.deepcopy(model.state_dict())
    z = torch.tensor(np.clip((data['z']-center)/scale, -8., 8.), dtype=torch.float32)
    c, base, y = [torch.tensor(data[k], dtype=torch.float32) for k in ('c', 'base', 'y')]
    ix = [np.flatnonzero(data['groups'] == g) for g in np.unique(data['groups'])]
    floor = max(float(np.mean((data['base']-data['y'])**2))*.05, 1e-8)
    baseline_risks = [(base[a]-y[a]).square().mean().clamp_min(floor) for a in ix]
    opt = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)
    history = []
    for epoch in range(max_epochs):
        correction = (c*model(z)).sum(1)
        prediction = (base+correction).clamp_min(0.)
        if cap is not None:
            prediction = prediction.clamp_max(cap)
        risk = torch.stack([(prediction[a]-y[a]).square().mean()/b for a, b in zip(ix, baseline_risks)])
        loss = risk.mean()+robust_weight*(risk-1).relu().square().mean()+.01*correction.square().mean()/max(floor/.05, 1e-8)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite component objective')
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.); opt.step()
        if (epoch+1) % 5 == 0:
            fit.enabled = True
            pred = predict_components(fit, validation['c'], validation['z'], validation['base'])
            val = float(vw @ (pred-validation['y'])**2)
            history.append(dict(epoch=epoch+1, source_ratio=float(risk.mean().detach()),
                                source_worst_ratio=float(risk.max().detach()), validation_unit_mse=val))
            if val < best-1e-8:
                best, best_epoch, best_enabled, state = val, epoch+1, True, copy.deepcopy(model.state_dict())
            if epoch+1-max(best_epoch, 0) >= patience:
                break
    model.load_state_dict(state)
    fit.enabled = best_enabled
    fit.selection = dict(seed=seed, selected_epoch=best_epoch, enabled=best_enabled,
        validation_unit_mse=best, history=history, conditional=conditional, robust_weight=robust_weight,
        meta_rows=len(base), meta_groups=len(ix), parameters=sum(p.numel() for p in model.parameters()))
    return fit
