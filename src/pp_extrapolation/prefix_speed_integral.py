"""Experimental prefix-conditioned inverse-speed model, not a PP-X replacement.

Given observed margin m and causal rate r, learn bounded coefficients of
log slowness: a(z) + b(z) q + c(z) q², q=(m-h)/m. Integrate from the current
health to a requested lower health. RUL is the integral to the known boundary.
Source-only observed transitions supervise partial integrals, not just RUL.
This requires a meaningful monotone degradation coordinate and positive rate;
it is not a generic covariate-extrapolation model or a novelty claim.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .boundary_quotient import equal_dataset_unit_weights


class PrefixSpeedIntegral(nn.Module):
    def __init__(self, center, scale, *, width=64, variable_speed=True, quadrature=24):
        super().__init__()
        center, scale = np.asarray(center), np.asarray(scale)
        if (center.ndim != 1 or center.shape != scale.shape or not np.isfinite(center).all()
                or not np.isfinite(scale).all() or np.any(scale <= 0) or width < 1
                or not isinstance(quadrature, int) or quadrature < 2):
            raise ValueError('finite feature scales and at least two quadrature points required')
        self.variable_speed = bool(variable_speed)
        self.register_buffer('center', torch.as_tensor(center, dtype=torch.float32))
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32))
        nodes, weights = np.polynomial.legendre.leggauss(quadrature)
        self.register_buffer('nodes', torch.tensor((nodes+1)/2, dtype=torch.float32))
        self.register_buffer('weights', torch.tensor(weights/2, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(len(center), width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 3))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def coefficients(self, x):
        if x.ndim != 2 or x.shape[1] != len(self.center) or not torch.isfinite(x).all():
            raise ValueError('finite aligned feature matrix required')
        value = 2 * torch.tanh(self.net((x-self.center)/self.scale))
        if not self.variable_speed:
            value = value * value.new_tensor([1., 0., 0.])
        return value

    def travel_time(self, x, margin, rate, target_margin=None):
        """All margins share units; rate has units margin/time; output is time.

        target_margin is a training target coordinate for observed segments.
        Deployment RUL uses zero and never receives future observed health.
        """
        if target_margin is None:
            target_margin = torch.zeros_like(margin)
        if any(v.shape != (len(x),) for v in (margin, rate, target_margin)):
            raise ValueError('margin/rate/target must be aligned vectors')
        if (any(not torch.isfinite(v).all() for v in (margin, rate, target_margin))
                or torch.any(margin < 0) or torch.any(rate <= 0)
                or torch.any(target_margin < 0) or torch.any(target_margin > margin)):
            raise ValueError('require finite 0 <= target <= margin and rate > 0')
        coeff = self.coefficients(x)
        distance = margin-target_margin
        fraction = distance / margin.clamp_min(torch.finfo(margin.dtype).tiny)
        q = fraction[:, None] * self.nodes[None]
        log_slowness = coeff[:, :1] + coeff[:, 1:2]*q + coeff[:, 2:3]*q.square()
        return distance/rate * torch.sum(log_slowness.exp()*self.weights[None], dim=1)

    def forward(self, x, margin, rate):
        return self.travel_time(x, margin, rate)


def make_prefix_pairs(rows, *, offsets=(1, 4, 16), min_cycle_gap=8):
    """Ordered prefix/suffix endpoints within each source unit.

    No RUL labels are read. Future endpoint margin and elapsed cycles are targets
    only. Source windows are sampled irregularly: offsets index available windows,
    while actual elapsed cycles and prefix ending cycles are retained. The gap
    is a cycle-distance threshold, not a claim that irregular windows are disjoint.
    """
    if min_cycle_gap <= 0 or not offsets or any(int(o) != o or o < 1 for o in offsets):
        raise ValueError('positive integer offsets and positive cycle gap required')
    n = len(rows['x'])
    for key in ('margin', 'time', 'cycles', 'units', 'dataset'):
        if np.asarray(rows[key]).shape != (n,) or not np.isfinite(rows[key]).all():
            raise ValueError(f'invalid pair field {key}')
    anchors, futures = [], []
    for di in np.unique(rows['dataset']):
        for unit in np.unique(rows['units'][rows['dataset'] == di]):
            ix = np.flatnonzero((rows['dataset'] == di) & (rows['units'] == unit))
            ix = ix[np.argsort(rows['time'][ix], kind='stable')]
            if np.any(np.diff(rows['time'][ix]) <= 0):
                raise ValueError('unit times must be unique and increasing')
            for offset in sorted(set(offsets)):
                for j in range(len(ix)-offset):
                    a, b = ix[j], ix[j+offset]
                    if (rows['cycles'][b]-rows['cycles'][a] >= min_cycle_gap
                            and rows['margin'][a] > rows['margin'][b] >= 0):
                        anchors.append(a)
                        futures.append(b)
    a, b = np.asarray(anchors, dtype=np.int64), np.asarray(futures, dtype=np.int64)
    return dict(anchor=a, future=b, target_margin=rows['margin'][b],
                elapsed=rows['time'][b]-rows['time'][a])


def _validate_rows(rows):
    n = len(rows['x'])
    if n == 0 or np.asarray(rows['x']).ndim != 2 or not np.isfinite(rows['x']).all():
        raise ValueError('nonempty finite feature matrix required')
    for key in ('margin', 'rate', 'y', 'dataset', 'units'):
        if np.asarray(rows[key]).shape != (n,) or not np.isfinite(rows[key]).all():
            raise ValueError(f'invalid field {key}')
    if np.any(rows['margin'] < 0) or np.any(rows['rate'] <= 0) or np.any(rows['y'] < 0):
        raise ValueError('nonnegative margin/target and positive rate required')


@dataclass
class PrefixSpeedFit:
    model: PrefixSpeedIntegral
    selection: dict


def predict_prefix_speed(fit, rows):
    model = fit.model if isinstance(fit, PrefixSpeedFit) else fit
    model.eval()
    with torch.no_grad():
        result = model(*(torch.as_tensor(rows[k], dtype=torch.float32)
                         for k in ('x', 'margin', 'rate'))).numpy()
    if not np.isfinite(result).all():
        raise RuntimeError('nonfinite prediction')
    return result


def fit_prefix_speed(train, validation, *, seed=42, variable_speed=True,
                     pair_weight=.1, max_epochs=500, patience=70, restore_best=True,
                     width=64, batch_size=512):
    _validate_rows(train)
    _validate_rows(validation)
    if max_epochs < 1 or patience < 0 or batch_size < 1 or pair_weight < 0:
        raise ValueError('invalid fit configuration')
    torch.manual_seed(seed)
    center, scale = train['x'].mean(0), train['x'].std(0)
    scale = np.where(scale < 1e-6, 1., scale)
    model = PrefixSpeedIntegral(center, scale, width=width, variable_speed=variable_speed)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    tensors = {k: torch.as_tensor(train[k], dtype=torch.float32) for k in ('x', 'margin', 'rate', 'y')}
    weights = torch.tensor(equal_dataset_unit_weights(train['dataset'], train['units']))
    pairs = make_prefix_pairs(train) if pair_weight else None
    if pairs is not None and not len(pairs['anchor']):
        raise ValueError('positive pair loss requires eligible source transitions')
    if pairs is not None:
        pa = pairs['anchor']
        pw = torch.tensor(equal_dataset_unit_weights(train['dataset'][pa], train['units'][pa]))
        pt = {k: torch.as_tensor(pairs[k]) for k in ('anchor', 'target_margin', 'elapsed')}
    masks = [validation['dataset'] == d for d in np.unique(validation['dataset'])]

    def risk():
        err = (predict_prefix_speed(model, validation).astype(float)-validation['y'])**2
        return float(np.mean([err[m].mean() for m in masks]))

    best, epoch_best = risk(), 0
    best_state = copy.deepcopy(model.state_dict())
    rng = np.random.default_rng(seed)
    history, updates = [], 0
    for epoch in range(1, max_epochs+1):
        model.train()
        order = rng.permutation(len(train['y']))
        losses = []
        for start in range(0, len(order), batch_size):
            ix = torch.as_tensor(order[start:start+batch_size])
            pred = model(tensors['x'][ix], tensors['margin'][ix], tensors['rate'][ix])
            loss = (weights[ix]*(pred-tensors['y'][ix]).square()).mean()
            if pairs is not None:
                # Fixed-size uniform source-pair sampling, with dataset/unit weights.
                pi = torch.as_tensor(rng.integers(len(pairs['anchor']), size=len(ix)))
                ai = pt['anchor'][pi]
                segment = model.travel_time(tensors['x'][ai], tensors['margin'][ai],
                                            tensors['rate'][ai], pt['target_margin'][pi])
                relative = (segment-pt['elapsed'][pi])/(pt['elapsed'][pi]+.1)
                loss = loss + pair_weight*(pw[pi]*relative.square()).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.)
            opt.step()
            updates += 1
            losses.append(float(loss.detach()))
        current = risk()
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), validation_mse=current))
        if current < best-1e-8:
            best, epoch_best, best_state = current, epoch, copy.deepcopy(model.state_dict())
        if restore_best and epoch-epoch_best > patience:
            break
    if restore_best:
        model.load_state_dict(best_state)
    model.eval()
    return PrefixSpeedFit(model, dict(seed=seed, selected_epoch=epoch_best,
        executed_epochs=epoch, updates=updates, validation_mse=best,
        pair_weight=pair_weight, source_pairs=0 if pairs is None else len(pairs['anchor']),
        variable_speed=variable_speed, restore_best=restore_best, history=history,
        parameters=sum(p.numel() for p in model.parameters())))
