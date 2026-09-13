"""Experimental relation-trained local linear operator for prior repair/fallback.

The same source-value-linear operator S(x) acts on residuals when an external
contract approves a prior, and on targets when the prior is rejected. Kernel
geometry is learned from unit-disjoint, ordered-support source episodes. This
is a local-smoothness assumption, NOT assumption-free extrapolation, a causal
fault identifier, or an established methodological novelty.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn


def _rows(rows):
    x, y, g = np.asarray(rows['x'], float), np.asarray(rows['y'], float), np.asarray(rows['groups'])
    if x.ndim != 2 or not len(x) or y.shape != (len(x),) or g.shape != y.shape:
        raise ValueError('nonempty aligned x/y/groups required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('finite observations required')
    return x, y, g.astype(str)


def group_weights(groups):
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    w = 1./counts[inverse]
    return w/w.sum()


def balanced_bank(groups, maximum=256):
    unique = np.unique(groups)
    if maximum < len(unique):
        raise ValueError('bank capacity must cover every source unit')
    count = max(1, maximum//len(unique))
    indices = []
    for g in unique:
        ix = np.flatnonzero(groups == g)
        indices.extend(ix[np.unique(np.linspace(0, len(ix)-1, min(count, len(ix))).astype(int))])
    return np.asarray(indices, dtype=np.int64)


def source_episodes(x, groups, bank, *, progress_index, increasing, max_queries=48):
    """Meta-training only: bank and query units AND progress supports differ.

    The global normalizer is fitted to TRAIN, including its meta-query units.
    These are training episodes, not nested independent performance estimates.
    """
    coordinate = x[:, progress_index] * (1 if increasing else -1)
    episodes = []
    for fraction in (.4, .6, .8):
        cutoff = float(np.quantile(coordinate, fraction))
        for group in np.unique(groups):
            anchors = bank[(groups[bank] != group) & (coordinate[bank] <= cutoff)]
            query = np.flatnonzero((groups == group) & (coordinate > cutoff))
            if len(anchors) < max(8, x.shape[1]+2) or len(query) < 3:
                continue
            query = query[np.argsort(coordinate[query], kind='stable')]
            query = query[np.unique(np.linspace(0, len(query)-1, min(max_queries, len(query))).astype(int))]
            episodes.append(dict(anchor=anchors, query=query, cutoff=cutoff, group=str(group)))
    if not episodes:
        raise ValueError('no unit-disjoint ordered-support episodes available')
    return episodes


class LocalRelationOperator(nn.Module):
    """Differentiable local ridge fit with a learned diagonal distance metric."""
    def __init__(self, dimensions, *, linear=True):
        super().__init__()
        if dimensions < 1:
            raise ValueError('positive input dimension required')
        self.linear = bool(linear)
        self.log_metric = nn.Parameter(torch.zeros(dimensions, dtype=torch.float64))
        self.log_bandwidth = nn.Parameter(torch.tensor(0., dtype=torch.float64))
        self.log_ridge = nn.Parameter(torch.tensor(np.log(.03), dtype=torch.float64))

    def forward(self, query, bank, values, weights, *, return_parts=False):
        if (query.ndim != 2 or bank.ndim != 2 or query.shape[1] != bank.shape[1]
                or values.shape != (len(bank),) or weights.shape != (len(bank),) or not len(bank)
                or query.shape[1] != len(self.log_metric)):
            raise ValueError('unaligned local regression inputs')
        if (any(not torch.isfinite(v).all() for v in (query, bank, values, weights))
                or torch.any(weights <= 0)):
            raise ValueError('finite inputs and positive bank weights required')
        metric = self.log_metric.clamp(-3., 3.).exp()
        distance = ((query[:, None]-bank[None]).square()*metric).mean(-1)
        bandwidth = self.log_bandwidth.clamp(-3., 3.).exp()
        w = torch.softmax(-distance/(2*bandwidth.square()) + weights.log()[None], dim=1)
        anchor_x, anchor_y = w @ bank, w @ values
        if self.linear:
            centered = bank[None]-anchor_x[:, None]
            gram = torch.einsum('bn,bni,bnj->bij', w, centered, centered)
            ridge = self.log_ridge.clamp(-9., 2.).exp()
            gram = gram + ridge*torch.eye(bank.shape[1], dtype=bank.dtype, device=bank.device)[None]
            rhs = torch.einsum('bn,bni,bn->bi', w, centered, values[None]-anchor_y[:, None])
            slope = torch.linalg.solve(gram, rhs[..., None]).squeeze(-1)
            extension = ((query-anchor_x)*slope).sum(-1)
        else:
            extension = torch.zeros_like(anchor_y)
        return (anchor_y, extension) if return_parts else anchor_y+extension


@dataclass
class RelationTransportFit:
    model: LocalRelationOperator
    center: np.ndarray
    scale: np.ndarray
    bank_x: np.ndarray
    bank_values: np.ndarray
    bank_weights: np.ndarray
    bank_indices: np.ndarray
    value_scale: float
    prior_approved: bool
    boundary_index: int | None
    cap: float | None
    selection: dict


def predict_relation_transport(fit, x, *, prior_prediction=None, level_only=False):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != len(fit.center) or not np.isfinite(x).all():
        raise ValueError('finite aligned query features required')
    if fit.prior_approved:
        base = np.asarray(prior_prediction, dtype=float)
        if base.shape != (len(x),) or not np.isfinite(base).all():
            raise ValueError('approved route requires finite aligned prior predictions')
    else:
        # Rejected priors are not even inspected. They cannot leak into fallback.
        base = np.zeros(len(x))
    factor = np.ones(len(x))
    if fit.prior_approved and fit.boundary_index is not None:
        factor = x[:, fit.boundary_index]
        if np.any(factor < 0) or np.any(np.abs(base[factor == 0]) > 1e-10):
            raise ValueError('nonnegative boundary factor and exact-zero boundary prior required')
    values = []
    fit.model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 64):
            q = torch.tensor((x[start:start+64]-fit.center)/fit.scale)
            level, slope = fit.model(q, torch.tensor(fit.bank_x), torch.tensor(fit.bank_values),
                                     torch.tensor(fit.bank_weights), return_parts=True)
            values.append((level if level_only else level+slope).numpy()*fit.value_scale)
    prediction = base+factor*np.concatenate(values) if values else base
    if not np.isfinite(prediction).all():
        raise RuntimeError('nonfinite transport prediction')
    return np.clip(prediction, 0, fit.cap) if fit.cap is not None else np.maximum(prediction, 0)


def fit_relation_transport(train, validation, *, progress_index, increasing, seed=42,
                           prior_approved=False, prior_train=None, prior_validation=None,
                           relation_weight=.1, learn=True, linear=True, max_epochs=100,
                           patience=25, maximum_bank=256, cap=None, boundary_index=None):
    tx, ty, tg = _rows(train)
    vx, vy, vg = _rows(validation)
    if set(tg) & set(vg):
        raise ValueError('train and validation units must be disjoint')
    if not 0 <= progress_index < tx.shape[1] or relation_weight < 0 or max_epochs < 1:
        raise ValueError('invalid fit configuration')
    if vx.shape[1] != tx.shape[1] or (cap is not None and (not np.isfinite(cap) or cap <= 0)):
        raise ValueError('invalid dimensions or output cap')
    weight = group_weights(tg)
    center = np.sum(weight[:, None]*tx, axis=0)
    scale = np.sqrt(np.sum(weight[:, None]*(tx-center)**2, axis=0))
    scale[scale < 1e-8] = 1.
    ycenter = float(weight @ ty)
    yscale = max(float(np.sqrt(weight @ (ty-ycenter)**2)), 1e-6)
    x, y = (tx-center)/scale, (ty-ycenter)/yscale
    bank = balanced_bank(tg, maximum_bank)
    raw_values = ty.copy()
    if prior_approved:
        pt = np.asarray(prior_train, float)
        pv = np.asarray(prior_validation, float)
        if pt.shape != ty.shape or pv.shape != vy.shape or not np.isfinite(pt).all() or not np.isfinite(pv).all():
            raise ValueError('approved prior arrays must be aligned and finite')
        raw_values -= pt
        if boundary_index is not None:
            if not 0 <= boundary_index < tx.shape[1] or np.any(tx[:, boundary_index] <= 0):
                raise ValueError('positive train boundary factor required')
            raw_values /= tx[:, boundary_index]
    episodes = source_episodes(x, tg, bank, progress_index=progress_index, increasing=increasing)
    model = LocalRelationOperator(tx.shape[1], linear=linear)
    fit = RelationTransportFit(model, center, scale, x[bank], raw_values[bank]/yscale,
        group_weights(tg[bank]), bank, yscale, bool(prior_approved),
        boundary_index if prior_approved else None, cap, {})
    xt, yt = torch.tensor(x), torch.tensor(y)
    validation_weights = group_weights(vg)

    def risk():
        p = predict_relation_transport(fit, vx, prior_prediction=prior_validation)
        return float(validation_weights @ (p-vy)**2)

    best, epoch_best = risk(), 0
    state = copy.deepcopy(model.state_dict())
    history, updates = [], 0
    opt = torch.optim.Adam(model.parameters(), lr=.02)
    rng = np.random.default_rng(seed)
    for epoch in range(1, max_epochs+1) if learn else ():
        model.train()
        losses = []
        # Four bounded episodes per epoch; chronological relation pairs stay intact.
        for index in rng.choice(len(episodes), size=min(4, len(episodes)), replace=False):
            episode = episodes[index]
            a, q = episode['anchor'], episode['query']
            pred = model(xt[q], xt[a], yt[a], torch.tensor(group_weights(tg[a])))
            loss = (pred-yt[q]).square().mean()
            if relation_weight:
                span = torch.abs(torch.diff(xt[q, progress_index]))+.25
                relation = (torch.diff(pred)-torch.diff(yt[q]))/span
                loss = loss+relation_weight*relation.square().mean()
            loss = loss + .001*model.log_metric.square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training objective')
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.)
            opt.step()
            updates += 1
            losses.append(float(loss.detach()))
        if epoch % 5 == 0 or epoch == max_epochs:
            current = risk()
            history.append(dict(epoch=epoch, loss=float(np.mean(losses)), validation_unit_mse=current))
            if current < best-1e-8:
                best, epoch_best, state = current, epoch, copy.deepcopy(model.state_dict())
            if epoch-epoch_best >= patience:
                break
    model.load_state_dict(state)
    model.eval()
    fit.selection = dict(selected_epoch=epoch_best, validation_unit_mse=best, updates=updates,
        history=history, source_episodes=len(episodes), source_bank=len(bank), seed=seed,
        prior_approved=bool(prior_approved), relation_weight=relation_weight, learn=learn,
        linear=linear, parameters=sum(p.numel() for p in model.parameters()),
        metric=model.log_metric.detach().clamp(-3, 3).exp().tolist(),
        bandwidth=float(model.log_bandwidth.detach().clamp(-3, 3).exp()),
        ridge=float(model.log_ridge.detach().clamp(-9, 2).exp()))
    return fit


def validation_acceptance(y, groups, baseline, candidate):
    """Global, source-validation-only policy. No test data or tie-break permitted."""
    y, baseline, candidate = map(lambda v: np.asarray(v, float), (y, baseline, candidate))
    groups = np.asarray(groups)
    if (y.shape != baseline.shape or y.shape != candidate.shape or groups.shape != y.shape
            or not len(y) or any(not np.isfinite(v).all() for v in (y, baseline, candidate))):
        raise ValueError('aligned finite validation arrays required')
    b = np.array([np.mean((baseline[groups == g]-y[groups == g])**2) for g in np.unique(groups)])
    c = np.array([np.mean((candidate[groups == g]-y[groups == g])**2) for g in np.unique(groups)])
    ratios = np.sqrt(c/np.maximum(b, 1e-12))
    accepted = c.mean() < .98*b.mean() and np.mean(c < b) >= .6 and ratios.max() <= 1.05
    return dict(accepted=bool(accepted), unit_mse_ratio=float(c.mean()/max(b.mean(), 1e-12)),
                unit_win_fraction=float(np.mean(c < b)), maximum_unit_rmse_ratio=float(ratios.max()))
