import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Optional
from .constants import (
    LABEL_PADDING_VALUE, TASKS, LABEL_DOMAINS, PCSETS_FILEPATH
)


def _derive_chord_change(labels: torch.Tensor) -> torch.Tensor:
    """Derive binary chord-change labels from any pitch/RN label row."""
    out = torch.zeros_like(labels)
    out[:, 0] = 1
    out[:, 1:] = (labels[:, 1:] != labels[:, :-1]).long()
    out[labels == LABEL_PADDING_VALUE] = LABEL_PADDING_VALUE
    return out


def _build_equiv_mask(device: torch.device) -> torch.Tensor:
    """
    Precomputed at init. Shape: [n_keys, n_rns, n_rns]
    equiv_mask[ki, ri, rj] = 1.0 if rn rj is enharmonically
    equivalent to rn ri given key ki, else 0.0.
    """
    key_domain = LABEL_DOMAINS['global_key']
    rn_domain  = LABEL_DOMAINS['roman_numeral']
    n_keys = len(key_domain)
    n_rns  = len(rn_domain)

    with open(PCSETS_FILEPATH, 'r') as f:
        pcsets = json.load(f)

    mask = torch.zeros(n_keys, n_rns, n_rns)

    for ki, key in enumerate(key_domain):
        if key not in pcsets:
            for ri in range(n_rns):
                mask[ki, ri, ri] = 1.0
            continue

        pc_to_rns: Dict[tuple, List[int]] = {}
        for ri, rn in enumerate(rn_domain):
            if rn not in pcsets[key]:
                continue
            pc = tuple(sorted(pcsets[key][rn]))
            pc_to_rns.setdefault(pc, []).append(ri)

        for ri, rn in enumerate(rn_domain):
            if rn not in pcsets[key]:
                mask[ki, ri, ri] = 1.0
                continue
            pc = tuple(sorted(pcsets[key][rn]))
            for rj in pc_to_rns.get(pc, [ri]):
                mask[ki, ri, rj] = 1.0

    return mask.to(device)


# Tonal-distance weighted key criterion

def _build_key_distance_matrix(n_keys: int = 24) -> torch.Tensor:
    """
    Returns [n_keys, n_keys] tonal-distance matrix.

    Encoding: 0-11 = C_major … B_major, 12-23 = C_minor … B_minor
    (root = label % 12,  mode = label // 12)

    Distance = circle-of-fifths distance between roots
               + 1 if modes differ (parallel key penalty)

    Values range from 0 (same key) to 7 (tritone + mode flip).
    """
    cof_pos = torch.tensor([(7 * r) % 12 for r in range(12)], dtype=torch.float)

    dist = torch.zeros(n_keys, n_keys)
    for i in range(n_keys):
        for j in range(n_keys):
            ri, mi = i % 12, i // 12
            rj, mj = j % 12, j // 12
            cof_i  = int(cof_pos[ri].item())
            cof_j  = int(cof_pos[rj].item())
            cof_d  = min(abs(cof_i - cof_j), 12 - abs(cof_i - cof_j))
            mode_d = 0 if mi == mj else 1
            dist[i, j] = cof_d + mode_d

    return dist


class TonalKeyCriterion(nn.Module):
    """
    Cross-entropy loss for global_key with tonal-distance label smoothing.

    Soft targets: the true key gets weight (1 − alpha); remaining alpha is
    spread across other keys inversely proportional to their tonal distance.
    Nearby-key errors are cheap; tritone errors are expensive.
    """

    def __init__(
        self,
        n_keys    : int   = 24,
        alpha     : float = 0.15,
        ignore_idx: int   = LABEL_PADDING_VALUE,
    ) -> None:
        super().__init__()
        self.ignore_idx = ignore_idx
        self.alpha      = alpha

        dist = _build_key_distance_matrix(n_keys)
        inv  = 1.0 / (dist + 1e-6)
        inv.fill_diagonal_(0.0)
        smooth_weights = inv / inv.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        self.register_buffer("smooth_weights", smooth_weights)   # [K, K]

    def forward(
        self,
        logits : torch.Tensor,   # [B, T, K]
        targets: torch.Tensor,   # [B, T]
    ) -> torch.Tensor:
        K = logits.size(-1)

        flat_logits  = logits.reshape(-1, K)
        flat_targets = targets.reshape(-1)

        mask = flat_targets != self.ignore_idx
        if not mask.any():
            return flat_logits.sum() * 0.0

        logits_v  = flat_logits[mask]
        targets_v = flat_targets[mask]

        one_hot = torch.zeros_like(logits_v)
        one_hot.scatter_(1, targets_v.unsqueeze(1), 1.0)
        soft        = self.smooth_weights.to(targets_v.device)[targets_v]
        target_dist = (1.0 - self.alpha) * one_hot + self.alpha * soft

        log_probs = F.log_softmax(logits_v, dim=-1)
        return -(target_dist * log_probs).sum(dim=-1).mean()


# Boundary-upweighted loss helper

def _compute_boundary_weights(
    labels         : torch.Tensor,
    boundary_weight: float,
    margin         : int = 1,
    ignore_idx     : int = LABEL_PADDING_VALUE,
) -> torch.Tensor:
    """
    Build a per-frame weight tensor that upweights frames near chord boundaries.

    A "boundary" is any frame-pair (t, t+1) where the label changes.  Both the
    frame before and after the transition are upweighted, along with `margin`
    extra frames on each side - dilated with a 1-D max-pool so the computation
    is fully vectorised.

    Parameters
    labels          : [B, T]  reference labels (roman_numeral is standard)
    boundary_weight : multiplier at boundary frames, e.g. 2.0
    margin          : extra frames to include on each side (default 1)
    ignore_idx      : padding value excluded from transition detection

    Returns
    [B, T]  float - boundary_weight at boundary frames, 1.0 elsewhere
    """
    B, T = labels.shape
    weights = torch.ones(B, T, device=labels.device)

    if boundary_weight == 1.0 or T < 2:
        return weights

    valid   = (labels[:, :-1] != ignore_idx) & (labels[:, 1:] != ignore_idx)
    changed = (labels[:, :-1] != labels[:, 1:]) & valid   # [B, T-1]

    # Mark both sides of each transition
    boundary = torch.zeros(B, T, device=labels.device)
    boundary[:, :-1] += changed.float()
    boundary[:, 1:]  += changed.float()
    boundary = (boundary > 0)

    # Dilate by margin (max-pool preserves any True neighbour)
    if margin > 0:
        kernel   = 2 * margin + 1
        b_float  = boundary.float().unsqueeze(1)             # [B, 1, T]
        dilated  = F.max_pool1d(b_float, kernel_size=kernel,
                                stride=1, padding=margin)     # [B, 1, T]
        boundary = dilated.squeeze(1) > 0.5                  # [B, T]

    weights[boundary] = boundary_weight
    return weights


# Within-segment consistency loss

class SegmentConsistencyLoss(nn.Module):
    """
    Penalises prediction variance inside stable chord segments.

    For every adjacent frame-pair (t, t+1) that shares the same ground-truth
    label, the loss is the mean squared difference between their softmax
    probability vectors.  This stabilises within-chord predictions -
    eliminating the spurious drifts that produce short glitch segments in
    the output timeline.

    Complementary role
    - Boundary upweighting  -> "be decisive at transitions"
    - Consistency loss      -> "be stable within a chord"
    - Chord-change head     -> "explicitly detect transitions"

    Together they encourage the model to produce flat plateaus (stable chord
    regions) separated by sharp steps (boundaries), which is the ideal shape
    for a chord-recognition output.

    Parameters
    ignore_idx : padding value (LABEL_PADDING_VALUE = -1)
    """

    def __init__(self, ignore_idx: int = LABEL_PADDING_VALUE) -> None:
        super().__init__()
        self.ignore_idx = ignore_idx

    def forward(
        self,
        logits : torch.Tensor,   # [B, T, C]
        labels : torch.Tensor,   # [B, T]
    ) -> torch.Tensor:
        B, T, C = logits.shape
        if T < 2:
            return logits.sum() * 0.0

        probs = F.softmax(logits, dim=-1)   # [B, T, C]

        l1    = labels[:, :-1]
        l2    = labels[:, 1:]
        valid = (l1 != self.ignore_idx) & (l2 != self.ignore_idx)
        same  = (l1 == l2) & valid          # True only for within-chord pairs

        if not same.any():
            return logits.sum() * 0.0

        p1  = probs[:, :-1]                 # [B, T-1, C]
        p2  = probs[:, 1:]                  # [B, T-1, C]

        mse  = ((p1 - p2) ** 2).sum(dim=-1)                        # [B, T-1]
        loss = (mse * same.float()).sum() / same.float().sum().clamp(min=1.0)
        return loss


# Multi-task loss

class MultiTaskLoss(nn.Module):
    def __init__(
            self,
            tasks                : List[str],
            task_weights         : torch.Tensor            = None,
            task_classes_weights : Dict[str, torch.Tensor] = None,
            reduction            : str                     = 'mean',
            device               : torch.device            = 'cuda',
            label_smoothing      : float                   = 0.0,
            use_equiv_loss       : bool                    = False,
            add_chord_change_head: bool                    = False,
            # Key criterion
            use_tonal_key_loss   : bool                    = True,
            tonal_key_alpha      : float                   = 0.15,
            # Boundary-upweighted loss
            boundary_weight      : float                   = 1.0,
            boundary_margin      : int                     = 1,
            # Within-segment consistency loss
            consistency_weight   : float                   = 0.0,
            consistency_tasks    : Optional[List[str]]     = None,
    ):
        """
        boundary_weight : float (default 1.0 = off)
            CE loss multiplier for frames near chord boundaries.  The boundary
            is detected from roman_numeral labels and applied to ALL frame-level
            task heads simultaneously, so every task head gets the same stronger
            gradient signal at transition points.  Recommended: 2.0–3.0.

        boundary_margin : int (default 1)
            Extra frames to include on each side of a detected boundary.  At
            46 ms/frame, margin=1 covers ±1 frame (3 frames total per boundary
            = ~140 ms).

        consistency_weight : float (default 0.0 = off)
            Weight of the SegmentConsistencyLoss added after the main CE loss.
            Recommended: 0.05–0.1.  Start at 0.05 - too high can suppress
            necessary within-chord variation (e.g. passing tones).

        consistency_tasks : list[str] | None
            Tasks to apply consistency to.  Defaults to the five primary
            frame-level tasks: roman_numeral, root_scale_degree, quality,
            root_pitch_class, bass_pitch_class.
        """
        super().__init__()
        assert reduction in ('sum', 'mean')

        self.tasks                = tasks
        self.reduction            = reduction
        self.use_equiv_loss       = use_equiv_loss
        self.add_chord_change_head= add_chord_change_head
        self.label_smoothing      = label_smoothing
        self.boundary_weight      = boundary_weight
        self.boundary_margin      = boundary_margin
        self.consistency_weight   = consistency_weight

        self.rn_task_idx  = tasks.index('roman_numeral')
        self.key_task_idx = tasks.index('global_key')

        self.task_weights = task_weights
        if self.task_weights is None:
            self.task_weights = torch.ones(len(tasks)).to(device)

        # Build criterion objects.
        # For non-key tasks the criterion is stored so we can read its
        # class_weight in forward; the actual reduction is done manually
        # so that per-frame boundary weights can be applied.
        self.task_criterions = nn.ModuleDict()

        for task in tasks:
            if task == 'global_key' and use_tonal_key_loss:
                self.task_criterions[task] = TonalKeyCriterion(
                    n_keys    = 24,
                    alpha     = tonal_key_alpha,
                    ignore_idx= LABEL_PADDING_VALUE,
                )
            elif task_classes_weights is not None and task in task_classes_weights:
                self.task_criterions[task] = nn.CrossEntropyLoss(
                    weight         = task_classes_weights[task],
                    ignore_index   = LABEL_PADDING_VALUE,
                    label_smoothing= label_smoothing,
                )
            else:
                self.task_criterions[task] = nn.CrossEntropyLoss(
                    ignore_index   = LABEL_PADDING_VALUE,
                    label_smoothing= label_smoothing,
                )

        if add_chord_change_head:
            self.task_criterions['chord_change'] = nn.CrossEntropyLoss(
                ignore_index=LABEL_PADDING_VALUE,
            )

        if use_equiv_loss:
            equiv_mask = _build_equiv_mask(device)
            self.register_buffer('equiv_mask', equiv_mask)

        # Consistency loss
        if consistency_weight > 0.0:
            self.consistency_criterion = SegmentConsistencyLoss(
                ignore_idx=LABEL_PADDING_VALUE,
            )
            self.consistency_tasks = consistency_tasks or [
                'roman_numeral',
                'root_scale_degree',
                'quality',
                'root_pitch_class',
                'bass_pitch_class',
            ]
        else:
            self.consistency_criterion = None
            self.consistency_tasks     = []

    # Internal helpers

    def _per_frame_ce(
        self,
        output   : torch.Tensor,   # [B, T, C]
        targets_t: torch.Tensor,   # [B, T]
        task     : str,
    ) -> torch.Tensor:
        """
        Per-frame CE [B, T] using stored class weights and module label_smoothing.
        Returns 0.0 at padding positions (F.cross_entropy with ignore_index).
        """
        criterion = self.task_criterions[task] if task in self.task_criterions else None
        class_w   = getattr(criterion, 'weight', None)

        return F.cross_entropy(
            output.transpose(1, 2),
            targets_t,
            weight         = class_w,
            ignore_index   = LABEL_PADDING_VALUE,
            label_smoothing= self.label_smoothing,
            reduction      = 'none',
        )   # [B, T]

    def _weighted_mean(
        self,
        per_frame       : torch.Tensor,   # [B, T]
        targets_t       : torch.Tensor,   # [B, T]
        boundary_weights: torch.Tensor,   # [B, T]
    ) -> torch.Tensor:
        """Weighted mean over non-padding frames."""
        mask    = (targets_t != LABEL_PADDING_VALUE).float()
        n_valid = mask.sum().clamp(min=1.0)
        return (per_frame * boundary_weights * mask).sum() / n_valid

    def _equiv_cross_entropy(
        self,
        logits  : torch.Tensor,
        true_rn : torch.Tensor,
        true_key: torch.Tensor,
    ) -> torch.Tensor:
        pad  = LABEL_PADDING_VALUE
        mask = (true_rn != pad) & (true_key != pad)
        if not mask.any():
            return torch.zeros((), device=logits.device)

        logits_valid = logits[mask]
        rn_valid     = true_rn[mask]
        key_valid    = true_key[mask]

        soft_targets = self.equiv_mask[key_valid, rn_valid]
        soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdims=True).clamp(min=1e-8)

        log_probs = F.log_softmax(logits_valid, dim=-1)
        return -(soft_targets * log_probs).sum(dim=-1).mean()

    # Forward

    def forward(
            self,
            outputs: Dict[str, torch.Tensor],
            targets: torch.Tensor
    ) -> torch.Tensor:
        main_loss   = 0.0
        valid_tasks = 0

        # Boundary weights from roman_numeral labels.
        # Applied to every frame-level task head so all tasks receive
        # a consistent "commit at boundaries" signal.
        rn_labels        = targets[:, self.rn_task_idx]
        boundary_weights = _compute_boundary_weights(
            rn_labels,
            self.boundary_weight,
            self.boundary_margin,
        )   # [B, T]

        for task, output in outputs.items():

            # Chord-change auxiliary head
            if task == 'chord_change':
                if not self.add_chord_change_head:
                    continue
                cc_labels = _derive_chord_change(rn_labels)
                if (cc_labels == LABEL_PADDING_VALUE).all():
                    continue
                valid_tasks += 1
                # Boundary upweighting here is particularly potent: the frames
                # the model most needs to get right (change vs. no-change) are
                # precisely the boundary frames that receive the higher weight.
                per_frame = F.cross_entropy(
                    output.transpose(1, 2),
                    cc_labels,
                    ignore_index=LABEL_PADDING_VALUE,
                    reduction='none',
                )
                main_loss += self._weighted_mean(per_frame, cc_labels, boundary_weights)
                continue

            task_idx = self.tasks.index(task)

            if (targets[:, task_idx] == LABEL_PADDING_VALUE).all():
                continue

            valid_tasks += 1

            # Global-key head (TonalKeyCriterion - segment level)
            # Boundary weighting is not applied: global_key is a CLS-token
            # prediction broadcast uniformly to all frames, so all T frames
            # carry the same logits and boundary weighting would have no effect.
            if task == 'global_key':
                task_loss = self.task_criterions[task](
                    output,
                    targets[:, task_idx],
                )

            # Roman numeral with enharmonic equivalence
            elif self.use_equiv_loss and task == 'roman_numeral':
                # Equiv loss uses soft targets; boundary weighting is skipped
                # here to keep the code simple. The chord-change head and
                # consistency loss still provide boundary supervision for rn.
                task_loss = self._equiv_cross_entropy(
                    output,
                    targets[:, self.rn_task_idx],
                    targets[:, self.key_task_idx],
                )

            # All other frame-level tasks (boundary-upweighted CE)
            else:
                per_frame = self._per_frame_ce(output, targets[:, task_idx], task)
                task_loss = self._weighted_mean(
                    per_frame, targets[:, task_idx], boundary_weights
                )

            main_loss += self.task_weights[task_idx] * task_loss

        if self.reduction == 'mean':
            main_loss /= max(valid_tasks, 1)

        # Within-segment consistency loss
        # Added as a separate term after the mean-reduced main loss so that
        # consistency_weight has a stable, interpretable scale: 0.05 means
        # the consistency term contributes ~5% of the averaged CE loss.
        consistency_loss = 0.0
        if self.consistency_weight > 0.0 and self.consistency_criterion is not None:
            n_consistency = 0
            for task in self.consistency_tasks:
                if task not in outputs:
                    continue
                task_idx = self.tasks.index(task)
                if (targets[:, task_idx] == LABEL_PADDING_VALUE).all():
                    continue
                consistency_loss += self.consistency_criterion(
                    outputs[task],
                    targets[:, task_idx],
                )
                n_consistency += 1
            if n_consistency > 0:
                consistency_loss /= n_consistency

        return main_loss + self.consistency_weight * consistency_loss