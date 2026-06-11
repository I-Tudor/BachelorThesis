import random
import numpy as np
from .constants import TASKS, LABEL_PADDING_VALUE

# Indices into the full TASKS list that encode absolute pitch class (mod-12 shift)
_PITCH_CLASS_TASK_IDXS = {
    TASKS.index('root_pitch_class'),
    TASKS.index('bass_pitch_class'),
    TASKS.index('tonicized_pitch_class'),
}
_GLOBAL_KEY_IDX = TASKS.index('global_key')

# LABEL_DOMAINS encodes global_key as 0-11 = major (C..B), 12-23 = minor (C..B)
# i.e. root-grouped layout A: root = label % 12, mode = label // 12
_KEY_N_CLASSES = 24


def _shift_key(labels_row: np.ndarray, k: int) -> np.ndarray:
    """Shift a global_key label row by k semitones. Layout: 0-11 major, 12-23 minor."""
    out = labels_row.copy()
    mask = labels_row != LABEL_PADDING_VALUE
    valid = labels_row[mask]
    root = valid % 12
    mode = valid // 12          # 0 = major, 1 = minor
    out[mask] = (root + k) % 12 + mode * 12
    return out


def _shift_pitch_class(labels_row: np.ndarray, k: int) -> np.ndarray:
    """Shift a 12-class pitch label row by k semitones."""
    out = labels_row.copy()
    mask = labels_row != LABEL_PADDING_VALUE
    out[mask] = (labels_row[mask] + k) % 12
    return out


def pitch_transpose(
    features: list,        # list of np.ndarray [C, T], C=12 for chroma/bass
    labels: np.ndarray,    # [n_tasks, T] full label array (before task_idxs slicing)
    p: float = 0.8,
) -> tuple:
    """
    Randomly transpose all features and pitch-sensitive labels by k semitones.
    Only call this during training, on the full labels array before task_idxs slicing.

    Args:
        features: list of [C, T] numpy arrays (chroma, basschroma)
        labels:   [n_tasks, T] numpy array (all tasks, not yet sliced by task_idxs)
        p:        probability of applying augmentation per sample

    Returns:
        (aug_features, aug_labels) - new arrays, originals untouched
    """
    if random.random() > p:
        return features, labels

    k = random.randint(1, 11)

    # features: cyclic roll along the channel axis (axis=0 for [C, T])
    aug_features = [np.roll(f, shift=k, axis=0) for f in features]

    # labels: copy and shift only pitch-sensitive rows
    aug_labels = labels.copy()

    aug_labels[_GLOBAL_KEY_IDX] = _shift_key(labels[_GLOBAL_KEY_IDX], k)

    for task_idx in _PITCH_CLASS_TASK_IDXS:
        if task_idx < len(labels):
            aug_labels[task_idx] = _shift_pitch_class(labels[task_idx], k)

    # All other tasks (root_scale_degree, quality, inversion,
    # tonicization, roman_numeral) are relative to key - unchanged.

    return aug_features, aug_labels