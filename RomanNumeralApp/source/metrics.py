import json
import torch
import numpy as np

from typing import Dict
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support
from .constants import TASKS, LABEL_DOMAINS, LABEL_PADDING_VALUE, PCSETS_FILEPATH


class MultiTaskMetrics:
    def __init__(self, split: str):
        assert split in ('train', 'valid', 'test')
        self.split = split

        self.num_batches = 0
        self.metrics = defaultdict(int)

    def update(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor]
    ) -> None:
        """ Updates metric scores with the current batch of outputs and labels. """
        matches = {}
        rn_mask = None

        if not isinstance(labels, np.ndarray):
            labels = labels.detach().cpu().numpy()
            outputs = {task: output.detach().cpu().numpy() for task, output in outputs.items()}

        # Default metrics
        for idx, task in enumerate(TASKS):
            y_true = labels[:, idx].flatten()
            y_pred = outputs[task].argmax(axis=-1).flatten()

            mask = (y_true != LABEL_PADDING_VALUE)
            if task == 'roman_numeral':
                rn_mask = mask

            matches[task] = (y_pred == y_true)
            self.metrics[f'{self.split}_{task}_acc'] += np.mean(y_pred[mask] == y_true[mask])

            macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
                y_true[mask], y_pred[mask], average='macro', zero_division=0
            )
            weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
                y_true[mask], y_pred[mask], average='weighted', zero_division=0
            )

            self.metrics[f'{self.split}_{task}_macro_f1'] += macro_f1
            self.metrics[f'{self.split}_{task}_weighted_f1'] += weighted_f1

            self.metrics[f'{self.split}_{task}_macro_recall'] += macro_recall
            self.metrics[f'{self.split}_{task}_weighted_recall'] += weighted_recall

            self.metrics[f'{self.split}_{task}_macro_precision'] += macro_precision
            self.metrics[f'{self.split}_{task}_weighted_precision'] += weighted_precision
        
        # Task combination accuracies
        rn_alt_matches = matches['global_key'] * matches['roman_numeral'] * matches['inversion']
        self.metrics[f'{self.split}_rn_alt_acc'] += np.mean(rn_alt_matches[rn_mask])

        rn_conv_matches = matches['global_key']
        for task in ['root_scale_degree', 'tonicization', 'quality', 'inversion', 'root_pitch_class']:
            rn_conv_matches *= matches[task]
        
        degree_matches = matches['root_scale_degree'] * matches['tonicization']
        
        self.metrics[f'{self.split}_degree_acc'] += np.mean(degree_matches[rn_mask])
        self.metrics[f'{self.split}_rn_conv_acc'] += np.mean(rn_conv_matches[rn_mask])

        self.num_batches += 1

    def compute(self) -> Dict[str, float]:
        return {key: value / self.num_batches for key, value in self.metrics.items()}        
        

class EquivalentAwareMetrics:
    def __init__(self, split: str):
        assert split in ('train', 'val', 'test')
        self.split = split

        self.num_batches = 0
        self.metrics = defaultdict(int)

        self.roman_numeral_domain = np.array(LABEL_DOMAINS['roman_numeral'])
        self.global_key_domain = np.array(LABEL_DOMAINS['global_key'])

        with open(PCSETS_FILEPATH, 'r') as fp:
            self.pcsets = json.load(fp)
            self.get_rn_pitch_classes = np.vectorize(lambda key, rn: self.pcsets[key][rn])

    def update(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor]
    ) -> None:
        """ Updates the accuracy score with the current batch of outputs and labels. """

        if not isinstance(labels, np.ndarray):
            labels = labels.detach().cpu().numpy()
            outputs = {task: output.detach().cpu().numpy() for task, output in outputs.items()}

        y_true_rn_idxs = labels[:, TASKS.index('roman_numeral')].flatten()
        y_true_key_idxs = labels[:, TASKS.index('global_key')].flatten()

        y_pred_rn_idxs = outputs['roman_numeral'].argmax(axis=-1).flatten()
        y_pred_key_idxs = outputs['global_key'].argmax(axis=-1).flatten()

        y_true_rns = self.roman_numeral_domain[y_true_rn_idxs]
        y_true_keys = self.global_key_domain[y_true_key_idxs]
        y_true_pitch_classes = self.get_rn_pitch_classes(y_true_keys, y_true_rns)

        y_pred_rns = self.roman_numeral_domain[y_pred_rn_idxs]
        y_pred_keys = self.global_key_domain[y_pred_key_idxs]
        y_pred_pitch_classes = self.get_rn_pitch_classes(y_pred_keys, y_pred_rns)

        mask = (y_true_rn_idxs != LABEL_PADDING_VALUE)
        matches = (y_pred_pitch_classes == y_true_pitch_classes)[mask]
        self.metrics[f'{self.split}_equiv_rn_acc'] += np.mean(matches)

        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true_pitch_classes[mask], y_pred_pitch_classes[mask], average='macro', zero_division=0
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true_pitch_classes[mask], y_pred_pitch_classes[mask], average='weighted', zero_division=0
        )

        self.metrics[f'{self.split}_equiv_rn_macro_f1'] += macro_f1
        self.metrics[f'{self.split}_equiv_rn_weighted_f1'] += weighted_f1

        self.metrics[f'{self.split}_equiv_rn_macro_recall'] += macro_recall
        self.metrics[f'{self.split}_equiv_rn_weighted_recall'] += weighted_recall

        self.metrics[f'{self.split}_equiv_rn_macro_precision'] += macro_precision
        self.metrics[f'{self.split}_equiv_rn_weighted_precision'] += weighted_precision

        self.num_batches += 1

    def compute(self) -> Dict[str, float]:
        return {key: value / self.num_batches for key, value in self.metrics.items()}


