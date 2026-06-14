import torch
import torch.nn as nn
from source.constants import LABEL_DOMAINS, LABEL_SIZES


class NaiveBaseline(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

        self.most_common_indexes = {
            'global_key': LABEL_DOMAINS['global_key'].index('C major'),
            'root_scale_degree': LABEL_DOMAINS['root_scale_degree'].index('1'),
            'tonicization': LABEL_DOMAINS['tonicization'].index('0'),
            'quality': LABEL_DOMAINS['quality'].index('M'),
            'inversion': LABEL_DOMAINS['inversion'].index(0),
            'root_pitch_class': 0,
            'bass_pitch_class': 0,
            'tonicized_pitch_class': 0,
            'roman_numeral': LABEL_DOMAINS['roman_numeral'].index('I')
        }

    def forward(self, x):
        batch_size, _, sequence_length = x[0].shape

        outputs = {}
        for task, size in LABEL_SIZES.items():
            # Setting 10000 for the most common index and -10000 for others to softmax of most common be the highest
            output = torch.full((batch_size, sequence_length, size), fill_value=-10000.0, device=self.device)
            output[:, :, self.most_common_indexes[task]] = 10000
            outputs[task] = output
            
        return outputs
