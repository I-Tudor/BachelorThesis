import torch
import torch.nn as nn
from typing import Iterable, Dict
from .common import PreprocessingBlock, MultiTaskPredictor


class BiGRU(nn.Module):
    def __init__(self, in_channels: Iterable[int], task_sizes: Dict[str, int], hidden_size: int = 128, dropout: float = 0.5):
        super().__init__()

        num_features_after_convs = 0
        self.conv_blocks = nn.ModuleList()
        for num_channels in in_channels:
            self.conv_blocks.append(
                PreprocessingBlock(num_channels, dropout=dropout)
            )
            num_features_after_convs += 96
            
        self.tanh = nn.Tanh()
        self.batchnorm = nn.BatchNorm1d(2 * hidden_size)
        self.gru = nn.GRU(num_features_after_convs, hidden_size, batch_first=True, bidirectional=True)
        self.mtl_head = MultiTaskPredictor(in_features=2*hidden_size, task_sizes=task_sizes)

    def forward(self, xs: Iterable[torch.Tensor]):
        common_embedding = [self.conv_blocks[i](xs[i]) for i in range(len(xs))]
        common_embedding = torch.cat(common_embedding, dim=1)

        common_embedding = common_embedding.transpose(1, 2)
        common_embedding, _ = self.gru(common_embedding)

        common_embedding = common_embedding.transpose(1, 2)
        common_embedding = self.tanh(self.batchnorm(common_embedding))

        common_embedding = common_embedding.transpose(1, 2)
        return self.mtl_head(common_embedding)
