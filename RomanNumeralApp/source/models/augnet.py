import torch
import torch.nn as nn

from torch import Tensor
from typing import Iterable, Dict
from .common import PreprocessingBlock, MultiTaskPredictor


class Conv1DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        use_relu: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.ReLU() if use_relu else nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.conv(x)


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels: int = 19, num_blocks: int = 6, **kwargs):
        super().__init__()

        in_channels = in_channels
        self.convs = nn.ModuleList()
        
        for i in range(num_blocks):
            kernel_size = 2**i
            out_channels = 2**(num_blocks-1-i)

            self.convs.append(Conv1DBlock(in_channels, out_channels, kernel_size, **kwargs))
            in_channels += out_channels

    def forward(self, x):
        for conv in self.convs:
            out = conv(x)
            x = torch.cat([x, out], dim=1)

        return x
    

class DenseBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        factor_multiplier: int = 1,
        use_relu: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()

        self.dense = nn.Linear(in_features, out_features * factor_multiplier)
        self.batchnorm = nn.BatchNorm1d(out_features * factor_multiplier)
        self.activation = nn.ReLU() if use_relu else nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dense(x)
        x = x.transpose(1, 2)

        x = self.batchnorm(x)
        x = x.transpose(1, 2)

        x = self.activation(x)
        x = self.dropout(x)

        return x


class BiGRUBlock(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, factor_multiplier: int = 1):
        super().__init__()

        self.tanh = nn.Tanh()
        self.batchnorm = nn.BatchNorm1d(2 * hidden_size * factor_multiplier)
        self.gru = nn.GRU(input_size, hidden_size * factor_multiplier, batch_first=True, bidirectional=True)

    def forward(self, x):
        x, _ = self.gru(x)
        x = x.transpose(1, 2)

        x = self.batchnorm(x)
        x = x.transpose(1, 2)

        x = self.tanh(x)
        return x


class AugmentedNet(nn.Module):
    def __init__(self, in_channels: Iterable[int], task_sizes: Dict[str, int], **kwargs):
        super().__init__()
        
        num_features_after_convs = 0
        self.conv_blocks = nn.ModuleList()
        for num_channels in in_channels:
            self.conv_blocks.append(ConvolutionalBlock(num_channels, **kwargs))
            num_features_after_convs += num_channels + (32 + 16 + 8 + 4 + 2 + 1)

        self.middle_part = nn.Sequential(
            DenseBlock(in_features=num_features_after_convs, out_features=64, **kwargs),
            DenseBlock(in_features=64, out_features=32, **kwargs),
            BiGRUBlock(input_size=32, hidden_size=30),
            BiGRUBlock(input_size=60, hidden_size=30)
        )

        self.mtl_head = MultiTaskPredictor(in_features=60, task_sizes=task_sizes)

    def forward(self, xs: Iterable[Tensor]):
        common_embedding = [self.conv_blocks[i](xs[i]) for i in range(len(xs))]
        common_embedding = torch.cat(common_embedding, dim=1)

        common_embedding = common_embedding.transpose(1, 2)
        common_embedding = self.middle_part(common_embedding)

        return self.mtl_head(common_embedding)
    

class AudioAugmentedNet(nn.Module):
    def __init__(
        self,
        in_channels: Iterable[int],
        task_sizes: Dict[str, int],
        factor_multiplier: int = 2,
        use_preprocessing: bool = True,
        dropout: float = 0.5,
        **kwargs
    ):
        super().__init__()
        
        num_features_after_convs = 0
        self.conv_blocks = nn.ModuleList()
        for num_channels in in_channels:
            if use_preprocessing:
                self.conv_blocks.append(nn.Sequential(
                    PreprocessingBlock(num_channels, dropout=dropout, **kwargs),
                    ConvolutionalBlock(96, dropout=dropout, **kwargs)
                ))

                num_features_after_convs += 96 + (32 + 16 + 8 + 4 + 2 + 1)
            else:
                self.conv_blocks.append(ConvolutionalBlock(num_channels, dropout=dropout, **kwargs))
                num_features_after_convs += num_channels + (32 + 16 + 8 + 4 + 2 + 1)

        self.middle_part = nn.Sequential(
            DenseBlock(in_features=num_features_after_convs, out_features=64*factor_multiplier, dropout=dropout, **kwargs),
            DenseBlock(in_features=64*factor_multiplier, out_features=32*factor_multiplier, dropout=dropout, **kwargs),
            BiGRUBlock(input_size=32*factor_multiplier, hidden_size=30*factor_multiplier),
            BiGRUBlock(input_size=2*30*factor_multiplier, hidden_size=30*factor_multiplier)
        )

        self.mtl_head = MultiTaskPredictor(in_features=2*30*factor_multiplier, task_sizes=task_sizes)

    def forward(self, xs: Iterable[Tensor]):
        common_embedding = [self.conv_blocks[i](xs[i]) for i in range(len(xs))]
        common_embedding = torch.cat(common_embedding, dim=1)

        common_embedding = common_embedding.transpose(1, 2)
        common_embedding = self.middle_part(common_embedding)

        return self.mtl_head(common_embedding)
