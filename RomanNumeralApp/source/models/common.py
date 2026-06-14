import torch.nn as nn
from typing import Dict


class Conv2DBlock(nn.Module):
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
            nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if use_relu else nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.conv(x)
    

class PreprocessingBlock(nn.Module):
    def __init__(self, in_features: int, **kwargs):
        super().__init__()
        
        assert in_features % 12 == 0
        self.num_channels = in_features // 12

        self.convs = nn.Sequential(
            Conv2DBlock(self.num_channels, 32, kernel_size=(7,5), **kwargs),
            Conv2DBlock(32, 16, kernel_size=(7,5), **kwargs),
            Conv2DBlock(16, 8, kernel_size=(7,5), **kwargs)
        )

    def forward(self, x):
        batch_size, _, seq_length = x.size()
        x = x.view(batch_size, self.num_channels, 12, seq_length)

        x = self.convs(x)
        x = x.view(batch_size, -1, seq_length)

        return x
    

class MultiTaskPredictor(nn.Module):
    def __init__(self, in_features: int, task_sizes: Dict[str, int]):
        super().__init__()

        self.denses = nn.ModuleDict()
        for task, size in task_sizes.items():
            self.denses[task] = nn.Linear(in_features, size)

    def forward(self, x):
        return {task: dense(x) for task, dense in self.denses.items()}
