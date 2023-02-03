from typing import Optional

import torch
from torch import Tensor, nn
from typing_extensions import override

from src.utils import unwrap_or

__all__ = ["TrainableImputer"]


class TrainableImputer(nn.Module):
    def __init__(self, in_channels: Optional[int]) -> None:
        super().__init__()
        self.in_channels = unwrap_or(in_channels, default=1)
        shape = (self.in_channels,) if self.in_channels > 1 else ()
        self.values = nn.Parameter(torch.zeros(shape))

    @override
    def forward(self, x: Tensor, *, mask: Tensor) -> Tensor:
        if self.in_channels == 1:
            x.masked_fill_(mask=mask, value=self.values)
        else:
            values = self.values.view(1, -1, *((1,) * (x.ndim - 2))).expand_as(mask)
            x[mask] = 0.0
            x = x + mask * values
        return x
