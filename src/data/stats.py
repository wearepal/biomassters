from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
from torch import Tensor

__all__ = [
    "CStatsPair",
    "ChannelStatistics",
]


@dataclass(unsafe_hash=True)
class ChannelStatistics:

    _mean: Optional[Tensor] = None
    _std: Optional[Tensor] = None
    _var: Optional[Tensor] = None
    _min: Optional[Tensor] = None
    _max: Optional[Tensor] = None
    _n: int = 0
    _n_var: int = 0

    @property
    def mean(self) -> Tensor:
        if self._mean is None:
            raise AttributeError("No statistics have been computed yet.")
        return self._mean

    @property
    def min(self) -> Tensor:
        if self._min is None:
            raise AttributeError("No statistics have been computed yet.")
        return self._min

    @property
    def max(self) -> Tensor:
        if self._max is None:
            raise AttributeError("No statistics have been computed yet.")
        return self._max

    @property
    def var(self) -> Tensor:
        if self._var is None:
            raise AttributeError("No updates have been computed yet.")
        if self._n_var < self._n:
            raise RuntimeError(
                "The number of samples used to compute the variance is less than the number "
                "used to compute the mean."
            )
        return self._var

    @property
    def std(self) -> Tensor:
        eps = torch.finfo(self.var.dtype).eps
        return self.var.clamp_min(eps).sqrt()

    def update_var(self, batch: Tensor) -> None:
        self._n_var += len(batch)
        if self._n_var > self._n:
            raise RuntimeError(
                "More samples passed to compute variance than were used to compute the mean."
            )
        mean = self.mean.view(1, -1, *((batch.ndim - 2) * (1,)))
        batch_var = (batch - mean).pow(2).transpose(0, 1).flatten(start_dim=1).sum(dim=1) / self._n
        if self._var is None:
            self._var = batch_var
        else:
            self._var += batch_var

    def update(self, batch: Tensor) -> None:
        batch_t = batch.movedim(1, 0)
        old_n = self._n
        new_n = self._n + batch_t[0].numel()
        batch_t_flat = batch_t.flatten(start_dim=1)
        batch_min = batch_t_flat.min(dim=1).values
        if self._min is None:
            self._min = batch_min
        else:
            self._min = self._min.min(batch_min)
        batch_max = batch_t_flat.max(dim=1).values
        if self._max is None:
            self._max = batch_max
        else:
            self._max = self._max.max(batch_max)
        if self._mean is None:
            self._mean = batch_t_flat.mean(dim=1)
        else:
            batch_mean = (batch_t_flat / new_n).sum(dim=1)
            # Avoid potential overflow by dividing then multiplying the mean.
            self._mean = ((self.mean / new_n) * old_n) + batch_mean
        self._n = new_n


class CStatsPair(NamedTuple):
    input: ChannelStatistics
    target: ChannelStatistics
