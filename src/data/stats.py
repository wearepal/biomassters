from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import torch
from torch import Tensor

from src.data.dataset import SentinelDataset
from src.utils import some, torch_eps

__all__ = [
    "CStatsPair",
    "ChannelStatistics",
]


@dataclass(unsafe_hash=True)
class ChannelStatistics:

    missing_value: Optional[SentinelDataset.MissingValue] = None

    _mean: Optional[Tensor] = field(init=False, default=None)
    _std: Optional[Tensor] = field(init=False, default=None)
    _var: Optional[Tensor] = field(init=False, default=None)
    _min: Optional[Tensor] = field(init=False, default=None)
    _max: Optional[Tensor] = field(init=False, default=None)
    _n: Optional[Tensor] = field(init=False, default=None)
    _n_var: Optional[Tensor] = field(init=False, default=None)

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
    def range(self) -> Tensor:
        return self.max - self.min

    @property
    def var(self) -> Tensor:
        if self._var is None or (self._n_var is None):
            raise AttributeError("No updates have been computed yet.")
        elif self._n is None:
            raise RuntimeError(
                "Mean must be computed with 'update' before variance can be computed."
            )
        elif (self._n_var < self._n).any():
            raise RuntimeError(
                "The number of samples used to compute the variance is less than the number "
                "used to compute the mean."
            )
        return self._var

    @property
    def std(self) -> Tensor:
        return self.var.clamp_min(torch_eps(self.var)).sqrt()

    def update_var(self, batch: Tensor) -> None:
        if self._n is None:
            raise RuntimeError(
                "Mean must be computed with 'update' before variance can be computed."
            )
        if some(self._n_var) and (self._n_var > self._n).any():
            raise RuntimeError(
                "More samples passed to compute variance than were used to compute the mean."
            )
        batch_t = batch.transpose(0, 1)
        batch_t_flat = batch_t.flatten(start_dim=1)
        mean = self.mean.unsqueeze(-1)
        mask: Optional[Tensor] = None
        if self.missing_value:
            mask = self.missing_value.checker(batch_t_flat)
            counts = (~mask).count_nonzero(dim=1)
        else:
            counts = torch.tensor(
                (batch_t_flat.size(1),) * batch_t_flat.size(0), device=batch.device
            )

        sq_err = (batch_t_flat - mean).pow(2)
        if some(mask):
            sq_err[mask] = 0.0
        batch_var = sq_err.sum(dim=1) / self._n
        if self._var is None:
            self._var = batch_var
        else:
            self._var += batch_var

        if some(self._n_var):
            self._n_var += counts
        else:
            self._n_var = counts

    def update(self, batch: Tensor) -> None:
        batch_t = batch.movedim(1, 0)
        old_n = 0.0 if self._n is None else self._n
        batch_t_flat = batch_t.flatten(start_dim=1)
        mask: Optional[Tensor] = None
        if self.missing_value:
            mask = self.missing_value.checker(batch_t_flat)
            counts = (~mask).count_nonzero(dim=1)
        else:
            counts = torch.tensor(
                (batch_t_flat.size(1),) * batch_t_flat.size(0), device=batch.device
            )

        new_n = old_n + counts
        if some(mask):
            batch_t_flat[mask] = torch.inf
        batch_min = batch_t_flat.min(dim=1).values

        if self._min is None:
            self._min = batch_t_flat.min(dim=1).values
        else:
            self._min = self._min.min(batch_min)

        if some(mask):
            batch_t_flat[mask] = -torch.inf
        batch_max = batch_t_flat.max(dim=1).values

        if self._max is None:
            self._max = batch_max
        else:
            self._max = self._max.max(batch_max)

        if some(mask):
            batch_t_flat[mask] = 0.0
        batch_mean = (batch_t_flat / new_n.unsqueeze(1)).sum(dim=1)
        if self._mean is None:
            self._mean = batch_mean
        else:
            # Avoid potential overflow by dividing then multiplying the mean.
            self._mean = ((self.mean / new_n) * old_n) + batch_mean
        self._n = new_n


class CStatsPair(NamedTuple):
    input: ChannelStatistics
    target: ChannelStatistics
