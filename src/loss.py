"""Quantile metrics for forecasting multiple quantiles per time step."""
from typing import List

from ranzen.torch.loss import ReductionType, reduce
import torch
from torch import Tensor
import torch.nn as nn
from typing_extensions import override

__all__ = ["QuantileLoss"]


class QuantileLoss(nn.Module):
    """
    Quantile loss, i.e. a quantile of ``q=0.5`` will give half of the mean absolute error as it is calcualted as

    Defined as ``max(q * (y-y_pred), (1-q) * (y_pred-y))``
    """

    def __init__(
        self,
        quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
        *,
        reduction: ReductionType = ReductionType.mean,
    ) -> None:
        super().__init__()
        self.quantiles = quantiles
        self.reduction = reduction

    @override
    def forward(self, y_pred: Tensor, *, y_true: Tensor) -> Tensor:
        # calculate quantile loss
        loss = y_pred.new_zeros(())
        for i, q in enumerate(self.quantiles):
            errors_i = y_true - y_pred[..., i]
            losses_i = 2 * torch.max((q - 1) * errors_i, q * errors_i).unsqueeze(-1)
            loss += reduce(losses=losses_i, reduction_type=self.reduction)
        return loss
