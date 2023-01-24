from typing import List

from ranzen.torch.loss import ReductionType, reduce
import torch
from torch import Tensor
import torch.nn as nn
from typing_extensions import override

from src.utils import torch_eps

__all__ = [
    "CharbonnierLoss",
    "QuantileLoss",
    "stable_mse_loss",
    "CCCLoss",
]


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


def stable_mse_loss(input: Tensor, *, target: Tensor, samplewise: bool = False) -> Tensor:
    """
    A more numerically stable version of MSE that simply reorders the
    exponentiation and normalisation operations -- useful for avoiding overflow when
    working with half precision.
    """
    diff = input - target
    reduction_dim = 1 if samplewise else 0
    diff = diff.flatten(start_dim=reduction_dim)
    return ((diff / diff.size(reduction_dim)) * diff).sum(reduction_dim)


class CharbonnierLoss(nn.Module):
    """
    The Charbonnier loss, also known as the pseudo Huber loss, or L1-L2 loss.
    """

    def __init__(
        self,
        alpha: float = 1,
        *,
        eps: float = 1.0e-5,
        reduction: ReductionType = ReductionType.mean,
    ) -> None:
        super().__init__()
        self.eps_sq = eps**2
        self.exponent = alpha / 2.0
        self.reduction = reduction

    @override
    def forward(self, input: Tensor, *, target: Tensor) -> Tensor:
        losses = ((input - target).pow(2) + self.eps_sq).pow(self.exponent)
        return reduce(losses=losses, reduction_type=self.reduction)


class CCCLoss(nn.Module):
    r"""
    Loss based on (Lin's) Concordance Correlation Coefficient.

    The concordance correlation coefficient measures the agreement between two
    variables, e.g., to evaluate reproducibility or for inter-rater
    reliability.

    The coefficient takes the form:

      .. math::
          \frac{2 \rho  \sigma_x \sigma_y}{\sigma_x^2 + \sigma_y^2 + (\mu_x - \mu_y)^2}

    and ranges between -1 (denoting perfect discordance) and +1 (denoting perfect concordance).
    """

    def __init__(
        self,
        *,
        reduction: ReductionType = ReductionType.mean,
    ) -> None:
        super().__init__()
        self.reduction = reduction

    @override
    def forward(self, input: Tensor, *, target: Tensor) -> Tensor:
        mean_x = input.mean(keepdim=True, dim=1)
        resid_x = input - mean_x
        var_x = resid_x.pow(2).mean(dim=1)
        mean_y = target.mean(keepdim=True, dim=1)
        resid_y = target - mean_y
        var_y = resid_y.pow(2).mean(dim=1)
        covar = (resid_x * resid_y).mean(dim=1)
        denom = var_x + var_y + (mean_x - mean_y).pow(2)
        ccc = 2 * covar / denom.clamp_min(torch_eps(input))
        losses = 1 - ccc
        return reduce(losses=losses, reduction_type=self.reduction)
