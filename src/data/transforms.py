from typing import Final, Protocol

import torch
from torch import Tensor
import torch.nn as nn
from typing_extensions import override

from src.types import Sample

__all__ = [
    "AGBMLog1PScale",
    "ClampAGBM",
    "DropBands",
    "Sentinel2Scale",
    "Sentinel1Scale",
    "AppendRatioAB",
]

_EPSILON: Final[float] = 1e-10


class Transform(Protocol):
    def __call__(self, inputs: Sample) -> Sample:
        ...


class AGBMLog1PScale(Transform):
    """Apply ln(x + 1) Scale to AGBM Target Data"""

    @override
    def __call__(self, inputs: Sample) -> Sample:
        inputs["label"] = torch.log1p(inputs["label"])
        return inputs


class ClampAGBM(Transform):
    """Clamp AGBM Target Data to [vmin, vmax]"""

    def __init__(self, vmin: float = 0.0, vmax: float = 500.0) -> None:
        """Initialize ClampAGBM
        :param vmin: minimum clamp value
        :param vmax: maximum clamp value, 500 is reasonable default per empirical analysis of AGBM data
        """
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax

    @override
    def __call__(self, inputs: Sample) -> Sample:
        inputs["label"] = torch.clamp(inputs["label"], min=self.vmin, max=self.vmax)
        return inputs


class DropBands(Transform):
    """Drop specified bands by index"""

    def __init__(self, bands_to_keep=None) -> None:
        self.bands_to_keep = bands_to_keep

    def __call__(self, inputs: Sample) -> Sample:
        if not self.bands_to_keep:
            return inputs
        X = inputs["image"].detach()
        if X.ndim == 4:
            slice_dim = 1
        else:
            slice_dim = 0
        inputs["image"] = X.index_select(
            slice_dim,
            torch.tensor(
                self.bands_to_keep,
                device=inputs["image"].device,
            ),
        )
        return inputs


class Sentinel2Scale(nn.Module):
    """Scale Sentinel 2 optical channels"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, image: Tensor) -> Tensor:
        scale_val = 4000.0  # True scaling is [0, 10000], most info is in [0, 4000] range
        image = image / scale_val

        # CLP values in band 10 are scaled differently than optical bands, [0, 100]
        if image.ndim == 4:
            image[:][10] = image[:][10] * scale_val / 100.0
        else:
            image[10] = image[10] * scale_val / 100.0
        return image.clamp(0, 1.0)


class Sentinel1Scale(nn.Module):
    """Scale Sentinel 1 SAR channels"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, image) -> Tensor:
        s1_max = 20.0  # S1 db values range mostly from -50 to +20 per empirical analysis
        s1_min = -50.0
        image = (image - s1_min) / (s1_max - s1_min)
        return image.clamp(0, 1)


class AppendRatioAB(Transform):
    """Append the ratio of specified bands to the tensor."""

    def __init__(self, index_a: int, *, index_b: int) -> None:
        """Initialize a new transform instance.
        :param index_a: numerator band channel index
        :param index_b: denominator band channel index
        """
        super().__init__()
        self.dim = -3
        self.index_a = index_a
        self.index_b = index_b

    def _compute_ratio(self, band_a: Tensor, *, band_b: Tensor) -> Tensor:
        """Compute ratio band_a/band_b.
        :param band_a: numerator band tensor
        :param band_b: denominator band tensor
        :returns: band_a/band_b
        """
        return band_a / (band_b + _EPSILON)

    @override
    def __call__(self, sample: Sample) -> Sample:
        """Compute and append ratio to input tensor.
        :param sample: dict with tensor stored in sample['image']
        :returns: the transformed sample
        """
        X = sample["image"].detach()
        ratio = self._compute_ratio(
            band_a=X[..., self.index_a, :, :],
            band_b=X[..., self.index_b, :, :],
        )
        ratio = ratio.unsqueeze(self.dim)
        sample["image"] = torch.cat([X, ratio], dim=self.dim)
        return sample
