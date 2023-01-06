from dataclasses import dataclass
from typing import (
    ClassVar,
    overload,
    List,
    TypeVar,
    Optional,
    Generic,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

import torch
from torch import Tensor
import torch.nn as nn
from typing_extensions import override

from src.types import  SampleU, SampleL

__all__ = [
    "AGBMLog1PScale",
    "AppendRatioAB",
    "ClampAGBM",
    "Compose",
    "DropBands",
    "InputTransform",
    "Sentinel1Scale",
    "Sentinel2Scale",
    "TensorTransform",
]


@dataclass(unsafe_hash=True)
@runtime_checkable
class TensorTransform(Protocol):
    def __call__(self, x: Tensor) -> Tensor:
        ...


@dataclass(unsafe_hash=True)
@runtime_checkable
class InputTransform(Protocol):
    def __call__(self, inputs: SampleU) -> SampleU:
        ...

@dataclass(unsafe_hash=True)
@runtime_checkable
class TargetTransform(Protocol):
    def __call__(self, inputs: SampleL) -> SampleL:
        ...

T = TypeVar("T", Union[InputTransform, TargetTransform],  InputTransform, TargetTransform)
S = TypeVar("S", bound=Union[SampleU, SampleL])


@dataclass(unsafe_hash=True)
class Compose(InputTransform, Generic[T]):
    transforms: Sequence[T]

    @overload
    def __call__(self: "Compose[InputTransform]", inputs: SampleU) -> SampleU:
        ...
    @overload
    def __call__(self: "Compose[TargetTransform]", inputs: SampleL) -> SampleL:
        ...
    # @overload
    # def __call__(self: "Compose[TensorTransform]", inputs: SampleU) -> SampleU:
    #     ...
    # @overload
    # def __call__(self: "Compose[Union[InputTransform, TensorTransform]]", inputs: SampleU) -> SampleU:
    #     ...
    @overload
    def __call__(self: "Compose[Union[InputTransform, TargetTransform]]", inputs: SampleL) -> SampleL:
        ...
    # @overload
    # def __call__(self: "Compose[Union[TargetTransform, TensorTransform]]", inputs: SampleL) -> SampleL:
    #     ...
    # @overload
    # def __call__(self: "Compose[Union[InputTransform, TargetTransform, TensorTransform]]", inputs: SampleL) -> SampleL:
        # ...

    @override
    def __call__(self, inputs):
        for transform in self.transforms:
            if isinstance(transform, TensorTransform):
                inputs["image"] = transform(inputs["image"])
            else:
                inputs = transform(inputs) # type: ignore
        return inputs


class AGBMLog1PScale(TargetTransform):
    """Apply ln(x + 1) Scale to AGBM Target Data"""

    @override
    def __call__(self, inputs: SampleL) -> SampleL:
        inputs["label"] = torch.log1p(inputs["label"])
        return inputs


@dataclass(unsafe_hash=True)
class ClampAGBM(TargetTransform):
    """Clamp AGBM Target Data to [vmin, vmax]"""

    vmin: float = 0.0
    "minimum clamp value"
    vmax: float = 500.0
    "maximum clamp value, 500 is reasonable default per empirical analysis of AGBM data"

    @override
    def __call__(self, inputs: SampleL) -> SampleL:
        inputs["label"] = torch.clamp(inputs["label"], min=self.vmin, max=self.vmax)
        return inputs


@dataclass(unsafe_hash=True)
class DropBands(InputTransform):
    """Drop specified bands by index"""

    bands_to_keep = Optional[List[int]]

    @override
    def __call__(self, inputs: SampleU) -> SampleU:
        if self.bands_to_keep is None:
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


@dataclass(unsafe_hash=True)
class AppendRatioAB(InputTransform):
    """Append the ratio of specified bands to the tensor."""

    EPSILON: ClassVar[float] = 1e-10
    DIM: ClassVar[int] = -3
    index_a: int
    "numerator band channel index"
    index_b: int
    "denominator band channel index"

    def _compute_ratio(self, band_a: Tensor, *, band_b: Tensor) -> Tensor:
        """Compute ratio band_a/band_b.
        :param band_a: numerator band tensor
        :param band_b: denominator band tensor
        :returns: band_a/band_b
        """
        return band_a / band_b.clamp_min(self.EPSILON)

    @override
    def __call__(self, sample: SampleU) -> SampleU:
        """Compute and append ratio to input tensor.
        :param sample: dict with tensor stored in sample['image']
        :returns: the transformed sample
        """
        X = sample["image"].detach()
        ratio = self._compute_ratio(
            band_a=X[..., self.index_a, :, :],
            band_b=X[..., self.index_b, :, :],
        )
        ratio = ratio.unsqueeze(self.DIM)
        sample["image"] = torch.cat([X, ratio], dim=self.DIM)
        return sample


@dataclass(unsafe_hash=True)
class Sentinel2Scale(TensorTransform):
    """Scale Sentinel 2 optical channels"""

    @override
    def __call__(self, x: Tensor) -> Tensor:
        scale_val = 4000.0  # True scaling is [0, 10000], most info is in [0, 4000] range
        x = x / scale_val

        # CLP values in band 10 are scaled differently than optical bands, [0, 100]
        if x.ndim == 4:
            x[:][10] = x[:][10] * scale_val / 100.0
        else:
            x[10] = x[10] * scale_val / 100.0
        return x.clamp(0, 1.0)


@dataclass(unsafe_hash=True)
class Sentinel1Scale(nn.Module):
    """Scale Sentinel 1 SAR channels"""

    @override
    def __call__(self, x: Tensor) -> Tensor:
        s1_max = 20.0  # S1 db values range mostly from -50 to +20 per empirical analysis
        s1_min = -50.0
        image = (x - s1_min) / (s1_max - s1_min)
        return image.clamp(0, 1)
