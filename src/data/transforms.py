from dataclasses import dataclass, field
from typing import (
    ClassVar,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    final,
    overload,
    runtime_checkable,
)

import torch
from torch import Tensor
import torch.nn as nn
from typing_extensions import override

from src.types import ImageSample, TrainSample

__all__ = [
    "AGBMLog1PScale",
    "AppendRatioAB",
    "ClampAGBM",
    "Compose",
    "DenormalizeModule",
    "DropBands",
    "Identity",
    "InputTransform",
    "MinMaxNormalizeTarget",
    "MoveDim",
    "Permute",
    "Sentinel1Scale",
    "Sentinel2Scale",
    "TensorTransform",
    "Transpose",
    "ZScoreNormalizeTarget",
    "scale_sentinel1_data",
    "scale_sentinel2_data",
]


@runtime_checkable
class TensorTransform(Protocol):
    def __call__(self, x: Tensor) -> Tensor:
        ...


S = TypeVar("S", bound=ImageSample)


@runtime_checkable
class InputTransform(Protocol):
    def __call__(self, inputs: S) -> S:
        ...


@runtime_checkable
class TargetTransform(Protocol):
    def __call__(self, inputs: TrainSample) -> TrainSample:
        ...


T = TypeVar("T", Union[InputTransform, TargetTransform], InputTransform, TargetTransform)


class Compose(Generic[T]):
    def __init__(self, *transforms: T) -> None:
        self.transforms = list(transforms)

    @overload
    def __call__(self: "Compose[InputTransform]", inputs: S) -> S:
        ...

    @overload
    def __call__(self: "Compose[TargetTransform]", inputs: TrainSample) -> TrainSample:
        ...

    @overload
    def __call__(self: "Compose[T]", inputs: Union[S, TrainSample]) -> Union[S, TrainSample]:
        ...

    def __call__(self, inputs):
        for transform in self.transforms:
            inputs = transform(inputs)  # type: ignore
        return inputs

    def __getitem__(self, index: int) -> T:
        return self.transforms[index]


@dataclass(unsafe_hash=True)
class DropBands(InputTransform):
    """Drop specified bands by index"""

    bands_to_keep: Optional[Union[List[int], Tuple[int, ...]]]
    slice_dim: Optional[int] = 0

    @override
    def __call__(self, inputs: S) -> S:
        if self.bands_to_keep is None:
            return inputs

        x = inputs["image"]
        if self.slice_dim is not None:
            slice_dim = self.slice_dim
        elif x.ndim > 3:
            slice_dim = 1
        else:
            slice_dim = 0
        inputs["image"] = x.index_select(
            slice_dim,
            torch.tensor(
                self.bands_to_keep,
                device=x.device,
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
    def __call__(self, sample: S) -> S:
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


class AGBMLog1PScale(TargetTransform):
    """Apply ln(x + 1) Scale to AGBM Target Data"""

    @override
    def __call__(self, inputs: TrainSample) -> TrainSample:
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
    def __call__(self, inputs: TrainSample) -> TrainSample:
        inputs["label"] = torch.clamp(inputs["label"], min=self.vmin, max=self.vmax)
        return inputs


def scale_sentinel2_data(x: Tensor) -> Tensor:
    scale_val = 4000.0  # True scaling is [0, 10000], most info is in [0, 4000] range
    x = x / scale_val

    # CLP values in band 10 are scaled differently than optical bands, [0, 100]
    if x.ndim == 4:
        x[:][10] = x[:][10] * scale_val / 100.0
    else:
        x[10] = x[10] * scale_val / 100.0
    return x.clamp(0, 1.0)


class Sentinel2Scale(TensorTransform):
    """Scale Sentinel 2 optical channels"""

    @override
    def __call__(self, x: Tensor) -> Tensor:
        return scale_sentinel2_data(x)


def scale_sentinel1_data(x: Tensor) -> Tensor:
    s1_max = 20.0  # S1 db values range mostly from -50 to +20 per empirical analysis
    s1_min = -50.0
    image = (x - s1_min) / (s1_max - s1_min)
    return image.clamp(0, 1)


class Sentinel1Scale(TensorTransform):
    """Scale Sentinel 1 SAR channels"""

    @override
    def __call__(self, x: Tensor) -> Tensor:
        return scale_sentinel1_data(x)


def eps(data: Tensor) -> float:
    return torch.finfo(data.dtype).eps


@runtime_checkable
@dataclass(unsafe_hash=True)
class Normalize(Protocol):
    inplace: bool = True

    def _maybe_clone(self, data: Tensor) -> Tensor:
        if not self.inplace:
            data = data.clone()
        return data

    def _transform(self, data: Tensor) -> Tensor:
        """Can be in-place."""
        ...

    @final
    def transform(self, data: Tensor) -> Tensor:
        data = self._maybe_clone(data)
        return self._transform(data)

    def _inverse_transform(self, data: Tensor) -> Tensor:
        """Can be in-place."""
        ...

    @final
    def inverse_transform(self, data: Tensor) -> Tensor:
        data = self._maybe_clone(data)
        return self._inverse_transform(data)


@dataclass(unsafe_hash=True)
class _ZScoreNormalizeMixin:
    mean: Tensor
    std: Tensor


@dataclass(unsafe_hash=True)
class _ZScoreNormalize(Normalize, _ZScoreNormalizeMixin):
    @override
    def _inverse_transform(self, data: Tensor) -> Tensor:
        data *= self.std
        data += self.mean
        return data

    @override
    def _transform(self, data: Tensor) -> Tensor:
        data -= self.mean
        data /= self.std.clamp_min(data)
        return data


@dataclass(unsafe_hash=True)
class ZScoreNormalizeInput(_ZScoreNormalize, InputTransform):
    @override
    def __call__(self, inputs: S) -> S:
        inputs["image"] = self.transform(inputs["image"])
        return inputs


@dataclass(unsafe_hash=True)
class ZScoreNormalizeTarget(_ZScoreNormalize, TargetTransform):
    @override
    def __call__(self, inputs: TrainSample) -> TrainSample:
        inputs["label"] = self.transform(inputs["label"])
        return inputs


@dataclass(unsafe_hash=True)
class _MinMaxNormalizeMixin:
    orig_max: float
    orig_min: float
    orig_range: float = field(init=False)
    new_min: float = 0.0
    new_max: float = 1.0
    new_range: float = field(init=False)


@dataclass(unsafe_hash=True)
class _MinMaxNormalize(Normalize, _MinMaxNormalizeMixin):
    orig_range: float = field(init=False)
    new_min: float = 0.0
    new_max: float = 1.0
    new_range: float = field(init=False)

    def __post_init__(self) -> None:
        if self.new_min > self.new_max:
            raise ValueError("'new_min' cannot be greater than 'new_max'.")
        self.new_range = self.new_max - self.new_min
        self.orig_range = self.orig_max - self.orig_min

    @override
    def _inverse_transform(self, data: Tensor) -> Tensor:
        data -= self.new_min
        data /= self.new_range + eps(data)
        data *= self.orig_range
        data += self.orig_min
        return data

    @override
    def _transform(self, data: Tensor) -> Tensor:
        data -= self.orig_min
        data /= self.orig_range
        data *= self.new_range
        data += self.new_min
        return data


@dataclass(unsafe_hash=True)
class MinMaxNormalizeInput(_MinMaxNormalize, InputTransform):
    @override
    def __call__(self, inputs: S) -> S:
        inputs["image"] = self.transform(inputs["image"])
        return inputs


@dataclass(unsafe_hash=True)
class MinMaxNormalizeTarget(_MinMaxNormalize, TargetTransform):
    @override
    def __call__(self, inputs: TrainSample) -> TrainSample:
        inputs["label"] = self.transform(inputs["label"])
        return inputs


class DenormalizeModule(nn.Module):
    def __init__(self, *normalizers: Normalize) -> None:
        super().__init__()
        self.normalizers = reversed(list(normalizers))

    @override
    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            for normalizer in self.normalizers:
                x = normalizer.inverse_transform(x)
        return x


@dataclass(unsafe_hash=True)
class Transpose(InputTransform):
    dim0: int
    dim1: int

    @override
    def __call__(self, inputs: S) -> S:
        inputs["image"] = inputs["image"].transpose(dim0=self.dim0, dim1=self.dim1)
        return inputs


@dataclass(unsafe_hash=True)
class MoveDim(InputTransform):
    source: int
    destination: int

    @override
    def __call__(self, inputs: S) -> S:
        inputs["image"] = inputs["image"].movedim(source=self.source, destination=self.destination)
        return inputs


@dataclass(unsafe_hash=True, init=False)
class Permute(InputTransform):
    def __init__(self, *dims: int) -> None:
        self.dims = dims

    @override
    def __call__(self, inputs: S) -> S:
        inputs["image"] = inputs["image"].permute(self.dims)
        return inputs


@dataclass(unsafe_hash=True, init=False)
class Identity(InputTransform):
    def __init__(self, *dims: int) -> None:
        self.dims = dims

    @override
    def __call__(self, inputs: S) -> S:
        return inputs
