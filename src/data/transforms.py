from typing import (
    ClassVar,
    Final,
    Generic,
    List,
    Optional,
    Protocol,
    Sequence,
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
from src.utils import eps

__all__ = [
    "AGBMLog1PScale",
    "AppendRatioAB",
    "ClampAGBM",
    "Compose",
    "DenormalizeModule",
    "DropBands",
    "Flatten",
    "Identity",
    "InputTransform",
    "MinMaxNormalizeTarget",
    "MoveDim",
    "NanToNum",
    "Permute",
    "Sentinel1Scaler",
    "Sentinel2Scaler",
    "TensorTransform",
    "Transpose",
    "ZScoreNormalizeTarget",
    "ApplyToOpticalSlice",
    "ApplyToTimeSlice",
    "scale_sentinel1_data_",
    "scale_sentinel2_data_",
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
    def __init__(self, transforms: Sequence[T]) -> None:
        self.transforms = transforms

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

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class DropBands(InputTransform):
    """Drop specified bands by index"""

    def __init__(
        self,
        bands_to_keep: Optional[Union[List[int], Tuple[int, ...]]],
        *,
        slice_dim: Optional[int] = 0,
    ) -> None:
        self.bands_to_keep = bands_to_keep
        self.slice_dim = slice_dim

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


class AppendRatioAB(InputTransform):
    """Append the ratio of specified bands to the tensor."""

    EPSILON: ClassVar[float] = 1e-10
    DIM: ClassVar[int] = -3

    def __init__(self, *, index_a: int, index_b: int) -> None:
        self.index_a = index_a
        self.index_b = index_b

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


class ClampAGBM(TargetTransform):
    """Clamp AGBM Target Data to [vmin, vmax]"""

    def __init__(self, *, vmin: float = 0.0, vmax: float = 500.0) -> None:
        self.vmin = vmin
        self.vmax = vmax

    @override
    def __call__(self, inputs: TrainSample) -> TrainSample:
        inputs["label"] = torch.clamp(inputs["label"], min=self.vmin, max=self.vmax)
        return inputs


# S1 db values range mostly from -50 to +20 per empirical analysis
S1_MAX: Final[float] = 20.0
S1_MIN: Final[float] = -50.0
S1_RANGE = S1_MAX - S1_MIN


def scale_sentinel1_data_(
    x: Tensor, start_index: int = 11, end_index: Optional[int] = 15
) -> Tensor:
    x[start_index:end_index] -= S1_MIN
    x[start_index:end_index] /= S1_RANGE
    x[start_index:end_index].clamp_(0, 1)
    return x


class Sentinel1Scaler(InputTransform):
    """Scale Sentinel 1 SAR channels"""

    def __init__(
        self, start_index: int = 11, end_index: Optional[int] = 15, inplace: bool = True
    ) -> None:
        self.start_index = start_index
        self.end_index = end_index
        self.inplace = inplace

    @override
    def __call__(self, sample: S) -> S:
        if not self.inplace:
            sample["image"] = sample["image"].clone()
        sample["image"] = scale_sentinel1_data_(
            sample["image"], start_index=self.start_index, end_index=self.end_index
        )
        return sample


# True scaling is [0, 10000], most info is in [0, 4000] range
S2_PSEUDO_MAX: Final[float] = 4000.0


def scale_sentinel2_data_(
    x: Tensor, *, start_index: int = 0, end_index: Optional[int] = 11
) -> Tensor:
    x[start_index:end_index] /= S2_PSEUDO_MAX
    # CLP values in band 10 are scaled differently than optical bands, [0, 100]
    x[start_index + 10] *= S2_PSEUDO_MAX / 100.0
    x[start_index:end_index].clamp_(0, 1)
    return x


class ApplyToOpticalSlice(InputTransform):
    def __init__(
        self,
        transform: InputTransform,
        *,
        start_index: int,
        end_index: Optional[int] = None,
        inplace: bool = True,
    ) -> None:
        self.transform = transform
        self.start_index = start_index
        self.end_index = end_index
        self.inplace = inplace

    @override
    def __call__(self, inputs: S) -> S:
        if not self.inplace:
            inputs["image"] = inputs["image"].clone()
        sliced_input: ImageSample = {"image": inputs["image"][self.start_index : self.end_index]}
        inputs["image"][self.start_index : self.end_index] = self.transform(sliced_input)["image"]
        return inputs


class ApplyToTimeSlice(InputTransform):
    def __init__(
        self,
        transform: InputTransform,
        *,
        start_index: int,
        end_index: Optional[int] = None,
        inplace: bool = True,
    ) -> None:
        self.transform = transform
        self.start_index = start_index
        self.end_index = end_index
        self.inplace = inplace

    @override
    def __call__(self, inputs: S) -> S:
        if not self.inplace:
            inputs["image"] = inputs["image"].clone()
        sliced_input: ImageSample = {"image": inputs["image"][:, self.start_index : self.end_index]}
        inputs["image"][:, self.start_index : self.end_index] = self.transform(sliced_input)[
            "image"
        ]
        return inputs


class Sentinel2Scaler(InputTransform):
    """Scale Sentinel 2 optical channels"""

    def __init__(
        self, start_index: int = 0, end_index: Optional[int] = 11, inplace: bool = True
    ) -> None:
        self.start_index = start_index
        self.end_index = end_index
        self.inplace = inplace

    @override
    def __call__(self, inputs: S) -> S:
        if not self.inplace:
            inputs["image"] = inputs["image"].clone()
        inputs["image"] = scale_sentinel2_data_(
            inputs["image"], start_index=self.start_index, end_index=self.end_index
        )
        return inputs


class Normalize:
    def __init__(self, inplace: bool = False) -> None:
        self.inplace = inplace

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


class _ZScoreNormalize(Normalize):
    def __init__(self, *, mean: Tensor, std: Tensor, inplace: bool = False) -> None:
        super().__init__(inplace=inplace)
        self.mean = mean
        self.std = std

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


class ZScoreNormalizeInput(_ZScoreNormalize, InputTransform):
    @override
    def __call__(self, inputs: S) -> S:
        inputs["image"] = self.transform(inputs["image"])
        return inputs


class ZScoreNormalizeTarget(_ZScoreNormalize, TargetTransform):
    @override
    def __call__(self, inputs: TrainSample) -> TrainSample:
        inputs["label"] = self.transform(inputs["label"])
        return inputs


class _MinMaxNormalize(Normalize):
    def __init__(
        self,
        *,
        orig_min: float,
        orig_max: float,
        new_min: float = 0.0,
        new_max: float = 1.0,
        inplace: bool = False,
    ) -> None:
        super().__init__(inplace=inplace)
        self.orig_min = orig_min
        self.orig_max = orig_max
        self.new_min = new_min
        self.new_max = new_max
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


class MinMaxNormalizeInput(_MinMaxNormalize, InputTransform):
    @override
    def __call__(self, inputs: S) -> S:
        inputs["image"] = self.transform(inputs["image"])
        return inputs


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


class Transpose(InputTransform):
    def __init__(self, *, dim0: int, dim1: int) -> None:
        self.dim0 = dim0
        self.dim1 = dim1

    @override
    def __call__(self, inputs: S) -> S:
        inputs["image"] = inputs["image"].transpose(dim0=self.dim0, dim1=self.dim1)
        return inputs


class MoveDim(InputTransform):
    def __init__(self, *, source: int, destination: int) -> None:
        self.source = source
        self.destination = destination

    @override
    def __call__(self, inputs: S) -> S:
        inputs["image"] = inputs["image"].movedim(source=self.source, destination=self.destination)
        return inputs


class Permute(InputTransform):
    def __init__(self, dims: Sequence[int]) -> None:
        self.dims = dims

    @override
    def __call__(self, inputs: S) -> S:
        inputs["image"] = inputs["image"].permute(*self.dims)
        return inputs


class Identity(InputTransform):
    @override
    def __call__(self, inputs: S) -> S:
        return inputs


class Flatten(InputTransform):
    def __init__(self, *, start_dim: int, end_dim: int) -> None:
        self.start_dim = start_dim
        self.end_dim = end_dim

    @override
    def __call__(self, inputs: S) -> S:
        inputs["image"] = inputs["image"].flatten(start_dim=self.start_dim, end_dim=self.end_dim)
        return inputs


class NanToNum(InputTransform):
    def __init__(
        self,
        *,
        nan: Optional[float] = None,
        posinf: Optional[float] = None,
        neginf: Optional[float] = None,
        inplace: bool = True,
    ) -> None:
        self.nan = nan
        self.posinf = posinf
        self.neginf = neginf
        self.inplace = inplace

    @override
    def __call__(self, inputs: S) -> S:
        func = torch.nan_to_num_ if self.inplace else torch.nan_to_num
        inputs["image"] = func(
            inputs["image"], nan=self.nan, posinf=self.posinf, neginf=self.neginf
        )
        return inputs
