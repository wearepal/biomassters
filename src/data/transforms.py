import random
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
    cast,
    final,
    overload,
    runtime_checkable,
)

import kornia.augmentation as K
from kornia.constants import Resample
from ranzen import gcopy
import torch
from torch import Tensor
import torch.nn as nn
from typing_extensions import override

from src.types import ImageSample, TrainSample
from src.utils import eps

__all__ = [
    "AGBMLog1PScale",
    "AppendRatioAB",
    "ApplyToOpticalSlice",
    "ApplyToTimeSlice",
    "ClampInput",
    "ClampTarget",
    "ColorJiggle",
    "Compose",
    "DenormalizeModule",
    "DropBands",
    "Flatten",
    "Identity",
    "InputTransform",
    "MinMaxNormalizeTarget",
    "MoveDim",
    "NanToNum",
    "OneOf",
    "Permute",
    "RandomHorizontalFlip",
    "RandomResizedCrop",
    "RandomRotation",
    "RandomVerticalFlip",
    "Sentinel1Scaler",
    "Sentinel2Scaler",
    "TensorTransform",
    "Transpose",
    "ZScoreNormalizeTarget",
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


def should_apply(p: float) -> bool:
    if p >= 1.0:
        return True
    elif p <= 0.0:
        return False
    return random.random() < p


class OneOf(Generic[T]):
    def __init__(self, transforms: Sequence[T], *, p: float = 1.0) -> None:
        self.transforms = transforms
        self.p = p

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
        if should_apply(self.p):
            transform = random.choice(self.transforms)
            inputs = transform(inputs)
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


class ClampTarget(TargetTransform):
    """Clamp AGBM Target Data to [vmin, vmax]"""

    def __init__(self, *, min: float = 0.0, max: float = 500.0) -> None:
        self.min = min
        self.max = max

    @override
    def __call__(self, inputs: TrainSample) -> TrainSample:
        inputs["label"] = torch.clamp(inputs["label"], min=self.min, max=self.max)
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
    def __call__(self, inputs: S) -> S:
        if not self.inplace:
            inputs["image"] = inputs["image"].clone()
        inputs["image"] = scale_sentinel1_data_(
            inputs["image"], start_index=self.start_index, end_index=self.end_index
        )
        return inputs


# True scaling is [0, 10000], most info is in [0, 4000] range
S2_PSEUDO_MAX: Final[float] = 4000.0


def scale_sentinel2_data_(
    x: Tensor, *, start_index: int = 0, end_index: Optional[int] = 11
) -> Tensor:
    x[start_index:end_index] /= S2_PSEUDO_MAX
    # CLP values in band 10 are scaled differently than optical bands, [0, 100]
    x[start_index + 10] *= S2_PSEUDO_MAX / 100.0
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


class Normalize(nn.Module):
    def __init__(self, inplace: bool = False, batched: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.batched = batched

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

    def broadcast_shape(self, data: Tensor) -> Tuple[int, ...]:
        if self.batched:
            return (1, -1, *((1,) * (data.ndim - 2)))
        return (-1, *((1,) * (data.ndim - 1)))


class _ZScoreNormalize(Normalize):
    # Buffers
    mean: Tensor
    std: Tensor

    def __init__(
        self,
        *,
        mean: Union[float, List[float]],
        std: Union[float, List[float]],
        inplace: bool = False,
        batched: bool = False,
    ) -> None:
        super().__init__(inplace=inplace, batched=batched)

        self.register_buffer("mean", torch.as_tensor(mean))
        self.register_buffer("std", torch.as_tensor(std))

    @override
    def _inverse_transform(self, data: Tensor) -> Tensor:
        broadcast_shape = self.broadcast_shape(data)
        data *= self.std.view(broadcast_shape)
        data += self.mean.view(broadcast_shape)
        return data

    @override
    def _transform(self, data: Tensor) -> Tensor:
        broadcast_shape = self.broadcast_shape(data)
        data -= self.mean.view(broadcast_shape)
        data /= self.std.clamp_min(eps(data)).view(broadcast_shape)
        return data


class ZScoreNormalizeInput(_ZScoreNormalize, InputTransform):
    @override
    def forward(self, inputs: S) -> S:
        inputs["image"] = self.transform(inputs["image"])
        return inputs


class ZScoreNormalizeTarget(_ZScoreNormalize, TargetTransform):
    @override
    def forward(self, inputs: TrainSample) -> TrainSample:
        inputs["label"] = self.transform(inputs["label"])
        return inputs


class _MinMaxNormalize(Normalize):
    # Buffers
    orig_min: Tensor
    orig_max: Tensor
    new_min: Tensor
    new_max: Tensor

    def __init__(
        self,
        *,
        orig_min: Union[float, List[float]],
        orig_max: Union[float, List[float]],
        new_min: Union[float, List[float]] = 0.0,
        new_max: Union[float, List[float]] = 1.0,
        inplace: bool = False,
        batched: bool = False,
    ) -> None:
        super().__init__(inplace=inplace, batched=batched)

        self.register_buffer("orig_min", torch.as_tensor(orig_min))
        self.register_buffer("orig_max", torch.as_tensor(orig_max))
        self.register_buffer("new_min", torch.as_tensor(new_min))
        self.register_buffer("new_max", torch.as_tensor(new_max))

        if (self.orig_min > self.orig_max).any():
            raise ValueError("'orig_min' cannot be greater than 'orig_max'.")
        if (self.new_min > self.new_max).any():
            raise ValueError("'new_min' cannot be greater than 'new_max'.")
        self.new_range = self.new_max - self.new_min
        self.orig_range = self.orig_max - self.orig_min

    @override
    def _inverse_transform(self, data: Tensor) -> Tensor:
        broadcast_shape = self.broadcast_shape(data)
        data -= self.new_min.view(broadcast_shape)
        data /= (self.new_range + eps(data)).view(broadcast_shape)
        data *= self.orig_range.view(broadcast_shape)
        data += self.orig_min.view(broadcast_shape)
        return data

    @override
    def _transform(self, data: Tensor) -> Tensor:
        broadcast_shape = self.broadcast_shape(data)
        data -= self.orig_min.view(broadcast_shape)
        data /= self.orig_range.view(broadcast_shape)
        data *= self.new_range.view(broadcast_shape)
        data += self.new_min.view(broadcast_shape)
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
        self.normalizers = nn.ModuleList(
            [gcopy(normalizer, batched=True) for normalizer in normalizers]
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            for normalizer in self.normalizers:
                normalizer = cast(Normalize, normalizer)
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


class ClampInput(InputTransform):
    def __init__(
        self, min: Optional[float] = None, max: Optional[float] = None, inplace: bool = True
    ) -> None:
        self.min = min
        self.max = max
        self.inplace = inplace

    @override
    def __call__(self, inputs: S) -> S:
        func = torch.clamp_ if self.inplace else torch.clamp
        inputs["image"] = func(inputs["image"], max=self.max, min=self.min)
        return inputs


class RandomHorizontalFlip(TargetTransform):
    DIMS: ClassVar[Tuple[int]] = (-1,)

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    @override
    def __call__(self, inputs: TrainSample) -> TrainSample:
        if should_apply(self.p):
            inputs["image"] = torch.flip(inputs["image"], dims=self.DIMS)
            inputs["label"] = torch.flip(inputs["label"], dims=self.DIMS)
        return inputs


class RandomVerticalFlip(TargetTransform):
    DIMS: ClassVar[Tuple[int]] = (-2,)

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    @override
    def __call__(self, inputs: TrainSample) -> TrainSample:
        if should_apply(self.p):
            inputs["image"] = torch.flip(inputs["image"], dims=self.DIMS)
            inputs["label"] = torch.flip(inputs["label"], dims=self.DIMS)
        return inputs


class RandomRotation(TargetTransform):
    DIMS: ClassVar[Tuple[int, int]] = (-2, -1)

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    @override
    def __call__(self, inputs: TrainSample) -> TrainSample:
        if random.random() < self.p:
            k = random.randint(1, 3)
            inputs["image"] = torch.rot90(inputs["image"], k=k, dims=self.DIMS)
            inputs["label"] = torch.rot90(inputs["label"], k=k, dims=self.DIMS)
        return inputs


def _apply_along_time_axis(x: Tensor, *, fn: K.AugmentationBase2D) -> Tensor:
    return fn(x.transpose(0, 1)).transpose(0, 1)


class ColorJiggle(InputTransform):
    def __init__(
        self,
        *,
        brightness: Union[float, Tuple[float, float], List[float]] = 0.0,
        contrast: Union[float, Tuple[float, float], List[float]] = 0.0,
        saturation: Union[float, Tuple[float, float], List[float]] = 0.0,
        hue: Union[float, Tuple[float, float], List[float]] = 0.0,
        same_on_batch: bool = False,
        rgb_dims: Tuple[int, int, int] = (0, 1, 2),
        p: float = 1.0,
        inplace: bool = True,
    ) -> None:
        self.p = p
        self.rgb_dims = rgb_dims
        self.inplace = inplace
        self.fn = K.ColorJiggle(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            same_on_batch=same_on_batch,
        )

    @override
    def __call__(self, inputs: S) -> S:
        # inputs are assumed to be in CFHW format (no batch dim)
        if not self.inplace:
            inputs["image"] = inputs["image"].clone()
        # Kornia expects the input to be of shape (C, H, W) or (B, C, H, W) --
        # we treat the frame (F) dim as the batch dim, tranposing it to the 0th
        # dimension and then reversing the tranposition after the transform has
        # been applied.
        transformed = _apply_along_time_axis(x=inputs["image"][self.rgb_dims], fn=self.fn)
        inputs["image"][self.rgb_dims] = transformed
        return inputs


class RandomResizedCrop(TargetTransform):
    def __init__(
        self,
        *,
        size: Tuple[int, int],
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        resample: Resample = Resample.BICUBIC,
        align_corners: bool = True,
        p: float = 1.0,
        cropping_mode: str = "slice",
    ) -> None:
        self.p = p
        self.fn = K.RandomResizedCrop(
            size=size,
            scale=scale,
            ratio=ratio,
            resample=resample,
            same_on_batch=True,
            align_corners=align_corners,
            p=1.0,
            cropping_mode=cropping_mode,
            return_transform=False,
        )

    @override
    def __call__(self, inputs: TrainSample) -> TrainSample:
        if should_apply(self.p):
            # Kornia expects the input to be of shape (C, H, W) or (B, C, H, W)
            # -- we treat the frame (F) dim as the batch dim, tranposing it to
            # the 0th (batch) dimension and then reversing the tranposition
            # after the transform has been applied.
            inputs["image"] = _apply_along_time_axis(x=inputs["image"], fn=self.fn)
            transform_matrix = self.fn.transform_matrix[0]
            inputs["label"] = self.fn(inputs["label"], transform_matrix=transform_matrix)
        return inputs
