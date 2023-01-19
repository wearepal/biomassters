from __future__ import annotations
from typing import Any, Generic, List, Literal, TypeVar

from torch import Tensor
from typing_extensions import TypeAlias, TypedDict, TypeGuard

__all__ = [
    "ImageSample",
    "LitFalse",
    "LitTrue",
    "TestSample",
    "TrainSample",
    "is_image_sample",
    "is_test_sample",
    "is_train_sample",
]

LitTrue: TypeAlias = Literal[True]
LitFalse: TypeAlias = Literal[False]


class ImageSample(TypedDict):
    image: Tensor


C = TypeVar("C", str, List[str])


class TestSample(ImageSample, Generic[C]):
    image: Tensor
    chip: C


class TrainSample(ImageSample):
    label: Tensor


def is_image_sample(inputs: Any) -> TypeGuard[ImageSample]:
    return isinstance(inputs, dict) and "image" in inputs


def is_test_sample(inputs: Any) -> TypeGuard[TestSample]:
    return isinstance(inputs, dict) and "image" in inputs and "chip" in inputs


def is_train_sample(inputs: Any) -> TypeGuard[TrainSample]:
    return isinstance(inputs, dict) and "image" in inputs and "label" in inputs
