from __future__ import annotations
from typing import Generic, List, Literal, TypeVar, Union

from torch import Tensor
from typing_extensions import TypeAlias, TypedDict

__all__ = [
    "LitFalse",
    "LitTrue",
    "TrainSample",
    "TestSample",
    "ImageSample",
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
