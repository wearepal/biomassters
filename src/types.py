from __future__ import annotations
from typing import  Literal, Union

from torch import Tensor
from typing_extensions import TypeAlias, TypedDict

__all__ = [
    "LitFalse",
    "LitTrue",
    "SampleL",
    "SampleU",
]

LitTrue: TypeAlias = Literal[True]
LitFalse: TypeAlias = Literal[False]

class SampleU(TypedDict):
    image: Tensor

class SampleL(SampleU):
    label: Tensor
