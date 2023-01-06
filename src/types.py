from __future__ import annotations
from typing import Generic, Literal, Optional, TypeVar

from torch import Tensor
from typing_extensions import TypeAlias, TypedDict

__all__ = [
    "L",
    "LitFalse",
    "LitTrue",
    "Sample",
]

LitTrue: TypeAlias = Literal[True]
LitFalse: TypeAlias = Literal[False]

L = TypeVar("L", bound=Optional[Tensor])


class Sample(TypedDict, Generic[L]):
    image: Tensor
    label: L
