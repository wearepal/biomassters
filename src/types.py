from __future__ import annotations
from typing import Generic, Optional, TypeVar

from torch import Tensor
from typing_extensions import TypedDict

__all__ = ["Sample", "L"]


L = TypeVar("L", bound=Optional[Tensor])


class Sample(TypedDict, Generic[L]):
    image: Tensor
    label: L
