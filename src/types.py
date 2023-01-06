from typing import TypedDict

from torch import Tensor

__all__ = ["Sample"]


class Sample(TypedDict):
    image: Tensor
    label: Tensor
