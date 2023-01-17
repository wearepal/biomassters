from pathlib import Path
import tarfile
from typing import Optional, TypeVar, Union, overload

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
from torch.types import Number
from typing_extensions import TypeGuard

__all__ = [
    "default_if_none",
    "some",
    "to_item",
    "to_numpy",
    "to_targz",
    "torch_eps",
    "unwrap_or",
]


T = TypeVar("T")


def some(value: Optional[T]) -> TypeGuard[T]:
    return value is not None


def unwrap_or(value: Optional[T], *, default: T) -> T:
    return default if value is None else value


default_if_none = unwrap_or

DT = TypeVar("DT", bound=Union[np.number, np.bool_])


@overload
def to_numpy(tensor: Tensor, *, dtype: DT) -> npt.NDArray[DT]:
    ...


@overload
def to_numpy(tensor: Tensor, *, dtype: None = ...) -> npt.NDArray:
    ...


def to_numpy(tensor: Tensor, *, dtype: Optional[DT] = None) -> Union[npt.NDArray[DT], npt.NDArray]:
    arr = tensor.detach().cpu().numpy()
    if some(dtype):
        arr.astype(dtype)
    return arr


def to_item(tensor: Tensor) -> Number:
    return tensor.detach().cpu().item()


def torch_eps(data: Tensor) -> float:
    return torch.finfo(data.dtype).eps


def to_targz(source: Path, *, output: Optional[Path] = None) -> Path:
    output = source if output is None else output
    if output.suffixes[-2:] != [".tar", ".gz"]:
        output = output.with_suffix(".tar.gz")

    if not output.exists():
        output.parent.mkdir(exist_ok=True, parents=True)
        with tarfile.open(output, "w:gz") as tar:
            for file in source.iterdir():
                tar.add(file, arcname=file.name)
    return output
