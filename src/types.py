from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Type,
    TypeVar,
    get_type_hints,
)

from torch import Tensor
from typing_extensions import TypeAlias, TypedDict, TypeGuard

__all__ = [
    "ImageSample",
    "LitFalse",
    "LitTrue",
    "TestSample",
    "TrainSample",
    "is_image_sample",
    "is_td_instance",
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


TD = TypeVar("TD", bound=TypedDict)


def is_td_instance(dict_: Dict[str, Any], cls_: Type[TD], *, strict: bool = False) -> TypeGuard[TD]:
    hints = get_type_hints(cls_)
    if strict and (len(dict_) != len(hints)):
        return False
    for key, type_ in hints.items():
        if (key not in dict_) or (not isinstance(dict_[key], type_)):
            return False
    return True


is_image_sample = partial(is_td_instance, cls_=ImageSample)
is_test_sample = partial(is_td_instance, cls_=TestSample)
is_train_sample = partial(is_td_instance, cls_=TrainSample)
LossClosure: TypeAlias = Callable[..., Tensor]
