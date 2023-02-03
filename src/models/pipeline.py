from dataclasses import dataclass
from typing import Generic, Optional, TypeVar, overload

from ranzen.torch.module import DcModule
from torch import Tensor, nn
from typing_extensions import override

from src.data.transforms import DenormalizeModule
from src.models.mask import TrainableImputer

__all__ = ["ModelPipeline"]


I = TypeVar("I", bound=Optional[TrainableImputer])


@dataclass(unsafe_hash=True)
class ModelPipeline(DcModule, Generic[I]):
    imputer: I
    temporal: bool
    model: nn.Module
    denorm: DenormalizeModule

    @overload
    def forward(self: "ModelPipeline[TrainableImputer]", x: Tensor, *, mask: Tensor) -> Tensor:
        ...

    @overload
    def forward(self: "ModelPipeline[None]", x: Tensor, *, mask: Optional[Tensor]) -> Tensor:
        ...

    @overload
    def forward(self: "ModelPipeline[I]", x: Tensor, *, mask: Optional[Tensor]) -> Tensor:
        ...

    @override
    def forward(self, x: Tensor, *, mask: Optional[Tensor]) -> Tensor:
        if self.imputer is not None:
            x = self.imputer(x, mask=mask)
        if not self.temporal:
            x = x.flatten(start_dim=1, end_dim=2)
        x = self.model(x)
        return self.denorm(x)
