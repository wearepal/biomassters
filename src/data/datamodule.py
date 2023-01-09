from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Generic, Optional, Tuple, TypeVar, Union, overload

import attr
from conduit.types import Stage
import pytorch_lightning as pl
from ranzen.torch import SequentialBatchSampler, TrainingMode
from torch.utils.data import DataLoader
from torchgeo.transforms import indices
from typing_extensions import TypeAlias, override

from src.data.dataset import SentinelDataset
import src.data.transforms as T
from src.types import LitFalse, LitTrue

__all__ = [
    "SentinelDataModule",
    "TrainValTestPredSplit",
]

TrainData: TypeAlias = SentinelDataset[LitTrue, Any]
EvalData: TypeAlias = SentinelDataset[LitTrue, Any]
PredData: TypeAlias = SentinelDataset[LitFalse, Any]

TrainTransform: TypeAlias = Union[T.InputTransform, T.TargetTransform]
EvalTransform: TypeAlias = T.InputTransform

TD = TypeVar("TD", bound=Optional[EvalData])


@dataclass(unsafe_hash=True)
class TrainValTestPredSplit(Generic[TD]):
    train: TrainData
    val: EvalData
    test: TD
    pred: PredData


TP = TypeVar("TP", None, float)


@attr.define(kw_only=True, eq=False)
class SentinelDataModule(pl.LightningDataModule, Generic[TP]):

    BANDS_TO_KEEP: ClassVar[Tuple[int, ...]] = (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
    )  # via offline feature selection

    train_batch_size: int = 16
    _eval_batch_size: Optional[int] = None
    num_workers: int = 0
    persist_workers: bool = False
    pin_memory: bool = True

    root: Path
    tile_dir: Optional[Path] = None
    group_by: SentinelDataset.GroupBy = SentinelDataset.GroupBy.CHIP_MONTH
    temporal_dim: int = 1
    preprocess: bool = True
    n_pp_jobs: int = 4

    _train_data: Optional[TrainData] = attr.field(default=None, init=False)
    _val_data: Optional[EvalData] = attr.field(default=None, init=False)
    _test_data: Optional[EvalData] = attr.field(default=None, init=False)
    _pred_data: Optional[PredData] = attr.field(default=None, init=False)

    split_seed: int = 47
    val_prop: float = 0.2
    test_prop: TP = attr.field(default=None)

    def __attrs_post_init__(self) -> None:
        super().__init__()

    @property
    def eval_batch_size(self) -> int:
        if self._eval_batch_size is None:
            return self.train_batch_size
        return self._eval_batch_size

    @property
    def train_data(self) -> TrainData:
        self._check_setup_called()
        assert self._train_data is not None
        return self._train_data

    @property
    def in_channels(self) -> int:
        return self.train_data.in_channels

    @property
    def val_data(self) -> EvalData:
        self._check_setup_called()
        assert self._val_data is not None
        return self._val_data

    @property
    def test_data(
        self,
    ) -> EvalData:
        self._check_setup_called()
        return self.val_data if self._test_data is None else self._test_data

    @property
    def pred_data(
        self,
    ) -> PredData:
        self._check_setup_called()
        assert self._pred_data is not None
        return self._pred_data

    @property
    def is_set_up(self) -> bool:
        return self._train_data is not None

    def _check_setup_called(self, caller: Optional[str] = None) -> None:
        if not self.is_set_up:
            if caller is None:
                # inspect the call stack to find out who called this function
                import inspect

                caller = inspect.getouterframes(inspect.currentframe(), 2)[1][3]

            cls_name = self.__class__.__name__
            raise AttributeError(
                f"'{cls_name}.{caller}' cannot be accessed as '{cls_name}.setup()' has "
                "not yet been called."
            )

    def _eval_dataloader(self, ds: SentinelDataset) -> DataLoader:
        """Factory method shared by all dataloaders."""
        return DataLoader(
            ds,
            batch_size=self.eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persist_workers,
        )

    @override
    def train_dataloader(self, eval: bool = False) -> DataLoader:
        """Factory method for train-data dataloaders."""
        if eval:
            return self._eval_dataloader(ds=self.train_data)
        batch_sampler = SequentialBatchSampler(
            data_source=self.train_data,
            batch_size=self.train_batch_size,
            shuffle=True,
            training_mode=TrainingMode.step,
            drop_last=False,
        )
        return DataLoader(
            self.train_data,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persist_workers,
            batch_sampler=batch_sampler,
            sampler=None,
        )

    @override
    def val_dataloader(self) -> DataLoader:
        # loaders = {f"{str(Stage.VALIDATE)}": self._eval_dataloader(ds=self.val_data)}
        # if with_test and (self._test_data is not None):
        #     loaders |= {f"{str(Stage.TEST)}": self._eval_dataloader(ds=self.test_data)}

        # return CombinedLoader(loaders, mode="max_size_cycle")
        return self._eval_dataloader(ds=self.val_data)

    @override
    def test_dataloader(self) -> DataLoader:
        # if self._test_data is None:
        #     loaders = {f"{str(Stage.VALIDATE)}": self._eval_dataloader(ds=self.val_data)}
        # else:
        #     loaders = {f"{str(Stage.TEST)}": self._eval_dataloader(ds=self.test_data)}
        # return CombinedLoader(loaders, mode="max_size_cycle")
        return self._eval_dataloader(ds=self.test_data)

    @override
    def predict_dataloader(self) -> DataLoader:
        return self._eval_dataloader(self.pred_data)

    @overload
    def _get_splits(self: "SentinelDataModule[float]") -> TrainValTestPredSplit[EvalData]:
        ...

    @overload
    def _get_splits(self: "SentinelDataModule[None]") -> TrainValTestPredSplit[None]:
        ...

    def _get_splits(self) -> TrainValTestPredSplit:
        train = SentinelDataset(
            root=self.root,
            tile_dir=self.tile_dir,
            group_by=self.group_by,
            preprocess=self.preprocess,
            n_pp_jobs=self.n_pp_jobs,
            train=True,
        )
        if self.test_prop is None:
            val, train = train.random_split(self.val_prop, seed=self.split_seed)
            test = None
        else:
            val, test, train = train.random_split(
                (self.val_prop, self.test_prop), seed=self.split_seed
            )
        pred = SentinelDataset(
            root=self.root,
            tile_dir=self.tile_dir,
            group_by=self.group_by,
            preprocess=self.preprocess,
            n_pp_jobs=self.n_pp_jobs,
            train=False,
        )
        return TrainValTestPredSplit(train=train, val=val, test=test, pred=pred)

    @override
    def prepare_data(self) -> None:
        self._get_splits()

    @override
    def setup(self, stage: Optional[Stage] = None, force_reset: bool = False) -> None:
        # Only perform the setup if it hasn't already been done
        if force_reset or (not self.is_set_up):
            splits = self._get_splits()
            self._train_data = splits.train
            self._val_data = splits.val
            self._test_data = splits.test
            self._pred_data = splits.pred

            self.train_transforms = self.train_transforms
            self.eval_transforms = self.eval_transforms

    @property
    def train_transforms(self) -> TrainTransform:
        return (
            self._default_train_transforms
            if self._train_transforms is None
            else self._train_transforms
        )

    @train_transforms.setter
    def train_transforms(self, transform: Optional[TrainTransform]) -> None:
        self._train_transforms = transform
        if self._train_data is not None:
            self._train_data.transform = transform

    @property
    def eval_transforms(self) -> EvalTransform:
        return (
            self._default_eval_transforms
            if self._eval_transforms is None
            else self._eval_transforms
        )

    @eval_transforms.setter
    def eval_transforms(self, transform: EvalTransform) -> None:
        self._eval_transforms = transform
        if self._test_data is not None:
            self._test_data.transform = transform
        if self._val_data is not None:
            self._val_data.transform = transform
        if self._pred_data is not None:
            self._pred_data.transform = transform

    @property
    def _default_train_transforms(self) -> TrainTransform:
        return T.Compose(
            T.ClampAGBM(
                vmin=0.0, vmax=500.0
            ),  # exclude AGBM outliers, 500 is good upper limit per AGBM histograms
            indices.AppendNDVI(index_nir=6, index_red=2),  # NDVI, index 15
            indices.AppendNormalizedDifferenceIndex(
                index_a=11, index_b=12
            ),  # (VV-VH)/(VV+VH), index 16
            indices.AppendNDBI(
                index_swir=8, index_nir=6
            ),  # Difference Built-up Index for development detection, index 17
            indices.AppendNDRE(
                index_nir=6, index_vre1=3
            ),  # Red Edge Vegetation Index for canopy detection, index 18
            indices.AppendNDSI(index_green=1, index_swir=8),  # Snow Index, index 19
            indices.AppendNDWI(
                index_green=1, index_nir=6
            ),  # Difference Water Index for water detection, index 20
            indices.AppendSWI(
                index_vre1=3, index_swir2=8
            ),  # Standardized Water-Level Index for water detection, index 21
            T.AppendRatioAB(index_a=11, index_b=12),  # VV/VH Ascending, index 22
            T.AppendRatioAB(index_a=13, index_b=14),  # VV/VH Descending, index 23
            T.DropBands(self.BANDS_TO_KEEP),  # DROPS ALL BUT SPECIFIED bands_to_keep
        )

    @property
    def _default_eval_transforms(self) -> EvalTransform:
        # Same as _default_train_transforms save for the target transform, ClampAGBM.
        return T.Compose(
            indices.AppendNDVI(index_nir=6, index_red=2),  # NDVI, index 15
            indices.AppendNormalizedDifferenceIndex(
                index_a=11, index_b=12
            ),  # (VV-VH)/(VV+VH), index 16
            indices.AppendNDBI(
                index_swir=8, index_nir=6
            ),  # Difference Built-up Index for development detection, index 17
            indices.AppendNDRE(
                index_nir=6, index_vre1=3
            ),  # Red Edge Vegetation Index for canopy detection, index 18
            indices.AppendNDSI(index_green=1, index_swir=8),  # Snow Index, index 19
            indices.AppendNDWI(
                index_green=1, index_nir=6
            ),  # Difference Water Index for water detection, index 20
            indices.AppendSWI(
                index_vre1=3, index_swir2=8
            ),  # Standardized Water-Level Index for water detection, index 21
            T.AppendRatioAB(index_a=11, index_b=12),  # VV/VH Ascending, index 22
            T.AppendRatioAB(index_a=13, index_b=14),  # VV/VH Descending, index 23
            T.DropBands(self.BANDS_TO_KEEP),  # DROPS ALL BUT SPECIFIED bands_to_keep
        )
