from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import attr
from conduit.types import Stage
import pytorch_lightning as pl
from ranzen.torch import SequentialBatchSampler, TrainingMode
from torch.utils.data import DataLoader
from torchgeo.transforms import indices
from tqdm import tqdm
from typing_extensions import TypeAlias, override

from src.data.dataset import SentinelDataset
from src.data.stats import ChannelStatistics, CStatsPair
import src.data.transforms as T
from src.types import LitFalse, LitTrue, TrainSample

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


@attr.define(kw_only=True)
class SentinelDataModule(pl.LightningDataModule):

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

    root: Union[Path, str] = "/srv/galene0/shared/data/biomassters/"
    tile_dir: Optional[Path] = None
    group_by: SentinelDataset.GroupBy = SentinelDataset.GroupBy.CHIP
    preprocess: bool = True
    n_pp_jobs: int = 4
    temporal_dim: Optional[Literal[1]] = 1

    split_seed: int = 47
    val_prop: float = 0.2
    test_prop: Optional[float] = None

    _train_data: Optional[TrainData] = attr.field(default=None, init=False)
    _val_data: Optional[EvalData] = attr.field(default=None, init=False)
    _test_data: Optional[EvalData] = attr.field(default=None, init=False)
    _pred_data: Optional[PredData] = attr.field(default=None, init=False)
    _train_transforms: Optional[TrainTransform] = None
    _eval_transforms: Optional[EvalTransform] = None

    def __attrs_pre_init__(self) -> None:
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
        return self._eval_dataloader(ds=self.val_data)

    @override
    def test_dataloader(self) -> DataLoader:
        return self._eval_dataloader(ds=self.test_data)

    @override
    def predict_dataloader(self) -> DataLoader:
        return self._eval_dataloader(self.pred_data)

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

    def train_statistics(self, compute_var: bool = True) -> CStatsPair:
        dl = self.train_dataloader(eval=True)
        input_stats = ChannelStatistics()
        target_stats = ChannelStatistics()
        for batch in tqdm(dl, desc="Computing channel-wise statistics"):
            batch = cast(TrainSample, batch)
            input_stats.update(batch["image"])
            target_stats.update(batch["label"])
        if compute_var:
            for batch in tqdm(dl, desc="Computing channel-wise variance"):
                batch = cast(TrainSample, batch)
                input_stats.update_var(batch["image"])
                target_stats.update_var(batch["label"])
        return CStatsPair(input=input_stats, target=target_stats)

    @property
    def is_spatiotemporal(self) -> bool:
        return self.group_by is SentinelDataset.GroupBy.CHIP

    @property
    def _default_train_transforms(self) -> TrainTransform:
        # NOTE: torchgeo's indices are appended to dim -3, meaning we have to
        # transpose the temporal and spatial dims prior to (and after) applying
        # them.
        # transpose = T.Transpose(0, 1) if self.is_spatiotemporal else T.Identity()
        return T.Compose(
            [
                # # -- Input transforms --
                # transpose,
                # indices.AppendNDVI(index_nir=6, index_red=2),  # NDVI, index 15
                # indices.AppendNormalizedDifferenceIndex(
                #     index_a=11, index_b=12
                # ),  # (VV-VH)/(VV+VH), index 16
                # indices.AppendNDBI(
                #     index_swir=8, index_nir=6
                # ),  # Difference Built-up Index for development detection, index 17
                # indices.AppendNDRE(
                #     index_nir=6, index_vre1=3
                # ),  # Red Edge Vegetation Index for canopy detection, index 18
                # indices.AppendNDSI(index_green=1, index_swir=8),  # Snow Index, index 19
                # indices.AppendNDWI(
                #     index_green=1, index_nir=6
                # ),  # Difference Water Index for water detection, index 20
                # indices.AppendSWI(
                #     index_vre1=3, index_swir2=8
                # ),  # Standardized Water-Level Index for water detection, index 21
                # T.AppendRatioAB(index_a=11, index_b=12),  # VV/VH Ascending, index 22
                # T.AppendRatioAB(index_a=13, index_b=14),  # VV/VH Descending, index 23
                # transpose,
                # T.DropBands(self.BANDS_TO_KEEP, slice_dim=0),  # DROPS ALL BUT SPECIFIED bands_to_keep
                T.Flatten(start_dim=0, end_dim=1),
                # -- Target transforms --
                T.ClampAGBM(
                    vmin=0.0, vmax=500.0
                ),  # exclude AGBM outliers, 500 is good upper limit per AGBM histograms
                T.MinMaxNormalizeTarget(
                    orig_min=0.0, orig_max=500, new_min=0.0, new_max=1.0, inplace=True
                ),
            ]
        )

    @property
    def _default_eval_transforms(self) -> EvalTransform:
        # NOTE: torchgeo's indices are appended to dim -3, meaning we have to
        # transpose the temporal and spatial dims prior to (and after) applying
        # them.
        # transpose = T.Transpose(0, 1) if self.is_spatiotemporal else T.Identity()
        # Same input transforms as defined in _default_train_transforms
        return T.Compose(
            [
                # transpose,
                # indices.AppendNDVI(index_nir=6, index_red=2),  # NDVI, index 15
                # indices.AppendNormalizedDifferenceIndex(
                #     index_a=11, index_b=12
                # ),  # (VV-VH)/(VV+VH), index 16
                # indices.AppendNDBI(
                #     index_swir=8, index_nir=6
                # ),  # Difference Built-up Index for development detection, index 17
                # indices.AppendNDRE(
                #     index_nir=6, index_vre1=3
                # ),  # Red Edge Vegetation Index for canopy detection, index 18
                # indices.AppendNDSI(index_green=1, index_swir=8),  # Snow Index, index 19
                # indices.AppendNDWI(
                #     index_green=1, index_nir=6
                # ),  # Difference Water Index for water detection, index 20
                # indices.AppendSWI(
                #     index_vre1=3, index_swir2=8
                # ),  # Standardized Water-Level Index for water detection, index 21
                # T.AppendRatioAB(index_a=11, index_b=12),  # VV/VH Ascending, index 22
                # T.AppendRatioAB(index_a=13, index_b=14),  # VV/VH Descending, index 23
                # transpose,
                # T.DropBands(self.BANDS_TO_KEEP, slice_dim=0),  # DROPS ALL BUT SPECIFIED bands_to_keep
                T.Flatten(start_dim=0, end_dim=1),
            ]
        )

    @property
    def target_normalizers(self) -> List[T.Normalize]:
        def _collect(tform: TrainTransform, *, collected: List[T.Normalize]) -> List[T.Normalize]:
            # criteria for a target normalizer
            if isinstance(tform, T.Normalize) and isinstance(tform, T.TargetTransform):
                collected.append(tform)
            # recursively traverse Composed transforms
            elif isinstance(tform, T.Compose):
                for sub_tform in tform.transforms:
                    _collect(sub_tform, collected=collected)
            return collected

        return _collect(self.train_transforms, collected=[])
