from enum import Enum
from functools import lru_cache
from pathlib import Path
import shutil
from typing import (
    Callable,
    ClassVar,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Sequence,
    TypeVar,
    Union,
    cast,
    overload,
)
import warnings

import joblib  # type: ignore
from loguru import logger
import numpy as np
import numpy.typing as npt
import pandas as pd
from ranzen import gcopy
from ranzen.torch.data import prop_random_split
import rasterio  # type: ignore
import rasterio.errors  # type: ignore
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from typing_extensions import Self, TypeAlias

from src.data.transforms import InputTransformP, TargetTransformP
from src.logging import tqdm_joblib
from src.types import LitFalse, LitTrue, TestSample, TrainSample
from src.utils import some

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

__all__ = ["SentinelDataset"]


class _SentinelType(Enum):
    S1 = 4
    S2 = 11

    def __init__(self, n_channels: int) -> None:
        self.n_channels = n_channels


class _GroupBy(Enum):
    CHIP = ["chip"]
    CHIP_MONTH = ["chip", "month"]


TR = TypeVar("TR", bound=Literal[True, False])
P = TypeVar("P", bound=Literal[True, False])


def iszero(x: Tensor) -> Tensor:
    return x == 0.0


class _MissingValue(Enum):
    ZERO = (0.0, iszero)
    NAN = (torch.nan, torch.isnan)
    INF = (torch.inf, torch.isposinf)
    NEG_INF = (-torch.inf, torch.isneginf)

    def __init__(self, float: float, checker: Callable[[Tensor], Tensor]) -> None:
        self.float = float
        self.checker = checker


class _SavePrecision(Enum):
    HALF = (np.float16, torch.float16)
    SINGLE = (np.float32, torch.float32)

    def __init__(self, np_: np.dtype, torch_: torch.dtype) -> None:
        self.np = np_
        self.torch = torch_


class _SaveWith(Enum):
    NP = "np"
    TORCH = "torch"


class SentinelDataset(Dataset, Generic[TR, P]):
    """Sentinel 1 & 2 dataset."""

    SentinelType: TypeAlias = _SentinelType
    GroupBy: TypeAlias = _GroupBy
    MissingValue: TypeAlias = _MissingValue
    SaveWith: TypeAlias = _SaveWith
    SavePrecision: TypeAlias = _SavePrecision

    RESOLUTION: ClassVar[int] = 256
    NUM_FRAMES: ClassVar[int] = 12
    CHANNEL_DIM: ClassVar[int] = 0
    TEMPORAL_DIM: ClassVar[int] = 1

    TRAIN_DIR_NAME: ClassVar[str] = "train_features"
    TARGET_DIR_NAME: ClassVar[str] = "train_agbm"
    TEST_DIR_NAME: ClassVar[str] = "test_features"
    PP_METADATA_FN: ClassVar[str] = "metadata.csv"
    MONTH_MAP: ClassVar[Dict[int, str]] = {
        0: "Sep",
        1: "Oct",
        2: "Nov",
        3: "Dec",
        4: "Jan",
        5: "Feb",
        6: "Mar",
        7: "Apr",
        8: "May",
        9: "Jun",
        10: "Jul",
        11: "Aug",
    }
    CHANNEL_MAP: ClassVar[Dict[int, str]] = {
        # Sentinel2 channels
        0: "S2-B2: Blue-10m",
        1: "S2-B3: Green-10m",
        2: "S2-B4: Red-10m",
        3: "S2-B5: VegRed-704nm-20m",
        4: "S2-B6: VegRed-740nm-20m",
        5: "S2-B7: VegRed-780nm-20m",
        6: "S2-B8: NIR-833nm-10m",
        7: "S2-B8A: NarrowNIR-864nm-20m",
        8: "S2-B11: SWIR-1610nm-20m",
        9: "S2-B12: SWIR-2200nm-20m",
        10: "S2-CLP: CloudProb-160m",
        # Sentinel1 channels
        11: "S1-VV-Asc: Cband-10m",
        12: "S1-VH-Asc: Cband-10m",
        13: "S1-VV-Desc: Cband-10m",
        14: "S1-VH-Desc: Cband-10m",
    }
    INVERSE_CHANNEL_MAP = dict(zip(CHANNEL_MAP.values(), CHANNEL_MAP.keys()))
    # Sentinel1 data spans channels 0-10
    S1_SLICE: slice = slice(0, 11)
    # Sentinel2 data spans channels 11-14
    S2_SLICE: slice = slice(11, 15)

    @overload
    def __init__(
        self,
        root: Union[Path, str],
        *,
        train: LitTrue,
        tile_dir: Optional[Path] = ...,
        group_by: GroupBy = ...,
        preprocess: Literal[True, False] = ...,
        n_pp_jobs: int = ...,
        transform: Optional[Union[InputTransformP, TargetTransformP]] = ...,
        missing_value: MissingValue = ...,
        save_with: SaveWith = ...,
        save_precision: SavePrecision = ...,
        indices_fp: Optional[Path] = ...,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        root: Union[Path, str],
        *,
        train: LitFalse,
        tile_dir: Optional[Path] = ...,
        group_by: GroupBy = ...,
        preprocess: Literal[True, False] = ...,
        n_pp_jobs: int = ...,
        transform: Optional[InputTransformP] = ...,
        missing_value: MissingValue = ...,
        save_with: SaveWith = ...,
        save_precision: SavePrecision = ...,
        indices_fp: Optional[Path] = ...,
    ) -> None:
        ...

    def __init__(
        self,
        root: Union[Path, str],
        *,
        train: TR = True,
        tile_dir: Optional[Path] = None,
        group_by: GroupBy = GroupBy.CHIP_MONTH,
        preprocess: P = True,
        n_pp_jobs: int = 8,
        transform: Optional[Union[InputTransformP, TargetTransformP]] = None,
        missing_value: MissingValue = MissingValue.INF,
        save_with: SaveWith = SaveWith.NP,
        save_precision: SavePrecision = SavePrecision.HALF,
        indices_fp: Optional[Path] = None,
    ) -> None:
        self.root = Path(root)
        self.train = train
        self.group_by = group_by
        self.missing_value = missing_value
        self.n_pp_jobs = n_pp_jobs
        self.input_dir = self.root / (self.TRAIN_DIR_NAME if self.train else self.TEST_DIR_NAME)
        self.target_dir = (self.root / self.TARGET_DIR_NAME) if self.train else None
        self.tile_dir = tile_dir
        self._preprocess = preprocess
        self.save_with = save_with
        self.save_precision = save_precision

        if (not self.preprocess) or (not self.is_preprocessed):
            if self.tile_file is None:
                self.metadata = self._generate_metadata(self.input_dir)
            else:
                self.metadata = pd.read_csv(self.tile_file, index_col=0)
            # Rename 'chipid' for backwards compatibility.
            if "chipid" in self.metadata.columns:
                self.metadata.rename({"chipid": "chip"}, axis=1, inplace=True)
            # Currrently we sample a chip/month and then sample both data from
            # both satellites; this scheme is very much subject to change. For instance,
            # we might wish to sample by chip and treat the monthly data as a spatiotemporal series.
            self.metadata.drop_duplicates(subset=self.group_by.value, inplace=True)
        else:
            self.metadata = pd.read_csv(self.preprocessed_dir / self.PP_METADATA_FN)
        if indices_fp:
            indices = cast(pd.Series, pd.read_csv(indices_fp, header=None).squeeze("columns"))
            self.metadata = self.metadata.iloc[indices]
        self.indices = self.metadata.index.to_numpy()
        self.chip = self.metadata["chip"].to_numpy()
        self.month = (
            self.metadata["month"].to_numpy() if self.group_by is _GroupBy.CHIP_MONTH else None
        )
        if self.preprocess and (not self.is_preprocessed):
            self._preprocess_data()

        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def preprocess(self) -> bool:
        return self._preprocess

    @property
    def split(self) -> str:
        return "train" if self.train else "test"

    @property
    def tile_file(self) -> Optional[Path]:
        if some(self.tile_dir):
            return (self.tile_dir / self.split).with_suffix(".csv")

    @property
    @lru_cache(maxsize=16)
    def preprocessed_dir(self: "SentinelDataset[TR, LitTrue]") -> Path:
        pp_dir_stem = f"group_by={self.group_by.name}_missing_value={self.missing_value.name}_precision={self.save_precision.name}"
        if some(self.tile_file):
            tf_stem = self.tile_file.stem
            pp_dir_stem += f"_tile_file={tf_stem}"
        subroot_dir = f"preprocessed_{self.save_with.value}"
        return self.root / subroot_dir / self.split / pp_dir_stem

    @property
    @lru_cache(maxsize=16)
    def is_preprocessed(self: "SentinelDataset[TR, LitTrue]") -> bool:
        return self.preprocessed_dir.exists()

    def _preprocess_data(self) -> None:
        pp_dir = self.preprocessed_dir
        pp_dir.mkdir(parents=True, exist_ok=False)

        def _load_save(index: int) -> None:
            sample = self._load_unprocessed(index)
            fp = pp_dir / str(index)
            if self.save_with is _SaveWith.NP:
                fp_suffixed = fp.with_suffix(".npz")
                # np.savez(file=fp_suffixed, **{key: value for key, value in sample.items()})
                np.savez_compressed(
                    file=fp_suffixed,
                    **{
                        key: value.to(self.save_precision.torch).numpy()
                        if isinstance(value, Tensor)
                        else value
                        for key, value in sample.items()
                    },
                    allow_pickle=False,
                )
            else:
                fp_suffixed = fp.with_suffix(".pt")
                if self.save_precision is _SavePrecision.HALF:
                    sample = {
                        key: value.to(self.save_precision.torch).numpy()
                        if isinstance(value, Tensor)
                        else value
                        for key, value in sample.items()
                    }
                torch.save(sample, f=fp_suffixed)

        logger.info(f"Starting data-preprocessing with output directory '{pp_dir.resolve()}'.")
        try:
            with tqdm_joblib(tqdm(desc="Preprocessing", total=len(self))):
                joblib.Parallel(n_jobs=self.n_pp_jobs)(
                    joblib.delayed(_load_save)(index) for index in range(len(self))
                )
            self.metadata.to_csv(pp_dir / self.PP_METADATA_FN)
            logger.info(
                f"Preprocessing complete! Preprocessed data has been saved to '{pp_dir.resolve()}'."
            )
        except (Exception, KeyboardInterrupt) as e:
            logger.info(e)
            shutil.rmtree(pp_dir)

    @property
    def in_channels(self) -> int:
        return self[0]["image"].size(self.CHANNEL_DIM)

    @property
    def sequence_length(self) -> int:
        return self[0]["image"].size(self.TEMPORAL_DIM)

    def input_size(self) -> torch.Size:
        return self[0]["image"].size()

    def out_channels(self: "SentinelDataset[LitTrue, P]") -> int:
        if not self.train:
            raise AttributeError(
                "Cannot retrieve size of target as no targets exist when 'train=False'."
            )
        return self[0]["label"].size(0)

    def target_size(self: "SentinelDataset[LitTrue, P]") -> torch.Size:
        if not self.train:
            raise AttributeError(
                "Cannot retrieve size of target as no targets exist when 'train=False'."
            )
        return self[0]["label"].size()

    def _read_tif_to_tensor(self, tif_path: Path) -> Tensor:
        with rasterio.open(tif_path, driver="GTiff") as src:
            tif = torch.as_tensor(
                src.read().astype(np.float32),
            )
        return tif

    def _missing_data(self, sentinel_type: SentinelType) -> Tensor:
        # Currently this funtion just produces constant-padding -- we might want to
        # investigate more sophisticated schemes (it might be wiser to set the
        # the CloudProb band to 1s, for instance).
        return torch.full(
            (sentinel_type.n_channels, self.RESOLUTION, self.RESOLUTION),
            fill_value=self.missing_value.float,
        )

    def _load_sentinel_tiles(
        self: "SentinelDataset",
        sentinel_type: SentinelType,
        *,
        chip: int,
        month: Optional[int],
    ) -> Tensor:
        if month is None:
            tiles_all_months = cast(
                List[Tensor],
                joblib.Parallel(n_jobs=1)(
                    joblib.delayed(self._load_sentinel_tiles)(
                        sentinel_type=sentinel_type, chip=chip, month=month
                    )
                    for month in range(12)
                ),
            )
            data = torch.cat(tiles_all_months, dim=self.TEMPORAL_DIM)
            return data

        filename = f"{chip}_{sentinel_type.name}_{str(month).zfill(2)}.tif"
        filepath = self.input_dir / filename
        if filepath.exists():
            data = self._read_tif_to_tensor(filepath)
        else:
            # Substitute data with padding if the data for the given month is unavailable
            data = self._missing_data(sentinel_type=sentinel_type)
        return data.unsqueeze(self.TEMPORAL_DIM)

    @overload
    def _load_agbm_tile(self: "SentinelDataset[LitTrue, LitTrue]", chip: int) -> Tensor:
        ...

    @overload
    def _load_agbm_tile(self: "SentinelDataset[LitFalse, LitTrue]", chip: int) -> None:
        ...

    def _load_agbm_tile(self: "SentinelDataset", chip: int) -> Optional[Tensor]:
        if some(self.target_dir):
            target_path = self.target_dir / f"{chip}_agbm.tif"
            return self._read_tif_to_tensor(target_path)

    def _generate_metadata(self, input_dir: Path) -> pd.DataFrame:
        filenames = pd.Series(input_dir.glob("*.tif"), dtype=str).str.rstrip(".tif")
        filenames = filenames.str.split("/", expand=True).iloc[:, -1]
        metadata = filenames.str.split("_", expand=True)
        metadata.rename(
            columns={0: "chip", 1: "sentinel", 2: "month"},
            inplace=True,
        )
        metadata.sort_values(by="chip", inplace=True)
        return metadata

    @overload
    def _load_unprocessed(self: "SentinelDataset[LitTrue, P]", index: int) -> TrainSample:
        ...

    @overload
    def _load_unprocessed(self: "SentinelDataset[LitFalse, P]", index: int) -> TestSample[str]:
        ...

    @overload
    def _load_unprocessed(
        self: "SentinelDataset[TR, P]", index: int
    ) -> Union[TestSample[str], TrainSample]:
        ...

    def _load_unprocessed(
        self: "SentinelDataset", index: int
    ) -> Union[TestSample[str], TrainSample]:
        chip = self.chip[index]
        month = None if self.month is None else self.month[index]
        # Load in the Sentinel1 imagery.
        s1_tile = self._load_sentinel_tiles(
            sentinel_type=_SentinelType.S1,
            chip=chip,
            month=month,
        )

        # Load in the Sentinel2 imagery.
        s2_tile = self._load_sentinel_tiles(
            sentinel_type=_SentinelType.S2,
            chip=chip,
            month=month,
        )
        # S2 data first, S1 data second.
        sentinel_tile = torch.cat((s2_tile, s1_tile), dim=0)
        sample: Union[TestSample[str], TrainSample]
        mask = self.missing_value.checker(sentinel_tile)
        if self.train:
            target_tile = self._load_agbm_tile(chip)
            sample = {"image": sentinel_tile, "label": target_tile, "mask": mask}
        else:
            sample = {"image": sentinel_tile, "chip": chip, "mask": mask}
        return sample

    @overload
    def _load_preprocessed(self: "SentinelDataset[LitTrue, LitTrue]", index: int) -> TrainSample:
        ...

    @overload
    def _load_preprocessed(
        self: "SentinelDataset[LitFalse, LitTrue]", index: int
    ) -> TestSample[str]:
        ...

    def _load_preprocessed(
        self: "SentinelDataset", index: int
    ) -> Union[TestSample[str], TrainSample]:
        if self.save_with is _SaveWith.NP:
            fp = self.preprocessed_dir / f"{self.indices[index]}.npz"
            sample = dict(np.load(file=fp))
            if "chip" in sample:
                sample["chip"] = sample["chip"].item()
                sample = cast(TestSample, sample)
            else:
                sample["label"] = torch.as_tensor(sample["label"], dtype=torch.float32)
                sample = cast(TrainSample, sample)
            sample["image"] = torch.as_tensor(sample["image"], dtype=torch.float32)
            # For backwards compatibility
            if sample["image"].ndim == 3:
                sample["image"] = sample["image"].unsqueeze(self.TEMPORAL_DIM)
            if "mask" not in sample:
                sample["mask"] = self.missing_value.checker(sample["image"])
            return sample

        fp = self.preprocessed_dir / f"{self.indices[index]}.pt"
        sample = torch.load(f=fp)
        # for backwards compatibility
        if "mask" not in sample:
            sample["mask"] = self.missing_value.checker(sample["image"])
        sample["image"] = sample["image"].to(torch.float32)
        if "chip" in sample:
            sample = cast(TestSample, sample)
        else:
            sample["label"] = sample["label"].to(torch.float32)
            sample = cast(TrainSample, sample)
        return sample

    @overload
    def __getitem__(self: "SentinelDataset[LitTrue, P]", index: int) -> TrainSample:
        ...

    @overload
    def __getitem__(self: "SentinelDataset[LitFalse, P]", index: int) -> TestSample[str]:
        ...

    @overload
    def __getitem__(
        self: "SentinelDataset[TR, P]", index: int
    ) -> Union[TestSample[str], TrainSample]:
        ...

    def __getitem__(self: "SentinelDataset", index: int) -> Union[TestSample[str], TrainSample]:
        if self.preprocess:
            sample = self._load_preprocessed(index)
        else:
            sample = self._load_unprocessed(index)
        if some(self.transform):
            sample = self.transform(sample)
        return sample

    def make_subset(
        self,
        indices: Union[List[int], npt.NDArray[np.uint64], Tensor, slice],
        *,
        deep: bool = False,
    ) -> Self:
        if isinstance(indices, (np.ndarray, Tensor)):
            if indices.ndim > 1:
                raise ValueError("If 'indices' is an array it must be a 0- or 1-dimensional.")
            indices = cast(List[int], indices.tolist())

        subset = gcopy(self, deep=deep)
        subset.indices = subset.indices[indices]
        subset.chip = subset.chip[indices]
        if some(subset.month):
            subset.month = subset.month[indices]
        subset.metadata = subset.metadata.iloc[indices]

        return subset

    @overload
    def random_split(
        self,
        props: Union[Sequence[float], float],
        *,
        deep: bool = ...,
        as_indices: LitTrue,
        seed: Optional[int] = ...,
    ) -> List[List[int]]:
        ...

    @overload
    def random_split(
        self,
        props: Union[Sequence[float], float],
        *,
        deep: bool = ...,
        as_indices: LitFalse = ...,
        seed: Optional[int] = ...,
    ) -> List[Self]:
        ...

    def random_split(
        self,
        props: Union[Sequence[float], float],
        *,
        deep: bool = False,
        as_indices: bool = False,
        seed: Optional[int] = None,
    ) -> Union[List[Self], List[List[int]]]:
        """Randomly split the dataset into subsets according to the given proportions.

        :param dataset: The dataset to split.
        :param props: The fractional size of each subset into which to randomly split the data.
            Elements must be non-negative and sum to 1 or less; if less then the size of the final
            split will be computed by complement.

        :param deep: Whether to create a copy of the underlying dataset as a basis for the random
            subsets. If False then the data of the subsets will be views of original dataset's data.

        :param as_indices: Whether to return the raw train/test indices instead of subsets of the
            dataset constructed from them.

        :param seed: PRNG seed to use for splitting the data.

        :returns: Random subsets of the data (or their associated indices) of the requested proportions
            with residuals.
        """
        split_indices = prop_random_split(dataset=self, props=props, as_indices=True, seed=seed)
        if as_indices:
            return split_indices
        return [self.make_subset(indices=indices, deep=deep) for indices in split_indices]
