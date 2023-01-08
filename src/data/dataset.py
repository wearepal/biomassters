from enum import Enum
from functools import lru_cache
from pathlib import Path
import shutil
from typing import ClassVar, Dict, Generic, Literal, Optional, TypeVar, Union, overload
import warnings

import joblib  # type: ignore
from loguru import logger
import numpy as np
import pandas as pd
import rasterio  # type: ignore
import rasterio.errors  # type: ignore
import torch
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from typing_extensions import TypeAlias

from src.data.transforms import (
    InputTransform,
    TargetTransform,
    scale_sentinel1_data,
    scale_sentinel2_data,
)
from src.logging import tqdm_joblib
from src.types import LitFalse, LitTrue, SampleL, SampleU

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

__all__ = ["SentinelDataset"]


class SentinelType(Enum):
    S1 = 4
    S2 = 11

    def __init__(self, n_channels: int) -> None:
        self.n_channels = n_channels


class GroupBy(Enum):
    CHIP = ["chip"]
    CHIP_MONTH = ["chip", "month"]


TR = TypeVar("TR", LitTrue, LitFalse)
P = TypeVar("P", LitTrue, LitFalse)


TemporalDim: TypeAlias = Literal[0, 1]


class SentinelDataset(Dataset[SampleU], Generic[TR, P]):
    """Sentinel 1 & 2 dataset."""

    SentinelType: TypeAlias = SentinelType
    GroupBy: TypeAlias = GroupBy

    RESOLUTION: ClassVar[int] = 256
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
        # Sentinel1 channels
        0: "S1-VV-Asc: Cband-10m",
        1: "S1-VH-Asc: Cband-10m",
        2: "S1-VV-Desc: Cband-10m",
        3: "S1-VH-Desc: Cband-10m",
        # Sentinel2 channels
        4: "S2-B2: Blue-10m",
        5: "S2-B3: Green-10m",
        6: "S2-B4: Red-10m",
        7: "S2-B5: VegRed-704nm-20m",
        8: "S2-B6: VegRed-740nm-20m",
        9: "S2-B7: VegRed-780nm-20m",
        10: "S2-B8: NIR-833nm-10m",
        11: "S2-B8A: NarrowNIR-864nm-20m",
        12: "S2-B11: SWIR-1610nm-20m",
        13: "S2-B12: SWIR-2200nm-20m",
        14: "S2-CLP: CloudProb-160m",
    }

    @overload
    def __init__(
        self,
        root: Union[Path, str],
        *,
        train: LitTrue,
        tile_dir: Optional[Path] = ...,
        group_by: GroupBy = ...,
        temporal_dim: TemporalDim = ...,
        preprocess: LitTrue = ...,
        n_pp_jobs: int = ...,
        transform: Optional[Union[InputTransform, TargetTransform]] = ...,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        root: Union[Path, str],
        *,
        train: LitTrue = ...,
        tile_dir: Optional[Path] = ...,
        group_by: GroupBy = ...,
        temporal_dim: TemporalDim = ...,
        preprocess: LitFalse,
        n_pp_jobs: int = ...,
        transform: Optional[Union[InputTransform, TargetTransform]] = ...,
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
        temporal_dim: TemporalDim = ...,
        preprocess: LitTrue = ...,
        n_pp_jobs: int = ...,
        transform: Optional[InputTransform] = ...,
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
        temporal_dim: TemporalDim = ...,
        preprocess: LitFalse,
        n_pp_jobs: int = ...,
        transform: Optional[InputTransform] = ...,
    ) -> None:
        ...

    def __init__(
        self,
        root: Union[Path, str],
        *,
        train: TR = True,
        tile_dir: Optional[Path] = None,
        group_by: GroupBy = GroupBy.CHIP_MONTH,
        temporal_dim: TemporalDim = 1,
        preprocess: P = True,
        n_pp_jobs: int = 8,
        transform: Optional[Union[InputTransform, TargetTransform]] = None,
    ) -> None:
        self.root = Path(root)
        self.train = train
        self.group_by = group_by
        self.temporal_dim = temporal_dim
        self.n_pp_jobs = n_pp_jobs
        self.input_dir = self.root / (self.TRAIN_DIR_NAME if self.train else self.TEST_DIR_NAME)
        self.target_dir = (self.root / self.TARGET_DIR_NAME) if self.train else None
        self.tile_dir = tile_dir
        self._preprocess = preprocess

        if (not self.preprocess) or (not self.is_preprocessed):
            if self.tile_file is None:
                self.metadata = self._generate_metadata(self.input_dir)
            else:
                self.metadata = pd.read_csv(self.tile_file, index_col=0)
            # Currrently we sample a chip/month and then sample both data from
            # both satellites; this scheme is very much subject to change. For instance,
            # we might wish to sample by chip and treat the monthly data as a spatiotemporal series.
            self.metadata.drop_duplicates(subset=self.group_by.value, inplace=True)
        else:
            self.metadata = pd.read_csv(self.preprocessed_dir / self.PP_METADATA_FN)
        self.chip = self.metadata["chip"].to_numpy()
        self.month = (
            self.metadata["month"].to_numpy() if self.group_by is GroupBy.CHIP_MONTH else None
        )
        if self.preprocess and (not self.is_preprocessed):
            self._preprocess_data()

        self.transform = transform

    @property
    def preprocess(self) -> bool:
        return self._preprocess

    def __len__(self) -> int:
        return len(self.metadata)

    @property
    def split(self) -> str:
        return "train" if self.train else "test"

    @property
    def tile_file(self) -> Optional[Path]:
        return None if self.tile_dir is None else (self.tile_dir / self.split).with_suffix(".csv")

    @property
    @lru_cache(maxsize=16)
    def preprocessed_dir(self: "SentinelDataset[TR, LitTrue]") -> Path:
        pp_dir_stem = f"group_by={self.group_by.name}"
        if self.group_by is GroupBy.CHIP:
            pp_dir_stem += f"_temporal_dim={self.temporal_dim}"
        if self.tile_file is not None:
            tf_stem = self.tile_file.stem
            pp_dir_stem += f"_tile_file={tf_stem}"
        return self.root / "preprocessed" / self.split / pp_dir_stem

    @property
    @lru_cache(maxsize=16)
    def is_preprocessed(self: "SentinelDataset[TR, LitTrue]") -> bool:
        return self.preprocessed_dir.exists()

    def _preprocess_data(self) -> None:
        pp_dir = self.preprocessed_dir
        pp_dir.mkdir(parents=True, exist_ok=False)

        def _load_save(index: int) -> None:
            torch.save(self._load_unprocessed(index), f=pp_dir / f"{index}.pt")

        try:
            with tqdm_joblib(tqdm(desc="Preprocessing", total=len(self))):
                joblib.Parallel(n_jobs=self.n_pp_jobs)(
                    joblib.delayed(_load_save)(index) for index in range(len(self))
                )
        except KeyboardInterrupt as e:
            logger.info(e)
            shutil.rmtree(pp_dir)

        self.metadata.to_csv(pp_dir / self.PP_METADATA_FN)

    @property
    @lru_cache(maxsize=16)
    def channel_dim(self) -> int:
        return 1 - self.temporal_dim if self.group_by is GroupBy.CHIP else 0

    def in_channels(self) -> int:
        return self[0]["image"].size(self.channel_dim)

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
        with rasterio.open(tif_path) as src:
            tif = torch.as_tensor(
                src.read().astype(np.float32),
                dtype=torch.float32,
            )
        return tif

    def _load_sentinel_tiles(
        self: "SentinelDataset",
        sentinel_type: SentinelType,
        *,
        chip: int,
        month: Optional[int],
    ) -> Tensor:
        if month is None:
            tiles_all_months = [
                self._load_sentinel_tiles(sentinel_type=sentinel_type, chip=chip, month=month)
                for month in range(12)
            ]
            return torch.stack(tiles_all_months, dim=self.temporal_dim)
        filename = f"{chip}_{sentinel_type.name}_{str(month).zfill(2)}.tif"
        filepath = self.input_dir / filename
        if filepath.exists():
            return self._read_tif_to_tensor(filepath)
        # Substitute data with zero-padding if the data for the given month is unavailable
        return torch.zeros((sentinel_type.n_channels, self.RESOLUTION, self.RESOLUTION))

    @overload
    def _load_agbm_tile(self: "SentinelDataset[LitTrue, LitTrue]", chip: int) -> Tensor:
        ...

    @overload
    def _load_agbm_tile(self: "SentinelDataset[LitFalse, LitTrue]", chip: int) -> None:
        ...

    def _load_agbm_tile(self: "SentinelDataset", chip: int) -> Optional[Tensor]:
        if self.target_dir is not None:
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
    def _load_unprocessed(self: "SentinelDataset[LitTrue, P]", index: int) -> SampleL:
        ...

    @overload
    def _load_unprocessed(self: "SentinelDataset[LitFalse, P]", index: int) -> SampleU:
        ...

    @overload
    def _load_unprocessed(self: "SentinelDataset[TR, P]", index: int) -> Union[SampleU, SampleL]:
        ...

    def _load_unprocessed(self: "SentinelDataset", index: int) -> Union[SampleU, SampleL]:
        chip = self.chip[index]
        month = None if self.month is None else self.month[index]
        # Load in the Sentinel1 imagery.
        s1_tile = self._load_sentinel_tiles(
            sentinel_type=SentinelType.S1,
            chip=chip,
            month=month,
        )

        s1_tile_scaled = scale_sentinel1_data(s1_tile)
        # Load in the Sentinel2 imagery.
        s2_tile = self._load_sentinel_tiles(
            sentinel_type=SentinelType.S2,
            chip=chip,
            month=month,
        )
        s2_tile_scaled = scale_sentinel2_data(s2_tile)

        sentinel_tile = torch.cat((s1_tile_scaled, s2_tile_scaled), dim=0)
        sample: Union[SampleU, SampleL]
        if self.train:
            target_tile = self._load_agbm_tile(chip)
            sample = {"image": sentinel_tile, "label": target_tile}
        else:
            sample = {"image": sentinel_tile}
        return sample

    @overload
    def _load_preprocessed(self: "SentinelDataset[LitTrue, LitTrue]", index: int) -> SampleL:
        ...

    @overload
    def _load_preprocessed(self: "SentinelDataset[LitFalse, LitTrue]", index: int) -> SampleU:
        ...

    def _load_preprocessed(self: "SentinelDataset", index: int) -> Union[SampleU, SampleL]:
        return torch.load(f=self.preprocessed_dir / f"{index}.pt")

    @overload
    def __getitem__(self: "SentinelDataset[LitTrue, P]", index: int) -> SampleL:
        ...

    @overload
    def __getitem__(self: "SentinelDataset[LitFalse, P]", index: int) -> SampleU:
        ...

    @overload
    def __getitem__(self: "SentinelDataset[TR, P]", index: int) -> Union[SampleU, SampleL]:
        ...

    def __getitem__(self: "SentinelDataset", index: int) -> Union[SampleU, SampleL]:
        if self.preprocess:
            sample = self._load_preprocessed(index)
        else:
            sample = self._load_unprocessed(index)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
