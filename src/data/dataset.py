from enum import Enum
from glob import glob
import os
from pathlib import Path
from typing import ClassVar, Generic, Literal, Optional, TypeVar, Union, overload
import warnings

import numpy as np
import pandas as pd
import rasterio  # type: ignore
import rasterio.errors  # type: ignore
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.transforms import SampleTransform, Sentinel1Scale, Sentinel2Scale
from src.types import LitFalse, LitTrue, Sample

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

__all__ = ["SentinelDataset"]


class SentinelType(Enum):
    S1 = "S1"
    S2 = "S2"


B = TypeVar("B", LitTrue, LitFalse)


class SentinelDataset(Dataset[Sample], Generic[B]):
    """Sentinel 1 & 2 dataset."""

    TRAIN_DIR_NAME: ClassVar[str] = "train_features"
    TARGET_DIR_NAME: ClassVar[str] = "train_agbm"
    TEST_DIR_NAME: ClassVar[str] = "test_features"

    def __init__(
        self,
        root: Union[Path, str],
        *,
        tile_file: Optional[Path] = None,
        max_chips: Optional[int] = None,
        train: B = True,
        transform: Optional[SampleTransform] = None,
    ) -> None:
        self.root = Path(root)
        self.img_dir = self.root / (self.TRAIN_DIR_NAME if train else self.TEST_DIR_NAME)
        self.dir_target = (self.root / self.TARGET_DIR_NAME) if train else None
        self.train = train

        if tile_file is None:
            self.metadata = self._make_df_tile_list(self.img_dir)
        else:
            self.metadata = pd.read_csv(tile_file, index_col=0)
        if max_chips is not None:
            self.metadata = self.metadata.iloc[:max_chips]
        self.chipid = self.metadata["chipid"].to_numpy()
        self.month = self.metadata["month"].to_numpy()

        self.transform_s2 = Sentinel2Scale()
        self.transform_s1 = Sentinel1Scale()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def in_channels(self) -> int:
        return self[0]["image"].size(0)

    def input_size(self) -> torch.Size:
        return self[0]["image"].size()

    def out_channels(self) -> int:
        if not self.train:
            raise AttributeError(
                "Cannot retrieve size of target as no targets exist when 'train=False'."
            )
        sample = self[0]["label"]
        return sample.size(0)

    def target_size(self) -> Optional[torch.Size]:
        if not self.train:
            raise AttributeError(
                "Cannot retrieve size of target as no targets exist when 'train=False'."
            )
        return self[0]["label"].size()

    @overload
    def __getitem__(self: "SentinelDataset[LitTrue]", index: int) -> Sample[Tensor]:
        ...

    @overload
    def __getitem__(self: "SentinelDataset[LitFalse]", index: bool) -> Sample[None]:
        ...

    def __getitem__(self: "SentinelDataset", index: int) -> Sample:
        chipid = self.chipid[index]
        month = self.month[index]
        s1_tile = self._load_sentinel_tiles(
            sentinel_type=SentinelType.S1,
            chipid=chipid,
            month=month,
        )
        s1_tile_scaled = self.transform_s1(s1_tile)
        s2_tile = self._load_sentinel_tiles(
            sentinel_type=SentinelType.S2,
            chipid=chipid,
            month=month,
        )
        s2_tile_scaled = self.transform_s2(s2_tile)
        sentinel_tile = torch.cat([s1_tile_scaled, s2_tile_scaled], dim=0)
        sentinel_tile = s1_tile_scaled
        target_tile = self._load_agbm_tile(chipid)

        sample: Sample = {
            "image": sentinel_tile,
            "label": target_tile,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _read_tif_to_tensor(self, tif_path: Path) -> Tensor:
        with rasterio.open(tif_path) as src:
            tif = torch.as_tensor(
                src.read().astype(np.float32),
                dtype=torch.float32,
            )
        return tif

    def _load_sentinel_tiles(
        self, sentinel_type: SentinelType, *, chipid: int, month: str
    ) -> Tensor:
        filename = f"{chipid}_{sentinel_type.value}_{str(month).zfill(2)}.tif"
        filepath = self.img_dir / filename
        return self._read_tif_to_tensor(filepath)

    def _load_agbm_tile(self, chipid: int) -> Optional[Tensor]:
        if self.dir_target is not None:
            target_path = self.dir_target / f"{chipid}_agbm.tif"
            return self._read_tif_to_tensor(target_path)

    def _make_df_tile_list(self, dir_tiles: Path) -> pd.DataFrame:
        tile_files = [os.path.basename(f).split(".")[0] for f in glob(f"{dir_tiles}/*.tif")]
        tile_tuples = []
        for tile_file in tile_files:
            chipid, _, month = tile_file.split("_")
            tile_tuples.append(tuple([chipid, int(month)]))
        tile_tuples = list(set(tile_tuples))
        tile_tuples.sort()
        return pd.DataFrame(tile_tuples, columns=["chipid", "month"])
