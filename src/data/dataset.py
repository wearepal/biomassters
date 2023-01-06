from enum import Enum
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

from src.data.transforms import InputTransform, Sentinel1Scale, Sentinel2Scale
from src.types import LitFalse, LitTrue, Sample, SampleL, SampleU

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

__all__ = ["SentinelDataset"]


class SentinelType(Enum):
    S1 = "S1"
    S2 = "S2"


B = TypeVar("B", LitTrue, LitFalse)


class SentinelDataset(Dataset[SampleU], Generic[B]):
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
        transform: Optional[InputTransform] = None,
    ) -> None:
        self.train = train
        self.root = Path(root)
        self.input_dir = self.root / (self.TRAIN_DIR_NAME if self.train else self.TEST_DIR_NAME)
        self.target_dir = (self.root / self.TARGET_DIR_NAME) if self.train else None
        self.tile_file = tile_file

        if self.tile_file is None:
            self.metadata = self._generate_metadata()
        else:
            self.metadata = pd.read_csv(self.tile_file, index_col=0)
        if max_chips is not None:
            self.metadata = self.metadata.iloc[:max_chips]
        self.chip = self.metadata["chip"].to_numpy()
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
    def __getitem__(self: "SentinelDataset[LitTrue]", index: int) -> SampleL:
        ...

    @overload
    def __getitem__(self: "SentinelDataset[LitFalse]", index: int) -> SampleU:
        ...

    def __getitem__(self: "SentinelDataset", index: int) -> Sample:
        chip = self.chip[index]
        month = self.month[index]

        s1_tile = self._load_sentinel_tiles(
            sentinel_type=SentinelType.S1,
            chip=chip,
            month=month,
        )

        s1_tile_scaled = self.transform_s1(s1_tile)
        sentinel_tile = s1_tile_scaled

        s2_tile = self._load_sentinel_tiles(
            sentinel_type=SentinelType.S2,
            chip=chip,
            month=month,
        )
        if s2_tile is not None:
            s2_tile_scaled = self.transform_s2(s2_tile)
        # TODO: THIS IS BAD
        else:
            # Replace S2 channels with zero-padding if S2 data is irretrievable for
            # the given chip.
            s2_tile_scaled = s1_tile_scaled.new_zeros((11, 256, 256))
        sentinel_tile = torch.cat([s1_tile_scaled, s2_tile_scaled], dim=0)
        sample = { "image": sentinel_tile}
        if self.train:
            target_tile = self._load_agbm_tile(chip)
            sample["label"] =  target_tile

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

    @overload
    def _load_sentinel_tiles(
        self, sentinel_type: Literal[SentinelType.S1], *, chip: int, month: str
    ) -> Tensor:
        ...

    # Some chip are missing S2 data.
    @overload
    def _load_sentinel_tiles(
        self, sentinel_type: Literal[SentinelType.S2], *, chip: int, month: str
    ) -> Optional[Tensor]:
        ...

    def _load_sentinel_tiles(
        self: "SentinelDataset", sentinel_type: SentinelType, *, chip: int, month: str
    ) -> Optional[Tensor]:
        filename = f"{chip}_{sentinel_type.value}_{str(month).zfill(2)}.tif"
        filepath = self.input_dir / filename
        if filepath.exists():
            return self._read_tif_to_tensor(filepath)

    @overload
    def _load_agbm_tile(self: "SentinelDataset[LitTrue]", chip: int) -> Tensor:
        ...
    @overload
    def _load_agbm_tile(self: "SentinelDataset[LitFalse]", chip: int) -> None:
        ...

    def _load_agbm_tile(self: "SentinelDataset", chip: int) -> Optional[Tensor]:
        if self.target_dir is not None:
            target_path = self.target_dir / f"{chip}_agbm.tif"
            return self._read_tif_to_tensor(target_path)

    def _generate_metadata(self) -> pd.DataFrame:
        filenames = pd.Series(self.input_dir.glob("*.tif"), dtype=str).str.rstrip(".tif")
        filenames = filenames.str.split("/", expand=True).iloc[:, -1]
        metadata = filenames.str.split("_", expand=True)
        metadata.rename(
            columns={0: "chip", 1: "sentinel", 2: "month"},
            inplace=True,
        )
        # Currrently we sample a chip/month and then sample both data from
        # both satellites; this scheme is very much subject to change. For instance,
        # we might wish to sample by chip and treat the monthly data as a spatiotemporal series.
        metadata.drop_duplicates(subset=["chip", "month"], inplace=True)
        metadata.sort_values(by="chip", inplace=True)
        return metadata
