from dataclasses import dataclass
from src.utils import to_targz
from loguru import logger
import shutil
from typing_extensions import override
from functools import partial
from pathlib import Path
from typing import List
import warnings

from PIL import Image
import joblib  # type: ignore
import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio  # type: ignore
import rasterio.errors  # type: ignore
from tqdm import tqdm
from src.data.dataset import SentinelDataset

from src.logging import tqdm_joblib
from src.utils import some
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ranzen.hydra import Relay
from torch.types import Number


__all__ = ["EnsembleRelay"]

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


def _load_tif(tif_path: Path) -> npt.NDArray[np.float32]:
    with rasterio.open(tif_path, driver="GTiff") as src:
        tif = src.read().astype(np.float32)
    return tif


def _load_ensemble_save(*, source_dirs: List[Path], chip: str, output_dir: Path) -> None:
    merged_np = None
    for dir in source_dirs:
        fp = dir / f"{chip}_agbm.tif"
        tif = _load_tif(fp)
        if some(merged_np):
            merged_np += tif
        else:
            merged_np = tif
    assert some(merged_np)
    merged_np /= len(source_dirs)
    merged_np = merged_np.reshape((SentinelDataset.RESOLUTION, SentinelDataset.RESOLUTION))
    merged_pil = Image.fromarray(merged_np)
    save_path = output_dir / f"{chip}_agbm.tif"
    merged_pil.save(save_path, format="TIFF", save_all=True)


@dataclass(unsafe_hash=True)
class EnsembleRelay(Relay):
    source_dirs: List[Path]
    output_dir: Path
    metadata_fp: Path = Path(
        "../datasets/biomassters/preprocessed_np/test/group_by=CHIP_missing_value=INF_precision=HALF/metadata.csv"
    )
    n_jobs: int = 1
    archive: bool = False

    @classmethod
    @override
    def with_hydra(
        cls,
        root: Union[Path, str],
        *,
        clear_cache: bool = False,
    ) -> None:
        super().with_hydra(
            root=root,
            instantiate_recursively=False,
            clear_cache=clear_cache,
        )

    def __post_init__(self) -> None:
        self.output_dir.mkdir(exist_ok=False, parents=True)

    @override
    def run(self, raw_config: Dict[str, Any]) -> Optional[Number]:
        chips = pd.read_csv(self.metadata_fp)["chip"].to_numpy().tolist()
        fn = partial(_load_ensemble_save, source_dirs=self.source_dirs, output_dir=self.output_dir)
        try:
            with tqdm_joblib(tqdm(desc="Ensembling predictions", total=len(chips))):
                joblib.Parallel(n_jobs=self.n_jobs)(joblib.delayed(fn)(chip=chip) for chip in chips)
        except (Exception, KeyboardInterrupt) as e:
            logger.info(e)
            shutil.rmtree(self.output_dir)
        logger.info(f"Predictions ensembled and saved to {self.output_dir.resolve()}")
        if self.archive:
            logger.info("Archiving the ensembled predictions so they're ready-for-submission.")
            archive_path = to_targz(source=self.output_dir)
            logger.info(f"Ensembled predictions archived to {archive_path.resolve()}")
