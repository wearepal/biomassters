from dataclasses import dataclass
from functools import partial
from pathlib import Path
import shutil
from typing import Any, Dict, Final, List, Tuple
import warnings

from PIL import Image
import joblib  # type: ignore
from loguru import logger
import numpy as np
import numpy.typing as npt
import pandas as pd
from ranzen.hydra import Relay
import rasterio  # type: ignore
import rasterio.errors  # type: ignore
from tqdm import tqdm
from typing_extensions import override

from src.data.dataset import SentinelDataset
from src.logging import tqdm_joblib
from src.utils import some, to_targz

__all__ = ["EnsembleRelay"]

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

TARGET_SHAPE: Final[Tuple[int, int]] = (SentinelDataset.RESOLUTION, SentinelDataset.RESOLUTION)


def _load_tif(tif_path: Path) -> npt.NDArray[np.float32]:
    with rasterio.open(tif_path, driver="GTiff") as src:
        tif = src.read().astype(np.float32)
    return tif


def _load_ensemble_save(*, source_dirs: List[Path], chip: str, output_dir: Path) -> None:
    ensembled_np = None
    for dir in source_dirs:
        fp = dir / f"{chip}_agbm.tif"
        tif = _load_tif(fp)
        if some(ensembled_np):
            ensembled_np += tif
        else:
            ensembled_np = tif
    assert some(ensembled_np)
    ensembled_np /= len(source_dirs)
    ensembled_np = ensembled_np.reshape(TARGET_SHAPE)
    ensembled_pil = Image.fromarray(ensembled_np)
    save_path = output_dir / f"{chip}_agbm.tif"
    ensembled_pil.save(save_path, format="TIFF", save_all=True)


@dataclass(unsafe_hash=True)
class EnsembleRelay(Relay):
    source_dirs: List[Path]
    output_dir: Path
    metadata_fp: Path = Path(
        "../datasets/biomassters/preprocessed_np/test/group_by=CHIP_missing_value=INF_precision=HALF/metadata.csv"
    )
    n_jobs: int = 1
    archive: bool = False

    def __post_init__(self) -> None:
        self.output_dir.mkdir(exist_ok=False, parents=True)

    @override
    def run(self, raw_config: Dict[str, Any]) -> None:
        chips = pd.read_csv(self.metadata_fp)["chip"].tolist()
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
