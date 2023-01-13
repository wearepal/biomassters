from pathlib import Path
import time

from PIL import Image
import pytorch_lightning as pl
import torch

from src.data import SentinelDataModule, SentinelDataset
from src.data.dataset import SavePrecision
import src.data.transforms as T

if __name__ == "__main__":
    pl.seed_everything(1, workers=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
    root = Path("../datasets/biomassters/")
    dm = SentinelDataModule(
        root=root,
        pin_memory=False,
        group_by=SentinelDataset.GroupBy.CHIP,
        preprocess=True,
        # tile_dir=Path("sample_selection/tile_list_best_months"),
        train_batch_size=16,  # type: ignore
        eval_batch_size=16,  # type: ignore
        n_pp_jobs=64,
        num_workers=3,
        val_prop=0.2,
        save_with=SentinelDataset.SaveWith.NP,
        missing_value=SentinelDataset.MissingValue.INF,
        save_precision=SentinelDataset.SavePrecision.HALF,
    )
    dm.setup()
    transform = T.NanToNum()
    dm.train_data[0]
    stats = dm.train_statistics(compute_var=True)
    breakpoint()
