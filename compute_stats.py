from pathlib import Path

import pytorch_lightning as pl
import torch

from src.data import SentinelDataModule, SentinelDataset
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
        train_batch_size=16,
        eval_batch_size=16,
        n_pp_jobs=64,
        num_workers=3,
        val_prop=0.2,
        save_with=SentinelDataset.SaveWith.NP,
        missing_value=SentinelDataset.MissingValue.INF,
        save_precision=SentinelDataset.SavePrecision.HALF,
        train_transforms=T.Identity(),
        eval_transforms=T.Identity(),
    )
    dm.setup()
    stats = dm.train_statistics(compute_var=True)
    print(stats)
