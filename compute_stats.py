from pathlib import Path
import time

from PIL import Image
import pytorch_lightning as pl
import torch

from src.data import SentinelDataModule, SentinelDataset

if __name__ == "__main__":
    pl.seed_everything(1, workers=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
    root = Path("/srv/galene0/shared/data/biomassters/")
    dm = SentinelDataModule(
        root=root,
        pin_memory=False,
        group_by=SentinelDataset.GroupBy.CHIP,
        preprocess=False,
        # tile_dir=Path("sample_selection/tile_list_best_months"),
        train_batch_size=16,  # type: ignore
        eval_batch_size=16,  # type: ignore
        n_pp_jobs=3,
        num_workers=16,
        save_with=SentinelDataset.SaveWith.TORCH,
    )
    dm.setup()
    stats = dm.train_statistics(compute_var=True)
