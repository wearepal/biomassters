from pathlib import Path
from typing import Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale  # type: ignore
from torch import Tensor
from torchgeo.transforms import indices
from tqdm import tqdm

from src.data import InputTransform, SentinelDataset
import src.data.transforms as T

warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")


def diff_ndvi_sar_vh(tile: Tensor, *, ndvi_idx: int, vh_idx: int):
    ndvi = minmax_scale(tile[ndvi_idx].clamp(0))
    vh = minmax_scale(tile[vh_idx])
    return ndvi - vh


def calc_frac_over_thresh(img: Tensor, thresh: float = 0.5) -> float:
    total_vals = 256 * 256.0
    count_bad = img[np.abs(img) > thresh].shape[0]
    count_bad += np.isnan(img).sum()
    return round((count_bad) / total_vals, 3)


def calc_quality_scores(dataset: SentinelDataset) -> pd.DataFrame:
    scores = []
    metadata = dataset.metadata.to_numpy()
    for idx, sample in tqdm(enumerate(iter(dataset)), total=len(dataset)):
        chipid, month_idx = metadata[idx]
        tile = sample["image"].detach().clone().cpu()
        diff_img = diff_ndvi_sar_vh(tile, ndvi_idx=15, vh_idx=12)
        score = 1 - calc_frac_over_thresh(diff_img, thresh=0.5)
        scores.append((chipid, month_idx, score))
    return pd.DataFrame(scores, columns=["chipid", "month", "score"])


def find_best_month_score(
    scores: np.ndarray, *, high_thresh: float = 0.95, min_thresh: float = 0.9
) -> Tuple[int, float]:
    """Calculate and return the best band score per heuristics

    Inputs:
    scores -- quality metric for each of 12 months from Sep = 0 to Aug = 11
    high_thresh -- high quality score threshold
    min_thresh -- lower quality score threshold
    """

    ranked_indexes = [8, 9, 7, 10, 11, 0]  # best months for vegetation data

    for idx in ranked_indexes:  # first see if any favored months meet high threshold
        if scores[idx] > high_thresh:
            return idx, scores[idx].item()

    for idx in ranked_indexes:  # then try lower threshold
        if scores[idx] > min_thresh:
            return idx, scores[idx].item()

    idx = np.argmax(scores).item()
    return idx, scores[idx].item()  # otherwise return month with highest score


def find_best_months(df_scores: pd.DataFrame) -> pd.DataFrame:
    best_months = []
    for chipid in tqdm(df_scores["chipid"].unique()):
        idx, score = find_best_month_score(df_scores[df_scores["chipid"] == chipid]["score"].values)
        best_months.append((chipid, idx, score))
    return pd.DataFrame(best_months, columns=["chipid", "month", "score"])


if __name__ == "__main__":
    root = Path("/srv/galene0/shared/data/biomassters/")
    transforms: InputTransform = T.Compose(
        indices.AppendNDVI(index_nir=10, index_red=4),  # type: ignore
        T.AppendRatioAB(index_a=0, index_b=1),  # VV/VH Ascending, index 16
    )
    ds_train = SentinelDataset(
        root=root,
        train=True,
        group_by=SentinelDataset.GroupBy.CHIP,
        transform=transforms,
    )
    df_scores = calc_quality_scores(ds_train)
    df_best = find_best_months(df_scores)
    df_best.to_csv("./TILE_LIST_BEST_MONTHS.csv")

    ds_test = SentinelDataset(
        root=root,
        train=False,
        group_by=SentinelDataset.GroupBy.CHIP,
        transform=transforms,
    )
    df_scores_test = calc_quality_scores(ds_test)
    df_best_test = find_best_months(df_scores_test)
    df_best_test.to_csv("./TILE_LIST_BEST_TEST_MONTHS_v2.csv")
