from functools import partial
from pathlib import Path
from typing import Dict, Final, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale  # type: ignore
from torch import Tensor
from torchgeo.transforms import indices
from tqdm import tqdm

from src.data import SentinelDataset
import src.data.transforms as T

warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")


CHANNEL_MAP: Final[Dict[int, str]] = {
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
    11: "S1-VV-Asc: Cband-10m",
    12: "S1-VH-Asc: Cband-10m",
    13: "S1-VV-Desc: Cband-10m",
    14: "S1-VH-Desc: Cband-10m",
    15: "S2-NDVI: (NIR-Red)/(NIR+Red) 10m",
    16: "S1-VV/VH-Asc: Cband-10m",
}


def diff_ndvi_sar_vh(tile: Tensor, *, ndvi_idx: int, vh_idx: int):
    ndvi = minmax_scale(tile[ndvi_idx].clamp(0))
    vh = minmax_scale(tile[vh_idx])
    return ndvi - vh


def calc_frac_over_thresh(img: Tensor, *, thresh: float = 0.5) -> float:
    total_vals = 256 * 256.0
    count_bad = img[np.abs(img) > thresh].shape[0]
    count_bad += np.isnan(img).sum()
    return round((count_bad) / total_vals, 3)


def calc_quality_scores(dataset: SentinelDataset) -> pd.DataFrame:
    scores = []
    metadata = dataset.metadata.to_numpy()
    assert dataset.month is not None
    for idx, sample in tqdm(enumerate(iter(dataset)), total=len(dataset)):
        chipid, month_idx = dataset.chip[idx], dataset.month[idx]
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
    transforms = T.Compose(
        indices.AppendNDVI(index_nir=6, index_red=2),  # NDVI, index 15
        T.AppendRatioAB(index_a=11, index_b=12),  # VV/VH Ascending, index 16
    )
    fact_func = partial(
        SentinelDataset,
        root=root,
        group_by=SentinelDataset.GroupBy.CHIP_MONTH,
        transform=transforms,
        preprocess=False,
    )
    ds_train = fact_func(train=True)
    df_scores = calc_quality_scores(ds_train)
    df_best = find_best_months(df_scores)
    root = Path("sample_selection")
    df_best.to_csv(root / "tile_list_best_months" / "train.csv")

    ds_test = fact_func(train=False)
    df_scores_test = calc_quality_scores(ds_test)
    df_best_test = find_best_months(df_scores_test)
    df_best.to_csv(root / "tile_list_best_months" / "test.csv")
