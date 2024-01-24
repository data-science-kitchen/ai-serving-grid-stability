import fire
import numpy as np
import os
import pandas as pd

from os import PathLike
from typing import Optional, Tuple, Union


def main(data_dir: Union[str, PathLike] = "data"):
    if not os.path.isdir(os.path.join("data", "label_studio")):
        os.makedirs(os.path.join("data", "label_studio"), exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "test.csv"), low_memory=False)
    # test_df = pd.read_csv(os.path.join(data_dir, "test.csv"), low_memory=False)

    segment_id = 0

    for control_area in [1, 2]:
        train_df_area = train_df[train_df["controlArea"] == control_area]

        num_segments = int(np.floor(len(train_df_area) / 1800))

        for idx in range(num_segments):
            start_idx = idx * 1800
            end_idx = start_idx + 1800

            segment = train_df_area[start_idx:end_idx]
            segment = segment.drop(columns={"id"})
            segment["segment_id"] = segment_id

            segment.to_csv(os.path.join("data", "label_studio", f"test_{segment_id:04d}.csv"), index=False)
            segment_id += 1


if __name__ == "__main__":
    fire.Fire(main)
