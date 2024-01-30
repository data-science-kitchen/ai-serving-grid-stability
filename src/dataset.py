import os

import numpy as np
import pandas as pd

from os import PathLike
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from typing import List, Optional, Union


class GridStabilityDataset(Dataset):
    SEQUENCE_LENGTH: int = 1800

    def __init__(self, data_dir: Union[str, PathLike], split="train", target_feature: Optional[str] = None) -> None:
        dataset_train_raw = pd.read_csv(os.path.join(data_dir, "train.csv"), low_memory=False)

        dataset_train: List[DataFrame] = []

        for control_area in [1, 2]:
            subset_train = dataset_train_raw[dataset_train_raw["controlArea"] == control_area]
            num_segments = len(subset_train) // self.SEQUENCE_LENGTH

            for segment_idx in range(num_segments):
                segment_idx_start = segment_idx * self.SEQUENCE_LENGTH
                segment_idx_end = segment_idx_start + self.SEQUENCE_LENGTH

                signal = subset_train[segment_idx_start:segment_idx_end]
                signal = signal.drop(
                    columns=[
                        "Datum_Uhrzeit_CET",
                        "id",
                        "participationCMO",
                        "participationIN",
                    ],
                )

                dataset_train.append(signal)

        self.scaler = StandardScaler()

        dataset_train_df = pd.concat(dataset_train)
        self.feature_names = dataset_train_df.columns

        self.scaler.fit(dataset_train_df)

        dataset_test_raw = pd.read_csv(os.path.join(data_dir, "test.csv"), low_memory=False)

        dataset_test: List[DataFrame] = []

        for segment_id in dataset_test_raw["test_data_segment_id"].unique():
            subset_test = dataset_test_raw[dataset_test_raw["test_data_segment_id"] == segment_id]

            signal = subset_test.drop(
                columns=[
                    "Datum_Uhrzeit_CET",
                    "id",
                    "participationCMO",
                    "participationIN",
                    "test_data_segment_id",
                ],
            )

            dataset_test.append(signal)

        if split == "train":
            self.dataset = dataset_train
        elif split == "test":
            self.dataset = dataset_test
        else:
            raise KeyError(f"Split method {split} is not supported. Please select either 'train' or 'test'.")

        if target_feature is not None:
            self.set_target_feature(target_feature)
        else:
            self.target_feature = None
            self.target_idx = None

    def set_target_feature(self, target_feature: str) -> None:
        if target_feature not in self.feature_names:
            raise ValueError(f"Target feature must be one of the following: {', '.join(self.feature_names)}.")

        self.target_feature = target_feature
        self.target_idx = a = np.asarray([1 if x == target_feature else 0 for x in self.feature_names]).astype(bool)

    def __getitem__(self, idx: int):
        segment = self.dataset[idx]
        segment = segment[self.feature_names]  # Make sure columns are in the correct order.
        segment = self.scaler.transform(segment).astype(np.float32)

        if self.target_feature is None:
            raise ValueError("Please specify a target feature before calling __get_item__().")

        return segment[:, ~self.target_idx], segment[:, self.target_idx]

    def __len__(self):
        return len(self.dataset)
