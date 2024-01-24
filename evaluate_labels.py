import fire
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from os import PathLike
from sklearn.metrics import fbeta_score
from typing import Union


def main(
    test_signal_file: Union[str, PathLike], test_label_file: Union[str, PathLike], submissions_dir: Union[str, PathLike]
) -> None:
    timestamps = pd.read_csv(test_signal_file)[["id", "Datum_Uhrzeit_CET"]]
    timestamps["Datum_Uhrzeit_CET"] = pd.to_datetime(timestamps["Datum_Uhrzeit_CET"], errors="coerce")
    timestamps = timestamps.set_index("Datum_Uhrzeit_CET")
    timestamps["anomaly"] = 0

    test_labels = pd.read_csv(test_label_file)

    # Keep test signals only.
    test_labels = test_labels[test_labels["csv"].str.contains("test")]
    test_labels["filename"] = test_labels["csv"].str.split("-").apply(lambda x: x[-1])
    test_labels = test_labels.sort_values("filename")

    labels = []

    for i, file in enumerate(test_labels["filename"].unique()):
        start_idx = i * 1800
        end_idx = start_idx + 1800
        timestamps_signal = timestamps[(timestamps["id"] >= start_idx) & (timestamps["id"] < end_idx)]

        signal_labels = test_labels[test_labels["filename"] == file]

        for row in signal_labels["label"]:
            if isinstance(row, str):
                annotations = pd.DataFrame.from_dict(json.loads(row))[["start", "end"]]
                annotations["start"] = pd.to_datetime(annotations["start"], errors="coerce")
                annotations["end"] = pd.to_datetime(annotations["end"], errors="coerce")

                for _, segment in annotations.iterrows():
                    timestamps_signal.loc[segment["start"] : segment["end"], "anomaly"] = 1

        labels.append(timestamps_signal)

    labels_data_frame = pd.concat(labels).reset_index(drop=True)

    # Compare with submissions
    submission_files = [f for f in os.listdir(submissions_dir) if f.endswith(".csv")]

    results = []

    for submission_idx, file in enumerate(submission_files):
        kaggle_score = int(file.split(".")[0]) / 100000

        submission_data_frame = pd.read_csv(os.path.join(submissions_dir, file))
        label_score = fbeta_score(
            labels_data_frame["anomaly"].values, submission_data_frame["anomaly"].values, beta=1.75
        )

        score_abs_difference = np.abs(kaggle_score - label_score)

        results.append(
            {
                "submission_idx": submission_idx,
                "kaggle_score": kaggle_score,
                "label_score": label_score,
                "score_abs_difference": score_abs_difference,
            }
        )

    results = pd.DataFrame.from_dict(results)

    fig, ax = plt.subplots(figsize=(16, 9))
    sns.scatterplot(data=results, x="kaggle_score", y="score_abs_difference", ax=ax)
    plt.show()


    pass


if __name__ == "__main__":
    fire.Fire(main)
