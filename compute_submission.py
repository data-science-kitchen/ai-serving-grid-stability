import fire
import numpy as np
import os
import pandas as pd

from os import PathLike
from sklearn.ensemble import IsolationForest
from typing import Optional, Tuple, Union


# def compute_windowed_correlation(
#     signal: pd.Series,
#     other: Optional[pd.Series] = None,
#     window_length: int = 15,
#     segment_id: Optional[pd.Series] = None,
# ) -> pd.Series:
#     r"""Computes windowed correlations, either between two signals or as an auto-correlation of a single signal.
#
#     Parameters
#     ----------
#     signal
#     window_length
#
#     Returns
#     -------
#
#     """
#
#     def correlation(
#         signal_: pd.Series, other_: Optional[pd.Series] = None, window_length_: int = 15
#     ) -> Tuple[pd.Series, pd.Series]:
#         r"""Correlation helper function with identical inputs as the main function."""
#         overlap = window_length_ - 1
#         num_windows = int(np.ceil((len(signal) - overlap) / (window_length - overlap)))
#
#         correlation_lag = []
#         correlation_max = []
#
#         for idx in range(num_windows - 1):
#             signal_window = signal_[idx : np.minimum(idx + window_length_, len(signal))]
#
#             if other_ is not None:
#                 other_window = other_[idx : np.minimum(idx + window_length_, len(signal))]
#                 c = np.correlate(signal_window, other_window, mode="same")
#                 n = np.dot(signal_window.abs(), other_window.abs())
#             else:
#                 c = np.correlate(signal_window, signal_window, mode="same")
#                 n = np.dot(signal_window.abs(), signal_window.abs())
#
#             correlation_lag.append(np.argmax(c))
#             correlation_max.append(np.max(c / n))
#
#         return pd.Series(correlation_lag), pd.Series(correlation_max)
#
#     if segment_id is not None:
#         for i in segment_id.unique():
#             signal_segment = signal[segment_id == i]
#
#             if other is not None:
#                 other_segment = other[segment_id == i]
#
#             correlation_segment = correlation(signal_segment, other_segment, window_length)
#
#     raise NotImplementedError


def compute_datetime_features(datetime_signal: pd.Series) -> pd.DataFrame:
    r"""Returns various features derived from datetime information.

    Parameters
    ----------
    datetime_signal

    Returns
    -------

    """
    datetime_signal = pd.to_datetime(datetime_signal, errors="coerce")

    result = pd.DataFrame()
    result["hour"] = datetime_signal.dt.hour
    result["dayofweek"] = datetime_signal.dt.dayofweek
    result["month"] = datetime_signal.dt.month
    result["daylight"] = ((result["hour"] >= 6) & (result["hour"] <= 18)).astype(int)
    result["workday"] = (result['dayofweek'] < 5).astype(int)

    return result


def main(data_dir: Union[str, PathLike] = "data"):
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"), low_memory=False)
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"), low_memory=False)

    submission_items = []

    for control_area in [1, 2]:
        train_df_area = train_df[train_df["controlArea"] == control_area]
        test_df_area = test_df[test_df["controlArea"] == control_area]

        model = IsolationForest(n_estimators=100, contamination="auto", random_state=42)

        predictions = pd.DataFrame()

        for data, split in zip([train_df_area, test_df_area], ["train", "test"]):
            data = data.sort_values("Datum_Uhrzeit_CET")

            datetime_features = compute_datetime_features(data["Datum_Uhrzeit_CET"])
            data = pd.concat([data, datetime_features], axis=1)

            if split == "train":
                data = data.drop(columns=["Datum_Uhrzeit_CET", "id"])
                model.fit(data)
            else:
                features = model.feature_names_in_

                predictions["id"] = data["id"]
                predictions["anomaly"] = model.predict(data[features])
                predictions["anomaly"] = predictions["anomaly"].map({1: 0, -1: 1})

        submission_items.append(predictions)

    submission_data_frame = pd.concat(submission_items)
    submission_data_frame = submission_data_frame.sort_values("id")
    submission_data_frame.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    fire.Fire(main)
