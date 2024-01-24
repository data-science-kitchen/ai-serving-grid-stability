import fire
import os
import pandas as pd

from os import PathLike
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sksfa import SFA
from typing import Union


NUMERIC_FEATURES = [
    # "Demand",
    # "correction",
    # "correctedDemand",
    # "FRCE",
    # "LFCInput",
    # "aFRRactivation",
    # "aFRRrequest",
    # "BandLimitedCorrectedDemand",
    "hour",
    "dayofweek",
    "month",
    "correction_squared_diff",
    "afrr_squared_diff",
    "demand_squared_diff",
    "frce_lfc_squared_diff",
]
ORDINAL_FEATURES = ["agg_categorical_features"]
SFA_FEATURES = [
    "Demand",
    "correction",
    "correctedDemand",
    "controlBandPos",
    "controlBandNeg",
    "FRCE",
    "LFCInput",
    "aFRRactivation",
    "aFRRrequest",
    "BandLimitedCorrectedDemand",
]


def compute_squared_difference(signal: pd.Series, other: pd.Series, name: str) -> pd.Series:
    r"""Computes the point-wise squared difference between two signals.

    Parameters
    ----------
    signal
    other
    name

    Returns
    -------

    """
    return pd.Series((signal - other).pow(2), name=name)


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
    result["workday"] = (result["dayofweek"] < 5).astype(int)

    return result


def main(data_dir: Union[str, PathLike] = "data"):
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"), low_memory=False)
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"), low_memory=False)

    submission_items = []

    for control_area in [1, 2]:
        train_df_area = train_df[train_df["controlArea"] == control_area]
        test_df_area = test_df[test_df["controlArea"] == control_area]

        sfa_pipeline = Pipeline(
            [
                ("scaler", RobustScaler()),
                ("sfa", SFA(n_components=8, batch_size=1024, random_state=42)),
            ]
        )

        feature_transformer = ColumnTransformer(
            [
                ("numeric_columns", RobustScaler(), NUMERIC_FEATURES),
                ("sfa_features", sfa_pipeline, SFA_FEATURES),
                (
                    "ordinal_columns",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                    ORDINAL_FEATURES,
                ),
            ],
            remainder="drop",
        )

        model = Pipeline(
            [
                ("feature_transformer", feature_transformer),
                ("estimator", IsolationForest(n_estimators=32, contamination="auto", random_state=42)),
            ]
        )

        predictions = pd.DataFrame()

        for data, split in zip([train_df_area, test_df_area], ["train", "test"]):
            data = data.sort_values("Datum_Uhrzeit_CET")

            datetime_features = compute_datetime_features(data["Datum_Uhrzeit_CET"])
            correction_squared_diff = compute_squared_difference(
                data["correction"], data["correctionEcho"], "correction_squared_diff"
            )
            afrr_squared_diff = compute_squared_difference(
                data["aFRRactivation"], data["aFRRrequest"], "afrr_squared_diff"
            )
            demand_squared_diff = compute_squared_difference(
                data["correctedDemand"], data["Demand"] + data["correction"], "demand_squared_diff"
            )
            frce_lfc_squared_diff = compute_squared_difference(data["FRCE"], data["LFCInput"], "frce_lfc_squared_diff")

            data = pd.concat(
                [
                    data,
                    datetime_features,
                    correction_squared_diff,
                    afrr_squared_diff,
                    demand_squared_diff,
                    frce_lfc_squared_diff,
                ],
                axis=1,
            )

            data["agg_categorical_features"] = (
                data["participationCMO"].astype(int).astype(str)
                + data["participationIN"].astype(int).astype(str)
                + data["daylight"].astype(str)
                + data["workday"].astype(str)
            )

            if split == "train":
                data = data.drop(columns=["controlArea", "Datum_Uhrzeit_CET", "id"])
                model.fit(data)
            else:
                features = model.feature_names_in_

                predictions["id"] = data["id"]
                predictions["anomaly"] = model.predict(data[features])
                predictions["anomaly"] = predictions["anomaly"].map({1: 0, -1: 1})

        anomalies_detected = predictions["anomaly"].sum()
        anomalies_percentage = 100 * anomalies_detected / len(predictions)
        print(f"Control Area {control_area}: {anomalies_detected} ({anomalies_percentage:.2f}%) anomalies detected.")

        submission_items.append(predictions)

    submission_data_frame = pd.concat(submission_items)
    submission_data_frame = submission_data_frame.sort_values("id")
    submission_data_frame.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    fire.Fire(main)
