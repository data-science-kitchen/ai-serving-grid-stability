import fire
import os

from os import PathLike

import numpy as np
import pandas as pd
from skorch import NeuralNet
from skorch.callbacks import Checkpoint
from src.dataset import GridStabilityDataset
from src.model import GRUPredictor
from torch import nn
from typing import Union


# DETECTION_THRESHOLDS = {
#     "Demand": 0.,
#     "correction": 0.,
#     "correctedDemand": 0.,
#     "FRCE": -9.,
#     "LFCInput": 0.,
#     "aFRRactivation": 2.5,
#     "aFRRrequest": -3.,
#     "correctionEcho": -1.,
#     "BandLimitedCorrectedDemand": -2.,
# }

DETECTION_THRESHOLDS = {
    "Demand": -8.,
    "correction": -8.,
    "correctedDemand": -8.,
    "FRCE": -9.,
    "LFCInput": -2.5,
    "aFRRactivation": -4.,
    "aFRRrequest": -6.,
    "correctionEcho": -6.,
    "BandLimitedCorrectedDemand": -6.,
}


def main(
    data_dir: Union[str, PathLike],
    model_dir: Union[str, PathLike] = os.path.join("data", "models"),
) -> None:
    dataset_test = GridStabilityDataset(data_dir, split="test")

    num_segments = dataset_test.__len__()
    submission = []

    for idx in range(num_segments):
        is_anomaly = np.asarray(1800 * [0]).ravel()

        for target_feature in DETECTION_THRESHOLDS.keys():
            if not os.path.isfile(os.path.join(model_dir, target_feature, "model.pkl")):
                raise FileNotFoundError("No model file found!")

            dataset_test.set_target_feature(target_feature)
            features, targets = dataset_test.__getitem__(idx)

            checkpoint = Checkpoint(
                monitor="valid_loss_best",
                dirname=os.path.join(model_dir, target_feature),
                f_pickle="model.pkl",
                load_best=True,
            )

            model = NeuralNet(GRUPredictor, criterion=nn.MSELoss())
            model.initialize()
            model.load_params(checkpoint=checkpoint)

            predictions = model.predict(features)
            target_prediction_error = np.log(np.power(targets - predictions, 2)).ravel()
            is_anomaly = is_anomaly + (target_prediction_error > DETECTION_THRESHOLDS[target_feature])

        submission.append(is_anomaly > 2)

    submission = pd.DataFrame.from_dict({"anomaly": np.concatenate(submission).astype(int).tolist()}).reset_index()
    submission = submission.rename(columns={"index": "id"})
    test_df = pd.read_csv("data/test.csv")
    submission.loc[~test_df[["participationIN", "participationCMO"]].all(axis=1), "anomaly"] = 0
    print(f"Percentage anomalies detected (submission): {submission['anomaly'].mean() * 100:.2f}%")

    best_submission = pd.read_csv(os.path.join("data", "submissions", "72563.csv"))
    print(f"Percentage anomalies detected (best submission): {best_submission['anomaly'].mean() * 100:.2f}%")

    overlap_with_best_submission = (submission["anomaly"] == best_submission["anomaly"]).mean()
    print(f"Overlap with best submission: {overlap_with_best_submission * 100:.2f}%")

    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    fire.Fire(main)
