import fire
import os

from os import PathLike

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from skorch import NeuralNet
from skorch.callbacks import Checkpoint
from src.dataset import GridStabilityDataset
from src.model import GRUPredictor
from torch import nn
from typing import Union

FEATURES = [
    "Demand",
    "correction",
    "correctedDemand",
    "FRCE",
    "LFCInput",
    "aFRRactivation",
    "aFRRrequest",
    "correctionEcho",
    "BandLimitedCorrectedDemand",
]

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
    log_dir: Union[str, PathLike] = "logs",
    threshold_quantile: float = 0.75,
) -> None:
    sns.set_style("whitegrid")

    dataset_test = GridStabilityDataset(data_dir, split="test")

    for target_feature in FEATURES:
        if os.path.isfile(os.path.join(model_dir, target_feature, "model.pkl")):
            log_path = os.path.join(model_dir, target_feature, log_dir)
            os.makedirs(log_path, exist_ok=True)

            dataset_test.set_target_feature(target_feature)

            checkpoint = Checkpoint(
                monitor="valid_loss_best",
                dirname=os.path.join(model_dir, target_feature),
                f_pickle="model.pkl",
                load_best=True,
            )

            model = NeuralNet(GRUPredictor, criterion=nn.MSELoss())
            model.initialize()
            model.load_params(checkpoint=checkpoint)

            num_segments = dataset_test.__len__()
            num_rows = int(np.ceil(num_segments / 2))

            fig, axs = plt.subplots(nrows=num_rows, ncols=2, figsize=(16, 9))

            targets_all = []
            predictions_all = []

            for idx in range(num_segments):
                if idx < num_rows:
                    row_idx = idx
                    col_idx = 0
                else:
                    row_idx = idx - num_rows
                    col_idx = 1

                features, targets = dataset_test.__getitem__(idx)
                target_df = pd.DataFrame.from_dict({"values": targets.ravel(), "signal": np.asarray(1800 * ["target"])})

                predictions = model.predict(features)
                predictions_df = pd.DataFrame.from_dict({"values": predictions.ravel(), "signal": np.asarray(1800 * ["predictions"])})

                targets_all.append(targets)
                predictions_all.append(predictions)

                result_df = pd.concat([target_df, predictions_df]).reset_index()

                sns.lineplot(data=result_df, x="index", y="values", hue="signal", ax=axs[row_idx, col_idx])
                axs[row_idx, col_idx].set_xlim([0, 1800])
                axs[row_idx, col_idx].set_xlabel("")
                axs[row_idx, col_idx].set_ylabel("")
                axs[row_idx, col_idx].get_legend().remove()

            plt.suptitle(f"Target Feature: {target_feature}")
            plt.savefig(os.path.join(log_path, target_feature + "_signals.png"))
            plt.close(fig)

            targets_all = np.asarray(targets_all).reshape(-1)
            predictions_all = np.asarray(predictions_all).reshape(-1)
            target_prediction_error = np.log(np.power(targets_all - predictions_all, 2))

            threshold = np.quantile(target_prediction_error, q=threshold_quantile)

            target_prediction_error_below = target_prediction_error[target_prediction_error <= threshold]
            target_prediction_error_above = target_prediction_error[target_prediction_error > threshold]

            fig, ax = plt.subplots(figsize=(16, 9))
            sns.histplot(x=target_prediction_error_below, binwidth=0.1, stat="frequency", ax=ax)
            sns.histplot(x=target_prediction_error_above, binwidth=0.1, stat="frequency", ax=ax)

            plt.suptitle(f"Target Feature: {target_feature}, Quantile Threshold: {threshold_quantile}")
            plt.savefig(os.path.join(log_path, target_feature + "_hist.png"))
            plt.close(fig)


if __name__ == "__main__":
    fire.Fire(main)
