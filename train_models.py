import fire
import os
import pickle

from os import PathLike

import numpy as np
import torch.optim
from skorch import NeuralNet
from skorch.callbacks import Checkpoint, EarlyStopping
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


def main(
    data_dir: Union[str, PathLike],
    model_dir: Union[str, PathLike] = os.path.join("data", "models"),
    hidden_dim: int = 16,
    dropout: float = 0.,
    max_epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 0.01
) -> None:
    np.random.seed(42)
    torch.manual_seed(42)

    os.makedirs(model_dir, exist_ok=True)

    dataset_train = GridStabilityDataset(data_dir, split="train")
    dataset_test = GridStabilityDataset(data_dir, split="test")

    for target_feature in FEATURES:
        if not os.path.isfile(os.path.join(model_dir, target_feature, "model.pkl")):
            dataset_train.set_target_feature(target_feature)
            dataset_test.set_target_feature(target_feature)

            checkpoint = Checkpoint(
                monitor="valid_loss_best",
                dirname=os.path.join(model_dir, target_feature),
                f_pickle="model.pkl",
                load_best=True
            )
            early_stopping = EarlyStopping(load_best=True)

            model = NeuralNet(
                GRUPredictor,
                module__hidden_dim=hidden_dim,
                module__dropout=dropout,
                max_epochs=max_epochs,
                batch_size=batch_size,
                criterion=nn.MSELoss(),
                lr=learning_rate,
                optimizer=torch.optim.AdamW,
                iterator_train__shuffle=True,
                callbacks=[checkpoint, early_stopping],
            )
            model.fit(dataset_train)

            # with open(os.path.join(model_dir, target_feature + ".pkl"), "wb") as file:
            #     pickle.dump(model, file)


if __name__ == "__main__":
    fire.Fire(main)
