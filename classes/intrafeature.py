import pandas as pd
from sklearn.metrics import r2_score
import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt

import tqdm


def train_intrafeature(train_data, val_data, group, regression_class, hyperparameters={}, metric=r2_score):
    """
    Takes a group of features and trains regression for each feature, based on all remaining features. 
    Returns classifiers as dict.
    """
    for feature_name in group:
        assert feature_name in train_data.columns

    model_dict = {}
    for target_name in group:
        if target_name.startswith("sfa"):
            continue
        remaining_feature_names = [feature_name for feature_name in group if not feature_name == target_name]
        model = regression_class(**hyperparameters)
        train_subset = train_data[remaining_feature_names]
        val_subset = val_data[remaining_feature_names]
        train_target = train_data[target_name]
        val_target = val_data[target_name]
        model.fit(train_subset, train_target)
        train_pred = model.predict(train_subset)
        val_pred = model.predict(val_subset)
        train_score = metric(train_target, train_pred)
        val_score = metric(val_target, val_pred)
        model_dict[target_name] = (model, remaining_feature_names, train_score, val_score)
    return model_dict

def run_intrafeature_model(data, model, target_name, remaining_feature_names):
    subset = data[remaining_feature_names]
    target = data[target_name]
    pred = model.predict(subset)
    return target, pred

def hypothesize_anomalies(target, prediction, quantile=0.15):
    residuals = target - prediction
    lower, upper = residuals.quantile(quantile), residuals.quantile(1 - quantile)
    anomalies = residuals.apply(lambda x: 0 if lower < x < upper else 1)
    return anomalies

def count_anomagrams(anomalies):
    """
    Determine counts of lengths of anomalous blocks
    """
    counts = []
    counter = 0
    for anomaly in anomalies:
        if anomaly == 1:
            counter += 1
        if anomaly == 0 and counter > 0:
            counts.append(counter)
            counter = 0
    return np.array(counts)