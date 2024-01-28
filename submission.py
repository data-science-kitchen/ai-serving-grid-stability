import pandas as pd
import mlflow
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

from classes.feature_factory import FeatureFactory

mlflow.set_tracking_uri("https://mlflow.preislers.de")
mlflow.set_experiment("ai-serving-grid-stability")

def train_intrafeature(train_data, val_data, group, regression_class, hyperparameters={}, metrics=[r2_score]):
    """
    Takes a group of features and trains regression for each feature, based on all remaining features. 
    Returns classifiers as dict.
    """
    for feature_name in group:
        assert feature_name in train_data.columns

    model_dict = {}
    for target_name in group:
        remaining_feature_names = [feature_name for feature_name in group if not feature_name == target_name]
        print(f"Training model: '{target_name}' from {remaining_feature_names} ")
        model = regression_class(**hyperparameters)
        train_subset = train_data[remaining_feature_names]
        val_subset = val_data[remaining_feature_names]
        train_target = train_data[target_name]
        val_target = val_data[target_name]
        model.fit(train_subset, train_target)
        train_pred = model.predict(train_subset)
        val_pred = model.predict(val_subset)
        for metric in metrics:
            train_score = metric(train_target, train_pred)
            val_score = metric(val_target, val_pred)
            print(f"\t{metric}")
            print(f"\t\ttrain:\t{train_score}")
            print(f"\t\tval:\t{val_score}")
        print("\t...done.")
        model_dict[target_name] = (model, remaining_feature_names)
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


def fill_anomalies(df, window_size=4, threshold=2, loops=2):
    count = 0
    for l in range(loops):
        for index, row in df.iterrows():
            if row["anomaly"] == 0:
                start_index = max(index - window_size, 0)
                end_index = min(index + window_size + 1, len(df))
                window = df["anomaly"][start_index:end_index]

                # Prüfe, ob mindestens eine '1' im Bereich vor und nach der '0' ist
                if 1 in df["anomaly"][start_index:index].values and 1 in df["anomaly"][index + 1: end_index].values:
                    window = df["anomaly"][start_index:end_index]
                    if window.sum() >= threshold:
                        df.at[index, "anomaly"] = 1
                        count += 1
    print("Gefüllt:", count)

    return df


def remove_anomalies(df, window_size=5, threshold=1):
    count = 0
    for index, row in df.iterrows():
        if row["anomaly"] == 1:
            start_index = max(index - window_size, 0)
            end_index = min(index + window_size + 1, len(df))

            if 0 in df["anomaly"][start_index:index].values and 0 in df["anomaly"][index + 1: end_index].values:
                window = df["anomaly"][start_index:end_index]
                if window.sum() <= threshold:
                    df.at[index, "anomaly"] = 0
                    count += 1
    print("Entfernt:", count)

    return df


with mlflow.start_run():

    # Trainingsdaten & Testdaten laden
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    # Beispiel für die Verwendung FeatureFactory
    factory = FeatureFactory(train_df, test_df)

    n_sfa_components = 4
    sfa_control_areas = [1, 2]
    sfa_degree = 2
    mlflow.log_param("n_sfa_components", n_sfa_components)
    mlflow.log_param("sfa_control_areas", ",".join(
        map(str, sfa_control_areas)))
    mlflow.log_param("sfa_degree", sfa_degree)

    factory.add_corrected_demand_feature()

    selected_sfa_features = ["Demand", "correction", "correctionEcho",
                             "FRCE", "LFCInput", "aFRRactivation", "aFRRrequest"]
    mlflow.log_param("selected_sfa_features", ",".join(
        map(str, selected_sfa_features)))

    for sfa_control_area in sfa_control_areas:
        factory.add_SFA(
            n_sfa_components,
            selected_sfa_features,
            poly_degree=sfa_degree,
            control_area=sfa_control_area,
            batch_size=100,
            cascade_length=1,
        )

    factory.add_time_features()
    factory.add_rolling_features(window_size=3)
    # factory.add_rolling_features_by_control_area(window_size=3)
    factory.add_ratio_and_diff_features()
    factory.add_aFRR_activation_request_ratio()
    factory.add_FRCE_LFCInput_difference()
    factory.add_participation_state()
    factory.add_demand_FRCE_interaction()

    # Features
    # New features beginn with 'day', ...
    features = [
        "Demand",
        "correction",
        "correctedDemand",
        "FRCE",
        "controlBandPos",
        "controlBandNeg",
        "LFCInput",
        "aFRRactivation",
        "aFRRrequest",
        "participationCMO",
        "participationIN",
        "correctionEcho",
        "BandLimitedCorrectedDemand",
        "controlArea",
        "hour",
        "day",
        "weekday",
        "month",
        # "daylight",
        # "workday",
        # "Demand_RollingMean",
        # "Demand_RollingStd",
        "Demand_CorrectedDemand_Ratio",
        "Demand_CorrectedDemand_Diff",
        "aFRR_Activation_Request_Ratio",
        "FRCE_LFCInput_Diff",
        "Participation_State",
        "Demand_FRCE_Interaction",
    ]  # , 'corrected_demand_diff']

    mlflow.log_param("features", ",".join(map(str, features)))

    for sfa_control_area in sfa_control_areas:
        sfa_features = [
            f"sfa{c}_{sfa_control_area}" for c in range(n_sfa_components)]
    features = features + sfa_features

    X_train = factory.train_data[features]
    X_test = factory.test_data[features]

    # Merlins intra-feature dingens
    hyperparameters = {"max_depth": 10}
    regression_class = DecisionTreeRegressor
    group = ["correction", "correctionEcho", "FRCE", "aFRRactivation"]#"correction", "correctedDemand", "Demand"]
    model_dict = train_intrafeature(train_df, 
                                    test_df,
                                    group=group, 
                                    regression_class=regression_class, 
                                    hyperparameters=hyperparameters,
                                    metrics=[r2_score])
    target_name = "aFRRactivation"
    model, remaining = model_dict[target_name]
    target, pred = run_intrafeature_model(train_df, model, target_name, remaining)
    X_train['residual'] = target-pred
    target, pred = run_intrafeature_model(test_df, model, target_name, remaining)
    X_test['residual'] = target-pred

    # Scaler
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)


    

    # Isolation Forest Modell initialisieren und trainieren
    model = IsolationForest(
        n_estimators=32, contamination="auto", random_state=42)
    for key, value in model.get_params().items():
        mlflow.log_param(key, value)
    model.fit(X_train_normalized)

    # Anomalien auf Testdaten vorhersagen und anzeigen
    test_df["anomaly"] = model.predict(X_test_normalized)
    # Konvertiere Anomalie-Vorhersagen: -1 (Anomalie) wird zu 1 und 1 (normal) wird zu 0
    test_df["anomaly"] = test_df["anomaly"].apply(
        lambda x: 1 if x == -1 else 0)

    


    df_filled = fill_anomalies(
        test_df.copy(), window_size=10, threshold=4, loops=2)
    mlflow.log_param('fill_window_size', 10)
    mlflow.log_param('fill_threshold', 4)
    mlflow.log_param('fill_loops', 2)
    submission_df = remove_anomalies(
        df_filled.copy(), window_size=5, threshold=4)
    mlflow.log_param('remove_window_size', 5)
    mlflow.log_param('remove_threshold', 4)

    submission_df = submission_df[["id", "anomaly"]]
    submission_df.loc[~test_df[["participationIN", "participationCMO"]].all(axis=1), "anomaly"] = 0
    submission_df.to_csv("submission.csv", index=False)
