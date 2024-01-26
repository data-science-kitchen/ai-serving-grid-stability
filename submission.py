import pandas as pd
import mlflow
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from classes.feature_factory import FeatureFactory

mlflow.set_tracking_uri("https://mlflow.preislers.de")
mlflow.set_experiment("ai-serving-grid-stability")



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
