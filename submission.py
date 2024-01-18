import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from classes.feature_factory import FeatureFactory


def fill_anomalies(df, window_size=4, threshold=2, loops=2):
    count = 0
    for l in range(loops):
        for index, row in df.iterrows():
            if row["anomaly"] == 0:
                start_index = max(index - window_size, 0)
                end_index = min(index + window_size + 1, len(df))
                window = df["anomaly"][start_index:end_index]

                # Prüfe, ob mindestens eine '1' im Bereich vor und nach der '0' ist
                if 1 in df["anomaly"][start_index:index].values and 1 in df["anomaly"][index + 1 : end_index].values:
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

            if 0 in df["anomaly"][start_index:index].values and 0 in df["anomaly"][index + 1 : end_index].values:
                window = df["anomaly"][start_index:end_index]
                if window.sum() <= threshold:
                    df.at[index, "anomaly"] = 0
                    count += 1
    print("Entfernt:", count)

    return df


# Trainingsdaten & Testdaten laden
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Beispiel für die Verwendung FeatureFactory
factory = FeatureFactory(train_df, test_df)

n_sfa_components = 4
sfa_control_areas = [1, 2]
sfa_degree = 2

factory.add_corrected_demand_feature()

selected_sfa_features = ["Demand", "correction", "correctionEcho", "FRCE", "LFCInput", "aFRRactivation", "aFRRrequest"]

for sfa_control_area in sfa_control_areas:
    factory.add_SFA(
        n_sfa_components,
        selected_sfa_features,
        poly_degree=sfa_degree,
        control_area=sfa_control_area,
        batch_size=200,
        cascade_length=1,
    )

factory.add_time_features()
factory.add_rolling_features(window_size=3)
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
    "Demand_RollingMean",
    "Demand_RollingStd",
    "Demand_CorrectedDemand_Ratio",
    "Demand_CorrectedDemand_Diff",
    "aFRR_Activation_Request_Ratio",
    "FRCE_LFCInput_Diff",
    "Participation_State",
    "Demand_FRCE_Interaction",
]  # , 'corrected_demand_diff']

for sfa_control_area in sfa_control_areas:
    sfa_features = [f"sfa{c}_{sfa_control_area}" for c in range(n_sfa_components)]
    features = features + sfa_features

X_train = factory.train_data[features]
X_test = factory.test_data[features]

X_train.isna().sum()

# Scaler
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Isolation Forest Modell initialisieren und trainieren
model = IsolationForest(n_estimators=100, contamination="auto", random_state=42)
model.fit(X_train_normalized)

# Anomalien auf Testdaten vorhersagen und anzeigen
test_df["anomaly"] = model.predict(X_test_normalized)
print(test_df[["Datum_Uhrzeit_CET", "Demand", "correctedDemand", "anomaly"]].head())

# Konvertiere Anomalie-Vorhersagen: -1 (Anomalie) wird zu 1 und 1 (normal) wird zu 0
test_df["anomaly"] = test_df["anomaly"].apply(lambda x: 1 if x == -1 else 0)

submission_df = fill_anomalies(test_df.copy(), window_size=4, threshold=4, loops=2)

submission_df = submission_df[["id", "anomaly"]]
submission_df.to_csv("submission.csv", index=False)

# # Erstellen der Subplots
# parts = 38 * 2
# teil_groesse = len(submission_df) // parts
# fig, axes = plt.subplots(parts, 2, figsize=(20, 2 * parts))
#
# for i in range(parts):
#        # Auswahl des entsprechenden Teils der DataFrames
#        teil_df = test_df.iloc[i * teil_groesse:(i + 1) * teil_groesse]
#        teil_df_filled = submission_df.iloc[i * teil_groesse:(i + 1) * teil_groesse]
#
#        # Erstellen der Balkendiagramme für den aktuellen Teil
#        axes[i, 0].bar(teil_df['id'], teil_df['anomaly'], width=1.0)
#
#        axes[i, 1].bar(teil_df_filled['id'], teil_df_filled['anomaly'], width=1.0)
#
#        axes[i, 0].set_ylabel('Part: ' + str(i))
#        # Setzen der Achsenbeschriftungen für alle Spalten
#        # for j in range(3):
#        #     axes[i, j].set_xlabel('ID')
#        #     axes[i, j].set_ylabel('Anomalie (1 oder 0)')
#        #     axes[i, j].set_xlim(i * teil_groesse, (i + 1) * teil_groesse)
#
# axes[0, 0].set_title(f'Original Daten Teil {i + 1}')
# axes[0, 1].set_title(f'Gefüllte Daten Teil {i + 1}')
#
# # axes[0, 0].set_xlabel('ID')
# # axes[0, 1].set_ylabel('Anomalie (1 oder 0)')
# # axes[0, 2].set_xlim(i * teil_groesse, (i + 1) * teil_groesse)
#
# # Anpassung des Layouts
# plt.tight_layout()
#
# # Anzeigen der Plots
# plt.show()
