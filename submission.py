import sys
import os

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from classes.feature_factory import FeatureFactory

# Trainingsdaten & Testdaten laden
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

print("A")
# Beispiel für die Verwendung FeatureFactory
factory = FeatureFactory(train_df, test_df)

n_sfa_components = 4
sfa_control_areas = [1, 2]
sfa_degree = 1

selected_sfa_features = ["Demand", "correction", "correctionEcho", "FRCE", "LFCInput", 'aFRRactivation', 'aFRRrequest',]

for sfa_control_area in sfa_control_areas:
       factory.add_SFA(n_sfa_components, selected_sfa_features, poly_degree=sfa_degree, control_area=sfa_control_area)
factory.add_time_features()
factory.add_rolling_features(window_size=3)
factory.add_ratio_and_diff_features()
factory.add_aFRR_activation_request_ratio()
factory.add_FRCE_LFCInput_difference()
factory.add_participation_state()
factory.add_demand_FRCE_interaction()
factory.add_corrected_demand_feature()

# Features
# New features beginn with 'day', ...
features = ['Demand', 'correction', 'correctedDemand',
       'FRCE', 'controlBandPos', 'controlBandNeg', 'LFCInput',
       'aFRRactivation', 'aFRRrequest', 'participationCMO', 'participationIN',
       'correctionEcho', 'BandLimitedCorrectedDemand', 'controlArea', 'hour',
       'day', 'weekday', 'month', 'Demand_RollingMean', 'Demand_RollingStd',
       'Demand_CorrectedDemand_Ratio', 'Demand_CorrectedDemand_Diff',
       'aFRR_Activation_Request_Ratio', 'FRCE_LFCInput_Diff',
       'Participation_State', 'Demand_FRCE_Interaction']#, 'corrected_demand_diff']


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
model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
model.fit(X_train_normalized)

# Anomalien auf Testdaten vorhersagen und anzeigen
test_df['anomaly'] = model.predict(X_test_normalized)
print(test_df[['Datum_Uhrzeit_CET', 'Demand', 'correctedDemand', 'anomaly']].head())

# Konvertiere Anomalie-Vorhersagen: -1 (Anomalie) wird zu 1 und 1 (normal) wird zu 0
test_df['anomaly'] = test_df['anomaly'].apply(lambda x: 1 if x == -1 else 0)
submission_df = test_df[['id', 'anomaly']]
submission_df.to_csv('submission.csv', index=False)

