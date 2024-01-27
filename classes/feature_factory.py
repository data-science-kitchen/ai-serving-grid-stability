import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
import sksfa
from typing import List, Tuple, Dict
from statsmodels.tsa.arima.model import ARIMA




# Ideas so far:
# a) activation, request Signal:    derivative, to detect opposite direction (asynchronous)
# b) correction, correctionError:   Freeze of at least echo signal with simple threshold (window size) and derivative
# c) correction, correctionError:   cross-correlation in window, to detect larger delay of 8 seconds
# d) FRCE, LFCInput:                According to "Anomaly 3" example that the signal FRCE gets a "harten" if signal LFCInput has a "higher"
#                                   positive peak. However, it seems only the part until FRCE falls down and intersects with LFCInput again
#                                   is counted as anomaly


def detect_freeze(dataset, signalname: str, min_duration: float, threshold: float = 0.001) -> pd.Series:
    """
    Detect periods where the signal is 'frozen'.

    :param signal: Pandas Series representing the signal.
    :param threshold: Threshold for determining if the signal is frozen.
    :param min_duration: Minimum duration (in number of data points) for the signal to be considered frozen.
    :return: Pandas Series of booleans indicating frozen periods.
    """
    diff = dataset[signalname].diff().abs()
    below_threshold = diff < threshold

    # Initialize a series to store the results
    frozen_periods = pd.Series(False, index=dataset.index)

    # Check for periods where the signal remains below the threshold for at least 'min_duration'
    frozen_endpoints = below_threshold.rolling(window=min_duration).sum() >= min_duration
    for endpoint_index in frozen_endpoints[frozen_endpoints == True].index:
        frozen_periods[endpoint_index - min_duration + 1 : endpoint_index + 1] = True
    return frozen_periods


class FeatureFactory:
    """
    A class to perform feature engineering on train and test data
    for AI Serving Grid Stability.

    Attributes:
        train_data (pd.DataFrame): The training dataset.
        test_data (pd.DataFrame): The testing dataset.

    Comments:
        - I know that the functions could be more efficients, when we
        just loop for once, but it is not for a production system,
        so give a fuck. :D
        - It could be that some features do not make sense, so control them.
        - I know that is not the best variant replace NaNs and Infs with
        mean, but the dataset is so huge that it most like won't have an impact.
        (Hopefully)
    """

    def __init__(self, train_data: pd.DataFrame = None, test_data: pd.DataFrame = None):
        """
        Initialize the FeatureFactory with training and testing data.

        Args:
            train_data (pd.DataFrame): The training dataset.
            test_data (pd.DataFrame): The testing dataset.
        """
        self.train_data = train_data
        self.test_data = test_data
        self._preprocess()

    def _preprocess(self):
        """Convert 'Datum_Uhrzeit_CET' column to datetime in both datasets."""
        if self.train_data is not None:
            self.train_data["Datum_Uhrzeit_CET"] = pd.to_datetime(self.train_data["Datum_Uhrzeit_CET"])
        if self.test_data is not None:
            self.test_data["Datum_Uhrzeit_CET"] = pd.to_datetime(self.test_data["Datum_Uhrzeit_CET"])

    def add_time_features(self):
        """
        Add time-based features (hour, day, weekday, month) to both datasets.
        """
        for data in [self.train_data, self.test_data]:
            if data is None:
                continue
            data["hour"] = data["Datum_Uhrzeit_CET"].dt.hour
            data["day"] = data["Datum_Uhrzeit_CET"].dt.day
            data["weekday"] = data["Datum_Uhrzeit_CET"].dt.weekday
            data["month"] = data["Datum_Uhrzeit_CET"].dt.month

    def add_rolling_features(self, window_size: int = 5):
        """
        Add rolling mean and standard deviation features to 'Demand' column.

        Args:
            window_size (int): The window size for calculating rolling statistics.
        """
        for data in [self.train_data, self.test_data]:
            if data is None:
                continue
            # MEAN
            data["Demand_RollingMean"] = data["Demand"].rolling(window=window_size).mean()
            # Refill
            mean_2d = data["Demand_RollingMean"].values.reshape(-1, 1)
            imputer = SimpleImputer(strategy="mean")
            data["Demand_RollingMean"] = imputer.fit_transform(mean_2d).ravel()

            # STD
            data["Demand_RollingStd"] = data["Demand"].rolling(window=window_size).std()
            # Refill
            std_2d = data["Demand_RollingStd"].values.reshape(-1, 1)
            imputer = SimpleImputer(strategy="mean")
            data["Demand_RollingStd"] = imputer.fit_transform(std_2d).ravel()
    
    def add_rolling_features_by_control_area(self, window_size: int = 5):
        """
        Add rolling mean and standard deviation features to 'Demand' column,
        grouped by 'controlArea', with group-wise imputation.

        Args:
            window_size (int): The window size for calculating rolling statistics.
        """
        for data in [self.train_data, self.test_data]:
            if data is None:
                continue
            # Bereite Spalten für die rollierenden Features vor
            data["Demand_RollingMean"] = pd.NA
            data["Demand_RollingStd"] = pd.NA

            # Gruppiere nach 'controlArea'
            for name, group in data.groupby('controlArea'):
                # Berechne rollierenden Mittelwert und Standardabweichung
                group['Demand_RollingMean'] = group['Demand'].rolling(window=window_size).mean()
                group['Demand_RollingStd'] = group['Demand'].rolling(window=window_size).std()

                # Führe Imputation innerhalb jeder Gruppe durch
                imputer = SimpleImputer(strategy="mean")
                group['Demand_RollingMean'] = imputer.fit_transform(group['Demand_RollingMean'].values.reshape(-1, 1)).ravel()
                group['Demand_RollingStd'] = imputer.fit_transform(group['Demand_RollingStd'].values.reshape(-1, 1)).ravel()

                # Aktualisiere den Haupt-DataFrame
                data.loc[group.index, 'Demand_RollingMean'] = group['Demand_RollingMean']
                data.loc[group.index, 'Demand_RollingStd'] = group['Demand_RollingStd']



    def add_ratio_and_diff_features(self):
        """
        Add ratio and difference features between 'Demand' and 'correctedDemand'.
        """
        for data in [self.train_data, self.test_data]:
            if data is None:
                continue
            # MEAN
            data["Demand_CorrectedDemand_Ratio"] = data["Demand"] / data["correctedDemand"]
            # Refill
            data["Demand_CorrectedDemand_Ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)
            mean_2d = data["Demand_CorrectedDemand_Ratio"].values.reshape(-1, 1)
            imputer = SimpleImputer(strategy="mean")
            data["Demand_CorrectedDemand_Ratio"] = imputer.fit_transform(mean_2d).ravel()

            # STD
            data["Demand_CorrectedDemand_Diff"] = data["Demand"] - data["correctedDemand"]

    def add_aFRR_activation_request_ratio(self):
        """
        Add a new feature representing the ratio of aFRR activation to aFRR request.
        """
        for data in [self.train_data, self.test_data]:
            if data is None:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                data["aFRR_Activation_Request_Ratio"] = data["aFRRactivation"] / data["aFRRrequest"]
                # Replace infinity with NaN
                data["aFRR_Activation_Request_Ratio"].replace([np.inf, -np.inf], np.nan, inplace=True)

            # Refill NaN values with the mean of the column
            ratio_2d = data["aFRR_Activation_Request_Ratio"].values.reshape(-1, 1)
            imputer = SimpleImputer(strategy="mean")
            data["aFRR_Activation_Request_Ratio"] = imputer.fit_transform(ratio_2d).ravel()

    def add_FRCE_LFCInput_difference(self):
        """
        Add a new feature representing the difference between FRCE and LFCInput.
        """
        for data in [self.train_data, self.test_data]:
            if data is None:
                continue
            data["FRCE_LFCInput_Diff"] = data["FRCE"] - data["LFCInput"]

    def add_corrected_demand_feature(self):
        """
        Feature that checks if the calculated demand
        """
        for data in [self.train_data, self.test_data]:
            if data is None:
                continue
            corrected_demand_calculated = data["Demand"] + data["correction"]
            corrected_demand_difference = np.abs(data["correctedDemand"] - corrected_demand_calculated)

            data["corrected_demand_diff"] = corrected_demand_difference

    # TODO: DO NOT USE
    # def add_FRCE_LFCInput_mavg_difference(self):
    #     """
    #     """
    #     for data in [self.train_data, self.test_data]:
    #         data.sort_index(inplace=True)
    #         results = []
    #         for control_area in [1, 2]:
    #             area_subset = data[data["controlArea"] == control_area]
    #             aFRRrequest_mavg = area_subset['aFRRrequest'].rolling(window=20).mean()
    #             aFRRactivation_mavg = area_subset['aFRRactivation'].rolling(window=15).mean()
    #             results.append(np.abs(aFRRrequest_mavg - aFRRactivation_mavg))
    #         data['FRCE_LFCInput_mavg_difference'] = pd.concat(results).sort_index()

    def add_aFRRactivation_mavg_diff(self):
        """
        Usually the derivative has a high peak when the asynchronous pattern appears
        """
        for data in [self.train_data, self.test_data]:
            if data is None:
                continue
            data.sort_index(inplace=True)
            results = []
            for control_area in [1, 2]:
                area_subset = data[data["controlArea"] == control_area]
                aFRRactivation_mavg = area_subset["aFRRactivation"].rolling(window=5).mean()
                results.append(aFRRactivation_mavg.diff(2))
            data["aFRRactivation_mavg_diff"] = pd.concat(results).sort_index()

    def add_correctionEcho_freeze(self):
        """
        Detect if the correctionEcho signal freezes for a certain time
        """
        for data in [self.train_data, self.test_data]:
            if data is None:
                continue
            data.sort_index(inplace=True)
            results = []
            for control_area in [1, 2]:
                area_subset = data[data["controlArea"] == control_area]
                results.append(detect_freeze(area_subset, "correctionEcho", min_duration=10))
            data["correctionEcho_freeze"] = pd.concat(results).sort_index()

    def add_participation_state(self):
        """
        Add a new feature representing the participation state, calculated from
        participationCMO and participationIN.
        """
        for data in [self.train_data, self.test_data]:
            if data is None:
                continue
            data["Participation_State"] = data["participationCMO"].astype(int) * 2 + data["participationIN"].astype(int)

    def add_demand_FRCE_interaction(self):
        """
        Add a new feature representing the interaction between Demand and FRCE.
        """
        for data in [self.train_data, self.test_data]:
            if data is None:
                continue
            data["Demand_FRCE_Interaction"] = data["Demand"] * data["FRCE"]

    def add_SFA(self, n_components, sfa_features, poly_degree=2, control_area=0, batch_size=100, cascade_length=1, trained_pipeline=None):
        """
        Add a new feature representing the interaction between Demand and FRCE.
        """
        if self.train_data is not None:
            numeric_train_ca = (
                self.train_data[self.train_data.controlArea == control_area]
                .drop("Datum_Uhrzeit_CET", axis=1)[sfa_features]
                .to_numpy()
            )
            numeric_train = self.train_data.drop("Datum_Uhrzeit_CET", axis=1)[sfa_features].to_numpy()
        if self.test_data is not None:
            numeric_test = self.test_data.drop("Datum_Uhrzeit_CET", axis=1)[sfa_features].to_numpy()

        if trained_pipeline is not None:
            processing_pipeline = trained_pipeline
        else:   
            pf = PolynomialFeatures(degree=poly_degree)
            sfa = sksfa.SFA(n_components, batch_size=batch_size)

            numeric_train_ca = pf.fit_transform(numeric_train_ca)
            numeric_train_ca = sfa.fit_transform(numeric_train_ca)
            processing_pipeline = [pf, sfa]

            for cascade_idx in range(cascade_length - 1):
                print(f"\tSFA Cascade level {cascade_idx + 2}, degree {poly_degree}")
                pf = PolynomialFeatures(degree=poly_degree)
                numeric_train_ca = pf.fit_transform(numeric_train_ca)
                processing_pipeline.append(pf)
                sfa = sksfa.SFA(n_components, batch_size=batch_size)
                numeric_train_ca = sfa.fit_transform(numeric_train_ca)
                processing_pipeline.append(sfa)

        for processing_step in processing_pipeline:
            if self.train_data is not None:
                numeric_train = processing_step.transform(numeric_train)
            if self.test_data is not None:
                numeric_test = processing_step.transform(numeric_test)

        for component_index in range(n_components):
            if self.train_data is not None:
                self.train_data[f"sfa{component_index}_{control_area}"] = numeric_train[:, component_index]
            if self.test_data is not None:
                self.test_data[f"sfa{component_index}_{control_area}"] = numeric_test[:, component_index]

        return processing_pipeline

    def add_lag_features(self, features_with_lags: List[Tuple[str, int]]) -> List[str]:
        """
        Adds lagged features to the dataset segmented by 'controlArea' and returns a list 
        of the new feature names. The modified datasets are stored back in self.train_data 
        and self.test_data.

        Parameters:
        features_with_lags (List[Tuple[str, int]]): A list of tuples, where each tuple 
                                                    contains a feature name and the number 
                                                    of lags to be created for that feature.

        Returns:
        List[str]: A list of the newly created lag feature names.
        """
        new_feature_names = []

        for df_name in ['train_data', 'test_data']:
            df = getattr(self, df_name)
            # Segment the data by 'controlArea'
            segmented_dfs = [group for _, group in df.groupby('controlArea')]
            processed_segments = []
            for segment_df in segmented_dfs:
                for feature, n_lags in features_with_lags:
                    for lag in range(1, n_lags + 1):
                        lag_feature_name = f"{feature}_lag_{lag}"
                        if lag_feature_name not in new_feature_names:
                            new_feature_names.append(lag_feature_name)

                        segment_df[lag_feature_name] = segment_df[feature].shift(lag)
                        # Fill missing values at the start and end of the series
                        segment_df[lag_feature_name].fillna(method='bfill', inplace=True)
                        segment_df[lag_feature_name].fillna(method='ffill', inplace=True)
                processed_segments.append(segment_df)

            # Reassemble the segmented dataframes
            setattr(self, df_name, pd.concat(processed_segments))

        return new_feature_names
    
    def add_arima_resid_features(self, features: List[str], arima_orders: Dict[str, Tuple[int, int, int]]) -> List[str]:
        """
        Applies ARIMA models to specified features within each controlArea and adds the residuals as new features.

        Parameters:
        features (List[str]): List of feature names to apply ARIMA models to.
        arima_orders (Dict[str, Tuple[int, int, int]]): Dictionary mapping feature names 
                                                        to their ARIMA order (p, d, q).

        Returns:
        List[str]: A list of the newly created residual feature names.
        """
        new_feature_names = []

        for df_name in ['train_data', 'test_data']:
            df = getattr(self, df_name)
            for feature in features:
                resid_feature_name = f"{feature}_resid"
                df[resid_feature_name] = None
                new_feature_names.append(resid_feature_name)

                for control_area, segment_df in df.groupby('controlArea'):
                    order = arima_orders.get(feature, (1, 0, 1))  # Default ARIMA order
                    model = ARIMA(segment_df[feature], order=order)
                    model_fit = model.fit()
                    
                    df.loc[segment_df.index, resid_feature_name] = model_fit.resid

            # Ensure the DataFrame is sorted by 'id'
            df = df.sort_values(by='id')
            setattr(self, df_name, df)

        return new_feature_names