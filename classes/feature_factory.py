import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

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
        frozen_periods[endpoint_index-min_duration+1:endpoint_index+1] = True
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

    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
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
        self.train_data['Datum_Uhrzeit_CET'] = pd.to_datetime(self.train_data['Datum_Uhrzeit_CET'])
        self.test_data['Datum_Uhrzeit_CET'] = pd.to_datetime(self.test_data['Datum_Uhrzeit_CET'])

    def add_time_features(self):
        """
        Add time-based features (hour, day, weekday, month) to both datasets.
        """
        for data in [self.train_data, self.test_data]:
            data['hour'] = data['Datum_Uhrzeit_CET'].dt.hour
            data['day'] = data['Datum_Uhrzeit_CET'].dt.day
            data['weekday'] = data['Datum_Uhrzeit_CET'].dt.weekday
            data['month'] = data['Datum_Uhrzeit_CET'].dt.month

    def add_rolling_features(self, window_size: int = 5):
        """
        Add rolling mean and standard deviation features to 'Demand' column.

        Args:
            window_size (int): The window size for calculating rolling statistics.
        """
        for data in [self.train_data, self.test_data]:
            
            # MEAN
            data['Demand_RollingMean'] = data['Demand'].rolling(window=window_size).mean()
            # Refill
            mean_2d = data['Demand_RollingMean'].values.reshape(-1, 1)
            imputer = SimpleImputer(strategy='mean')
            data['Demand_RollingMean'] = imputer.fit_transform(mean_2d).ravel()

            # STD
            data['Demand_RollingStd'] = data['Demand'].rolling(window=window_size).std()
            # Refill
            std_2d = data['Demand_RollingStd'].values.reshape(-1, 1)
            imputer = SimpleImputer(strategy='mean')
            data['Demand_RollingStd'] = imputer.fit_transform(std_2d).ravel()

    def add_ratio_and_diff_features(self):
        """
        Add ratio and difference features between 'Demand' and 'correctedDemand'.
        """
        for data in [self.train_data, self.test_data]:
            # MEAN
            data['Demand_CorrectedDemand_Ratio'] = data['Demand'] / data['correctedDemand']
            # Refill
            data['Demand_CorrectedDemand_Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
            mean_2d = data['Demand_CorrectedDemand_Ratio'].values.reshape(-1, 1)
            imputer = SimpleImputer(strategy='mean')
            data['Demand_CorrectedDemand_Ratio'] = imputer.fit_transform(mean_2d).ravel()

            # STD
            data['Demand_CorrectedDemand_Diff'] = data['Demand'] - data['correctedDemand']

    def add_aFRR_activation_request_ratio(self):
        """
        Add a new feature representing the ratio of aFRR activation to aFRR request.
        """ 
        for data in [self.train_data, self.test_data]:
            with np.errstate(divide='ignore', invalid='ignore'):
                data['aFRR_Activation_Request_Ratio'] = data['aFRRactivation'] / data['aFRRrequest']
                # Replace infinity with NaN
                data['aFRR_Activation_Request_Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)

            # Refill NaN values with the mean of the column
            ratio_2d = data['aFRR_Activation_Request_Ratio'].values.reshape(-1, 1)
            imputer = SimpleImputer(strategy='mean')
            data['aFRR_Activation_Request_Ratio'] = imputer.fit_transform(ratio_2d).ravel()

    def add_FRCE_LFCInput_difference(self):
        """
        Add a new feature representing the difference between FRCE and LFCInput.
        """ 
        for data in [self.train_data, self.test_data]:
            data['FRCE_LFCInput_Diff'] = data['FRCE'] - data['LFCInput']

    def add_corrected_demand_feature(self):
        """
        Feature that checks if the calculated demand
        """
        for data in [self.train_data, self.test_data]:
            corrected_demand_calculated = data["Demand"] + data["correction"]
            corrected_demand_difference = np.abs(data["correctedDemand"] - corrected_demand_calculated)

            data['corrected_demand_diff'] = corrected_demand_difference

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
            data.sort_index(inplace=True)
            results = []
            for control_area in [1, 2]:
                area_subset = data[data["controlArea"] == control_area]
                aFRRactivation_mavg = area_subset['aFRRactivation'].rolling(window=5).mean()
                results.append(aFRRactivation_mavg.diff(2))
            data['aFRRactivation_mavg_diff'] = pd.concat(results).sort_index()

    def add_correctionEcho_freeze(self):
        """
        Detect if the correctionEcho signal freezes for a certain time
        """
        for data in [self.train_data, self.test_data]:
            data.sort_index(inplace=True)
            results = []
            for control_area in [1, 2]:
                area_subset = data[data["controlArea"] == control_area]
                results.append(detect_freeze(area_subset, 'correctionEcho', min_duration=10))
            data['correctionEcho_freeze'] = pd.concat(results).sort_index()

    def add_participation_state(self):
        """
        Add a new feature representing the participation state, calculated from 
        participationCMO and participationIN.
        """ 
        for data in [self.train_data, self.test_data]:
            data['Participation_State'] = data['participationCMO'].astype(int) * 2 + data['participationIN'].astype(int)

    def add_demand_FRCE_interaction(self):
        """
        Add a new feature representing the interaction between Demand and FRCE.
        """ 
        for data in [self.train_data, self.test_data]:
            data['Demand_FRCE_Interaction'] = data['Demand'] * data['FRCE']
