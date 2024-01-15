import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


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
