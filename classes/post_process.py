class PostProcess:
    """
    The PostProcess class provides methods to manipulate and adjust anomalies within a dataset.
    It includes methods to fill anomalies based on certain conditions and remove anomalies
    if they do not meet a specified threshold within a given window size.
    """

    @staticmethod
    def fill_anomalies(df, window_size=4, threshold=2, loops=2):
        """
        Fills anomalies in the dataset based on a sliding window approach. An anomaly is filled
        if there are a sufficient number of anomalies within the window size around it.

        Parameters:
        - df: Pandas DataFrame with a column named 'anomaly' indicating anomaly status (0 or 1).
        - window_size: The size of the window to check around each anomaly. Default is 4.
        - threshold: The minimum number of anomalies required within the window to fill the current anomaly. Default is 2.
        - loops: The number of times to repeat the process over the dataset. Default is 2.

        Returns:
        - The modified DataFrame with certain anomalies filled.
        """
        count = 0
        for l in range(loops):
            for index, row in df.iterrows():
                if row["anomaly"] == 0:
                    start_index = max(index - window_size, 0)
                    end_index = min(index + window_size + 1, len(df))
                    window = df["anomaly"][start_index:end_index]

                    if 1 in df["anomaly"][start_index:index].values and 1 in df["anomaly"][index + 1: end_index].values:
                        window = df["anomaly"][start_index:end_index]
                        if window.sum() >= threshold:
                            df.at[index, "anomaly"] = 1
                            count += 1
        print("Filled:", count)

        return df

    @staticmethod
    def remove_anomalies(df, window_size=5, threshold=1):
        """
        Removes anomalies in the dataset if they are isolated within a given window size.
        An anomaly is removed if the sum of anomalies within the window does not exceed
        a specified threshold.

        Parameters:
        - df: Pandas DataFrame with a column named 'anomaly' indicating anomaly status (0 or 1).
        - window_size: The size of the window to check around each anomaly. Default is 5.
        - threshold: The maximum number of anomalies allowed within the window to keep the current anomaly. Default is 1.

        Returns:
        - The modified DataFrame with certain anomalies removed.
        """
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
        print("Removed:", count)

        return df
