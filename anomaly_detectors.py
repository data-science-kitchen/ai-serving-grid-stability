import pandas as pd

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