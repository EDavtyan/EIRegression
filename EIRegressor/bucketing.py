import pandas as pd
import numpy as np


def bucketing(data, bins, type):
    """
    Bucketing the data into bins

    :param data: array to bucket
    :param bins: number of bins
    :param type: type of bucketing('ranged'/'quantile'/'max_score')
    :return: (array of buckets, bins)
    """
    if type == "ranged":
        return pd.cut(data, bins=bins,
                      labels=False, retbins=True, duplicates="raise")
    elif type == "quantile":
        return pd.qcut(data, q=bins,
                       labels=False, retbins=True, duplicates="raise")
    elif type == "max_score":
        sorted_array = np.sort(np.array(data))
        total = sorted_array.sum()
        jump = total/bins
        count = 0
        group_number = 0
        sorted_groups = {}
        bins = []
        groups = np.zeros_like(data, dtype=np.int8)
        for i in range(len(sorted_array)):
            if count > jump*(group_number+1):
                group_number += 1
                bins += data[i]
            sorted_groups[sorted_array[i]] = group_number
            count += sorted_array[i]

        for i in range(len(groups)):
            groups[i] = sorted_groups[data[i]]
        return (groups, bins)
    else:
        print("type must be 'ranged', 'quantile' or 'max_score'")
        return ([], [])
