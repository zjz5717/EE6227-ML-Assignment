# -*- coding: UTF-8 -*-
"""
@Project     : ML Assignment1 
@File        : Naive Bayes.py
@Author      : Super_ze
@Date        : 2024/3/14
@Description : 

Copyright © 2024 Super_ze. All Rights Reserved.
"""

import numpy as np
import scipy.io as sci


def calculate(input_data: np.ndarray) -> (np.ndarray, np.ndarray, int):
    """
    Calculate the average and standard deviation of input_data
    :param input_data: input data
    :return : average, standard deviation and number of input_data
    """
    n = input_data.shape[0]
    average = np.mean(input_data, axis=0)
    standard = np.std(input_data, axis=0)
    return average, standard, n


def test_data_group(test_data: np.ndarray, average: np.ndarray, standard: np.ndarray, p: float) -> float:
    """
    Calculate the general multivariate normal density function。
    :param test_data: test data
    :param average: mean vector
    :param standard: standard deviation matrix
    :param p: scale factor
    :return: calculation result
    """
    d = test_data.shape[0]
    sum_p = 1.0
    for i in range(d):
        sum_p *= (1 / (np.sqrt(2 * np.pi) * standard[i])) * np.exp(
            -0.5 * ((test_data[i] - average[i]) / standard[i]) ** 2)
    return sum_p * p


def main():
    # Read data from .mat files
    data_train = sci.loadmat("data_train.mat")['data_train']
    data_label = sci.loadmat("label_train.mat")['label_train']
    data_test = sci.loadmat("data_test.mat")['data_test']

    # Split the training data into two groups based on the label
    group_1 = []
    group_2 = []
    data_all_count = data_train.shape[0]
    for i in range(data_all_count):
        if data_label[i] == 1:
            group_1.append(data_train[i])
        else:
            group_2.append(data_train[i])
    group_1 = np.array(group_1)
    group_2 = np.array(group_2)

    # Calculate the average and variance of each set of data
    average_group_1, standard_group_1, n_group_1 = calculate(group_1)
    average_group_2, standard_group_2, n_group_2 = calculate(group_2)
    p1 = n_group_1 / data_all_count
    p2 = n_group_2 / data_all_count

    # Output average and variance
    print("average of group 1:", average_group_1)
    print("standard deviation of group 1:", standard_group_1)
    print("average of group 2:", average_group_2)
    print("standard deviation of group 2:", standard_group_2)

    # Group test data and save result
    res = []
    for x in data_test:
        if_group_1 = test_data_group(x, average_group_1, standard_group_1, p1)
        if_group_2 = test_data_group(x, average_group_2, standard_group_2, p2)
        if if_group_1 >= if_group_2:
            res.append(np.concatenate((x, np.array([1])), axis=0))
        else:
            res.append(np.concatenate((x, np.array([-1])), axis=0))
    res = np.array(res)
    averages = np.array([average_group_1, average_group_2])
    sci.savemat("naive bayes.mat", {'GNB': res, 'average': averages})


if __name__ == "__main__":
    main()
