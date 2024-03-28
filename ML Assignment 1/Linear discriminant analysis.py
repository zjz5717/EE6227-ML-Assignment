# -*- coding: UTF-8 -*-
"""
@Project     : ML Assignment1 
@File        : Linear discriminant analysis.py
@Author      : Super_ze
@Date        : 2024/3/14
@Description : 

Copyright © 2024 Super_ze. All Rights Reserved.
"""

import numpy as np
import scipy.io as sci


def calculate(input_data: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate the average and scatter matrix of input_data
    :param input_data: input data
    :return : average and scatter matrix of input_data
    """
    average = np.mean(input_data, axis=0)
    scatter = np.dot((input_data - average).T, (input_data - average))
    return average, scatter


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
    average_group_1, scatter_group_1 = calculate(group_1)
    average_group_2, scatter_group_2 = calculate(group_2)
    scatter_all_group = scatter_group_1 + scatter_group_2
    w = np.dot(np.linalg.inv(scatter_all_group), (average_group_1 - average_group_2))
    w_0 = np.dot(w.T, (average_group_1 + average_group_2)) / -2

    # Output average and variance
    print("average of group 1:", average_group_1)
    print("scatter matrix of group 1:", scatter_group_1)
    print("average of group 2:", average_group_2)
    print("scatter matrix of group 2:", scatter_group_2)
    print("weight vector:", w)
    print("threshold weight:", w_0)

    # Group test data and save result
    res = []
    for x in data_test:
        test_res = np.dot(w.T, x) + w_0
        if test_res > 0:
            res.append(np.concatenate((x, np.array([1])), axis=0))
        elif test_res < 0:
            res.append(np.concatenate((x, np.array([-1])), axis=0))
        else:
            res.append(np.concatenate((x, np.array([0])), axis=0))
    res = np.array(res)
    sci.savemat("Linear discriminant analysis.mat", {'LNA': res, 'w': w, 'w_0': w_0})


if __name__ == "__main__":
    main()
