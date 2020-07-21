from functools import wraps
from random import random, seed, sample
import numpy as np
import pandas as pd
import datetime
import time
import Code.method_collector as mc
import Code.create_collector as cc
import cv2


def to_categorical_unit(target, nb_class):
    categorical = np.zeros([1, int(nb_class)])
    categorical[0, int(target)] = 1
    return categorical


def from_categorical(target):
    NotImplemented
    return target


def to_categorical(target, nb_class):
    categorical = np.zeros([len(target), nb_class])
    for idx in range(len(target)):
        categorical[idx, int(target[idx])] = 1
    return categorical


def chosen_method(param, comb, datasets):
    if param.datatype == "disease":
        return method_collect(param, comb, datasets)
    elif param.datatype == "type":
        return method_collect(param, comb, datasets)


def feature_add(param, datasets):
    data1, data2, data3 = datasets

    plabel = data1[:, -2]
    tlabel = data1[:, -1]

    # dataset index
    data1 = data1[:, :-2]
    data2 = data2[:, :-2]
    data3 = data3[:, :-2]

    nb_class = int(max(tlabel)) + 1
    nb_people = int(max(plabel)) + 1

    unit_step = cc.get_unit_step(data1)
    cc.get_index_sampling(param, data1, step_index=unit_step)


def method_collect(param, comb, datasets):
    if param.method == "Sn":
        """
            sampling data : [train_list, trainp, trainc, test_list, testp, testc]
        """
        return mc.method_sn(param, comb, datasets)
    elif param.method == "base":
        return mc.method_base(param, comb, datasets)
    elif param.method == "LeaveOne":
        return mc.method_leaveone(param, comb, datasets)
    elif param.method == "SelectLeaveOne":
        return mc.method_sleaveone(param, comb, datasets)
    elif param.method == "Feature_Added_LeaveOne":
        return mc.method_fa_leaveone(param, comb, datasets)
    elif param.method == "mdpi":
        return mc.method_mdpi(param, comb, datasets)
    elif param.method == "dhalf":
        return mc.method_dhalf(param, comb, datasets)
    elif param.method == "half":
        return mc.method_half(param, comb, datasets)
    elif param.method == "MCCV":
        return mc.method_MCCV(param, comb, datasets)
    elif param.method == "7CV" or param.method == "MCCV":
        return mc.method_CV(param, comb, datasets)


# sort people number
def sort_by_people(datasets):
    output_list = list()
    for data in datasets:
        row = data.shape[0]
        nb_people = int(max(data[:, -2])) + 1
        output = np.array([])
        for pn in range(nb_people):
            temp = np.array([])
            state = False
            if not max(data[:, -2] == pn):
                # if max(data[:, -2] == pn) == False:
                continue
            for r in range(row):
                if int(data[r, -2]) == pn and state is False:
                    temp = data[r, :]
                    state = True
                    continue
                elif int(data[r, -2]) == pn and state is True:
                    temp = np.vstack([temp, data[r, :]])
            if pn == 0:
                output = temp
            else:
                output = np.vstack([output, temp])

        output_list.append(output)
    return output_list


# Delete Dataset for Binary Classification
def del_subject(param, datasets, target):
    fired = list()
    for dataset in datasets:
        drow, dcol = dataset.shape
        dataset = dataset[dataset[:, -1] != param.collect["category"][target]]
        fired.append(dataset)

    return fired


def normalize_all_of_length(param, datasets):
    print("normalize")
    min_length = 0
    data_column = param.collect["csv_columns_names"]["datasets"]
    pressure_column = param.collect["csv_columns_names"]["pressure"]
    acc_column = param.collect["csv_columns_names"]["acc"]
    gyro_column = param.collect["csv_columns_names"]["gyro"]

    for key, data_paths in datasets.items():
        for path in data_paths:
            data = pd.read_csv(filepath_or_buffer=path, names=data_column, header=None, skiprows=1, encoding='utf7')
            row, col = data.to_numpy().shape
            if min_length == 0:
                min_length = row
            elif min_length > row and row > 10000:
                min_length = row

    init_state = True
    for key, data_paths in datasets.items():
        for i, path in enumerate(data_paths):
            peo_num = np.full([min_length, 1], int(path.split('/')[-1].split('_')[0]))
            type_num = np.full([min_length, 1], int(key))

            data = pd.read_csv(filepath_or_buffer=path, names=data_column, header=None, skiprows=1, encoding='utf7')
            target = np.array(data[pressure_column].to_numpy(), dtype='float32')
            pressure = cv2.resize(src=target, dsize=(target.shape[1], min_length), interpolation=cv2.INTER_CUBIC)

            target = np.array(data[acc_column].to_numpy(), dtype='float32')
            acc = cv2.resize(src=target, dsize=(target.shape[1], min_length), interpolation=cv2.INTER_CUBIC)

            target = np.array(data[gyro_column].to_numpy(), dtype='float32')
            gyro = cv2.resize(src=target, dsize=(target.shape[1], min_length), interpolation=cv2.INTER_CUBIC)

            if init_state is True:
                pressure_collect = pressure
                acc_collect = acc
                gyro_collect = gyro
                peo_collect = peo_num
                type_collect = type_num
                init_state = False
            else:
                pressure_collect = np.vstack([pressure_collect, pressure])
                acc_collect = np.vstack([acc_collect, acc])
                gyro_collect = np.vstack([gyro_collect, gyro])
                peo_collect = np.vstack([peo_collect, peo_num])
                type_collect = np.vstack([type_collect, type_num])

    return [np.hstack([pressure_collect, peo_collect, type_collect]),
            np.hstack([acc_collect, peo_collect, type_collect]), np.hstack([gyro_collect, peo_collect, type_collect])]
