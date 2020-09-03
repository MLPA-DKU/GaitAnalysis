from functools import wraps
from random import random, seed, sample
import numpy as np
import pandas as pd
import datetime
import time
import Code.method_collector as mc
import Code.create_collector as cc
import cv2
import math


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
    elif param.method == "LeaveOne" or param.method == "sleaveone":
        return mc.method_leaveone(param, comb, datasets)
    elif param.method == "Feature_Added_LeaveOne":
        return mc.method_fa_leaveone(param, comb, datasets)
    elif param.method == "mdpi":
        return mc.method_mdpi(param, comb, datasets)
    elif param.method == "smdpi":
        return mc.method_smdpi(param, comb, datasets)
    elif param.method == "dhalf":
        return mc.method_dhalf(param, comb, datasets)
    elif param.method == "half":
        return mc.method_half(param, comb, datasets)
    elif param.method == "MCCV":
        return mc.method_MCCV(param, comb, datasets)
    elif param.method == "7CV" or param.method == "MCCV":
        return mc.method_CV(param, comb, datasets)
    elif param.method == "div_vec":
        return mc.method_vec(param, comb, datasets)


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


def vti_preprocess(param, pns, cns, datasets):
    for pn, cn, dataset in zip(pns, cns, datasets):
        print(pn, cn)


def complementry_filter(acc, gyro):
    # angle = 0.98 * (previous angle +_ gyroData * dt) + 0.02 * (accData)
    pi = 3.14159
    rad_to_deg = 180 / pi
    alpha = 0.98
    gyro_anglel = 0
    gyro_angler = 0
    dt = 1
    w, h = acc.shape
    acclx = np.sum(acc[0, :]) / w
    accly = np.sum(acc[1, :]) / w
    acclz = np.sum(acc[2, :]) / w
    accrx = np.sum(acc[3, :]) / w
    accry = np.sum(acc[4, :]) / w
    accrz = np.sum(acc[5, :]) / w

    gylx = np.sum(gyro[0, :]) / w
    gyly = np.sum(gyro[1, :]) / w
    gylz = np.sum(gyro[2, :]) / w
    gyrx = np.sum(gyro[3, :]) / w
    gyry = np.sum(gyro[4, :]) / w
    gyrz = np.sum(gyro[5, :]) / w

    for i in enumerate(range(w)):
        acc[i, 0] -= acclx
        acc[i, 1] -= accly
        acc[i, 2] -= acclz
        acc[i, 3] -= accrx
        acc[i, 4] -= accry
        acc[i, 5] -= accrz

        angle_acc_y = math.atan2(-acc[i, 0] / np.sqrt(pow(acc[i, 1], 2) + pow(acc[i, 2], 2))) * rad_to_deg
        angle_acc_x = math.atan2(acc[i, 1] / np.sqrt(pow(acc[i, 0], 2) + pow(acc[i, 2], 2))) * rad_to_deg
        angle_acc_z = 0

        gyro /= 131
        dgyl_x = gyro[i, 1]
        dgyl_y = gyro[i, 2]
        dgyl_z = gyro[i, 3]
        dgyr_x = gyro[i, 4]
        dgyr_y = gyro[i, 5]
        dgyr_z = gyro[i, 6]

        gyro_anglel = (0.95 * (gyro_anglel + (dgyl_x * 0.001))) + (0.05 * angle_acc_x)


def vector_merge(datasets, class_count):
    merged_pressure = np.array([])
    merged_accl = np.array([])
    merged_accr = np.array([])
    merged_gyrl = np.array([])
    merged_gyrr = np.array([])
    dataset_info = np.array([])

    if min(class_count) == 0:
        np.add(class_count, 1)
    peo_length = len(datasets.keys())
    for enum_idx, [peo_nb, [class_nb, dataset]] in enumerate(datasets.items()):
        if min(class_count) == 0:
            class_nb += 1

        [pressure, acc, gyro] = dataset
        clear_length = 0
        for foot in range(2):
            tpre = np.array(pressure[foot])
            tacc = np.array(acc[foot])
            tgyr = np.array(gyro[foot])

            w, c = tpre.shape

            nan_cleaner = w
            for idx in range(w):
                nan_state = False
                for jdx in range(c):
                    if str(tpre[idx, jdx]) == 'nan':
                        nan_state = True
                if nan_state is True:
                    nan_cleaner -= 1

            nan_sur1 = nan_cleaner

            w, c = tacc.shape
            nan_cleaner = w
            for idx in range(w):
                nan_state = False
                for jdx in range(c):
                    if str(tacc[idx, jdx]) == 'nan':
                        nan_state = True
                if nan_state is True:
                    nan_cleaner -= 1

            nan_sur2 = nan_cleaner

            w, c = tgyr.shape
            nan_cleaner = w
            for idx in range(w):
                nan_state = False
                for jdx in range(c):
                    if str(tgyr[idx, jdx]) == 'nan':
                        nan_state = True
                if nan_state is True:
                    nan_cleaner -= 1

            nan_sur3 = nan_cleaner

            nan_cleaner = min(nan_sur1, nan_sur2, nan_sur3)

            if clear_length < nan_cleaner:
                clear_length = nan_cleaner

        labels = np.zeros([clear_length, 3])
        labels[:, 0].fill(int(peo_nb))
        labels[:, 1].fill(enum_idx)
        labels[:, 2].fill(class_nb)

        for idx, [tpre, tacc, tgyr] in enumerate(zip(pressure, acc, gyro)):
            pressure[idx] = np.array(tpre)[:clear_length, :].astype(np.int)
            acc[idx] = np.array(tacc)[:clear_length, :].astype(np.int)
            gyro[idx] = np.array(tgyr)[:clear_length, :].astype(np.int)

        if enum_idx == 0:
            merged_pressure = np.hstack([pressure[0], pressure[1]])
            merged_accl = acc[0]
            merged_accr = acc[1]
            merged_gyrl = gyro[0]
            merged_gyrr = gyro[1]
            dataset_info = labels
        else:
            temp_pressure = np.hstack([pressure[0], pressure[1]])
            merged_pressure = np.vstack([merged_pressure, temp_pressure])
            merged_accl = np.vstack([merged_accl, acc[0]])
            merged_accr = np.vstack([merged_accr, acc[1]])
            merged_gyrl = np.vstack([merged_gyrl, gyro[0]])
            merged_gyrr = np.vstack([merged_gyrr, gyro[1]])
            dataset_info = np.vstack([dataset_info, labels])

    return [merged_pressure, merged_accl, merged_accr, merged_gyrl, merged_gyrr, dataset_info]
