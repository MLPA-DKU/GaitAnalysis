import os
import numpy as np
import pandas as pd

from Code.dynamic_library import csv_columns_names
import skimage.transform
import cv2
import datetime
import scipy
from scipy.io import savemat
from sklearn.preprocessing import MinMaxScaler
from Code.utils import dt_printer as dt
from PIL import Image, ImageDraw
# from scipy.misc import imresize
# from skimage.transform import resize as skiresize


# Gaussian Filter Conv Create
def gc_dataset_init(param, file):
    GAUSS_SIGMA = 20

    data_column = csv_columns_names["datasets"]
    # pressure_column = csv_columns_names["pressure"]
    left_column = csv_columns_names["left_pressure"]
    right_column = csv_columns_names["right_pressure"]

    total_dataset = pd.read_csv(filepath_or_buffer=file
                                , names=data_column, header=None, skiprows=1, encoding='utf7')

    left_foot = total_dataset[left_column].to_numpy()
    right_foot = total_dataset[right_column].to_numpy()

    acc = np.array(total_dataset[csv_columns_names["acc"]].to_numpy(), dtype='float32')
    end_acc = int(len(acc[0]) / 2)
    left_acc = acc[:, 0:end_acc]
    right_acc = acc[:, end_acc: 2 * end_acc]

    gyr = np.array(total_dataset[csv_columns_names["gyro"]].to_numpy(), dtype='float32')
    end_gyr = int(len(gyr[0]) / 2)
    left_gyr = gyr[:, 0:end_gyr]
    right_gyr = gyr[:, end_gyr: 2 * end_gyr]

    lpavg = left_foot.mean(axis=1)
    rpavg = right_foot.mean(axis=1)

    lpconv = scipy.ndimage.filters.gaussian_filter(lpavg, sigma=GAUSS_SIGMA)
    rpconv = scipy.ndimage.filters.gaussian_filter(rpavg, sigma=GAUSS_SIGMA)

    diff = [np.diff(lpconv), np.diff(rpconv)]
    # Find the minimum of the curve
    # Process one foot at a time. Left=side 0, right=side 1
    unit_limits = [[], []]
    for side in [0, 1]:
        prev = 0
        for i in range(len(diff[side])):
            if prev < 0 and diff[side][i] >= 0:
                unit_limits[side].append(i)
            prev = diff[side][i]

    limit_L = np.concatenate(([0], unit_limits[0], [len(left_foot)]))
    limit_R = np.concatenate(([0], unit_limits[1], [len(right_foot)]))

    left_unit_list = []
    right_unit_list = []

    for i in range(2, min(len(limit_L), len(limit_R)) - 3):
        left = left_foot[limit_L[i]:limit_L[i + 1]].astype('int64')
        right = right_foot[limit_R[i]:limit_R[i + 1]].astype('int64')

        left_fin_acc = left_acc[limit_L[i]:limit_L[i + 1]].astype('int64')
        right_fin_acc = right_acc[limit_R[i]:limit_R[i + 1]].astype('int64')

        left_fin_gyr = left_gyr[limit_L[i]:limit_L[i + 1]].astype('int64')
        right_fin_gyr = right_gyr[limit_R[i]:limit_R[i + 1]].astype('int64')

        if len(left) >= 63 and len(right) >= 63:
            left_unitstep = [left, left_fin_acc, left_fin_gyr]
            right_unitstep = [right, right_fin_acc, right_fin_gyr]

            left_unit_list.append(left_unitstep)
            right_unit_list.append(right_unitstep)

    return left_unit_list, right_unit_list


# Gaussian Filter Sample Resize
def resize_v3_samples(param, samples):
    sensor_numbers = param.collect["sensor_number"]
    sensor_names = param.sensor_type
    mt = 63

    min_value = 1000
    collect_idle = np.array([])
    for idx, (left, right) in enumerate(zip(samples[0], samples[1])):
        smallest = min(left[0].shape[0], right[0].shape[0])

        min_value = min(min_value, smallest)
        print(f"minimum -> {min_value}")

        pn = np.full((mt, 1), int(param.pn))
        key = np.full((mt, 1), int(param.key))

        left = np.concatenate((left[0], left[1], left[2]), axis=1)
        right = np.concatenate((right[0], right[1], right[2]), axis=1)

        left = scipy.ndimage.zoom(left, [mt / len(left), 1])
        right = scipy.ndimage.zoom(right, [mt / len(right), 1])

        merged = np.concatenate((left, right), axis=1)
        if idx == 0:
            collect_idle = np.concatenate((merged, key, pn), axis=1)
        else:
            temp_idle = np.concatenate((merged, key, pn), axis=1)
            collect_idle = np.concatenate((collect_idle, temp_idle), axis=0)

    return collect_idle, min_value


# Gaussian Filter Normalize
def normalize_v3_samples(param, samples):
    mt = param.collect["minimum_threshold"]
    minValue = [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1] * 2

    total_merged = np.array([])
    for idx, [key, dataset] in enumerate(samples.items()):
        for jdx, data in enumerate(sorted(dataset)):
            [file, idle_sample] = data

            print(file, idle_sample.shape)
            if idle_sample.shape[0] < 200:
                continue
            if idx == 0 and jdx == 0:
                total_merged = idle_sample
            else:
                try:
                    total_merged = np.concatenate((total_merged, idle_sample), axis=0)
                except ValueError:
                    print("get error!!")
                    continue

            # irow, icol = idle_sample.shape # icol -> [0: -2], [pn, class]
            # breakpoint()

    trow, tcol = total_merged.shape
    normalized = []
    for i in range(tcol - 2):
        after = total_merged[:, i]
        after = after.reshape(-1, 1).astype("float64")
        scaler = MinMaxScaler(feature_range=(minValue[i], 1))
        scaler.fit(after)
        after = scaler.transform(after)[:, 0]
        normalized.append(after)

    rotate = np.concatenate((np.transpose(normalized), total_merged[:, -2:]), axis=1)
    rrow, rcol = rotate.shape

    dataset = np.array([])
    for i in range(0, (rrow // mt) * mt, mt):
        rows = rotate[i: i+mt]


        if i == 0:
            dataset = rows
        else:
            dataset = np.concatenate((dataset, rows), axis=0)

        print(f"now normalize.. {i}/{(rrow // mt) * mt}")

    return dataset


# Combine
def combined_samples_v3(samples, comb_degree):
    mt = 63
    samples = samples[samples[:, -2].argsort()]

    combine_sample = []
    label_sample = []
    for fidx, i in enumerate(np.unique(samples[:, -2])):
        pn = i
        temp1 = samples[np.where(samples[:, -2] == int(i))]
        temp1 = temp1[temp1[:, -1].argsort()]
        for fjdx, j in enumerate(np.unique(temp1[:, -1])):
            temp2 = temp1[np.where(temp1[:, -1] == int(j))]
            tag = j

            data = np.reshape(temp2[:len(temp2) // (mt * comb_degree) * (mt * comb_degree)],
                              (-1, mt * comb_degree, len(temp2[0] - 2)))

            label = temp2[:data.shape[0], -2:]

            combine_sample.append(data)
            label_sample.append(label)
            print(f" people {pn}, class {tag}, comb degree {comb_degree} combining")

    return [np.vstack(combine_sample), np.vstack(label_sample)]


def save_datasets_v3(param, data_collect):
    mt = param.collect["minimum_threshold"]
    for nb_comb, collected in data_collect.items():

        dataset, label = collected
        if param.object == "custom":
            save_dir = f"../Datasets/{param.folder}"
        else:
            save_dir = f"../Datasets/{param.folder}"
            # save_dir = f"../Datasets/{datetime.datetime.today().strftime('%y%m%d')}_{param.datatype}"

        folder_dir = os.path.join(save_dir, f"Sample_{nb_comb}")
        if os.path.exists(save_dir) is not True:
            os.mkdir(save_dir)
        if os.path.exists(folder_dir) is not True:
            os.mkdir(folder_dir)

        pl = 0
        if param.datatype == "disease_add" or param.datatype == "disease":
            keymap = dict()
            pressure_data = np.concatenate((dataset[:, :, 0:8], dataset[:, :, 14:22]), axis=1)
            pressure_data = np.reshape(pressure_data, (-1, mt * len(pressure_data[0]) * nb_comb))
            acc_data = np.concatenate((dataset[:, :, 8:11], dataset[:, :, 22:25]), axis=1)
            acc_data = np.reshape(acc_data, (-1, mt * len(acc_data[0]) * nb_comb))
            gyro_data = np.concatenate((dataset[:, :, 11:14], dataset[:, :, 25:28]), axis=1)
            gyro_data = np.reshape(gyro_data, (-1, mt * len(gyro_data[0]) * nb_comb))

            pressure_data = np.concatenate((pressure_data, label), axis=1)
            acc_data = np.concatenate((acc_data, label), axis=1)
            gyro_data = np.concatenate((gyro_data, label), axis=1)

            # converted = np.reshape((-1, mt * len(combine_sample[0]) * comb_degree))
            if param.model_name == "dat":
                file_name = os.path.join(folder_dir, "pressure_dataset.dat")
                np.savetxt(fname=file_name, X=pressure_data)
                file_name = os.path.join(folder_dir, "acc_dataset.dat")
                np.savetxt(fname=file_name, X=acc_data)
                file_name = os.path.join(folder_dir, "gyro_dataset.dat")
                np.savetxt(fname=file_name, X=gyro_data)

            elif param.model_name == "npy":
                np.save(os.path.join(folder_dir, "pressure_dataset.npy"), pressure_data)
                np.save(os.path.join(folder_dir, "acc_dataset.npy"), acc_data)
                np.save(os.path.join(folder_dir, "gyro_dataset.npy"), gyro_data)
            elif param.model_name == "mat":
                matdict = dict()
                matdict['pressure'] = pressure_data
                matdict['acc'] = acc_data
                matdict['gyro'] = gyro_data
                savemat(os.path.join(folder_dir, "matfiles.mat"), matdict)
            elif param.model_name == "all":
                file_name = os.path.join(folder_dir, "pressure_dataset.dat")
                np.savetxt(fname=file_name, X=pressure_data)
                file_name = os.path.join(folder_dir, "acc_dataset.dat")
                np.savetxt(fname=file_name, X=acc_data)
                file_name = os.path.join(folder_dir, "gyro_dataset.dat")
                np.savetxt(fname=file_name, X=gyro_data)

                np.save(os.path.join(folder_dir, "pressure_dataset.npy"), pressure_data)
                np.save(os.path.join(folder_dir, "acc_dataset.npy"), acc_data)
                np.save(os.path.join(folder_dir, "gyro_dataset.npy"), gyro_data)

                matdict = dict()
                matdict['pressure'] = pressure_data
                matdict['acc'] = acc_data
                matdict['gyro'] = gyro_data
                savemat(os.path.join(folder_dir, "matfiles.mat"), matdict)

            f = open(os.path.join(save_dir, 'keymap.txt'), 'w')
            f.write(str(keymap))
            f.close()

        elif param.datatype == "type":
            keymap = dict()
            pressure_data = np.concatenate((dataset[:, :, 0:8], dataset[:, :, 14:22]), axis=2)
            pressure_data = np.reshape(pressure_data, (-1, mt * pressure_data.shape[2] * nb_comb))
            acc_data = np.concatenate((dataset[:, :, 8:11], dataset[:, :, 22:25]), axis=2)
            acc_data = np.reshape(acc_data, (-1, mt * acc_data.shape[2] * nb_comb))
            gyro_data = np.concatenate((dataset[:, :, 11:14], dataset[:, :, 25:28]), axis=2)
            gyro_data = np.reshape(gyro_data, (-1, mt * gyro_data.shape[2] * nb_comb))

            pressure_data = np.concatenate((pressure_data, label -1), axis=1)
            acc_data = np.concatenate((acc_data, label -1), axis=1)
            gyro_data = np.concatenate((gyro_data, label -1), axis=1)

            if param.model_name == "dat":
                file_name = os.path.join(folder_dir, "pressure_dataset.dat")
                np.savetxt(fname=file_name, X=pressure_data)
                file_name = os.path.join(folder_dir, "acc_dataset.dat")
                np.savetxt(fname=file_name, X=acc_data)
                file_name = os.path.join(folder_dir, "gyro_dataset.dat")
                np.savetxt(fname=file_name, X=gyro_data)

            elif param.model_name == "npy":
                np.save(os.path.join(folder_dir, "pressure_dataset.npy"), pressure_data)
                np.save(os.path.join(folder_dir, "acc_dataset.npy"), acc_data)
                np.save(os.path.join(folder_dir, "gyro_dataset.npy"), gyro_data)
            elif param.model_name == "mat":
                matdict = dict()
                matdict['pressure'] = pressure_data
                matdict['acc'] = acc_data
                matdict['gyro'] = gyro_data
                savemat(os.path.join(folder_dir, "matfiles.mat"), matdict)
            elif param.model_name == "all":
                file_name = os.path.join(folder_dir, "pressure_dataset.dat")
                np.savetxt(fname=file_name, X=pressure_data)
                file_name = os.path.join(folder_dir, "acc_dataset.dat")
                np.savetxt(fname=file_name, X=acc_data)
                file_name = os.path.join(folder_dir, "gyro_dataset.dat")
                np.savetxt(fname=file_name, X=gyro_data)

                np.save(os.path.join(folder_dir, "pressure_dataset.npy"), pressure_data)
                np.save(os.path.join(folder_dir, "acc_dataset.npy"), acc_data)
                np.save(os.path.join(folder_dir, "gyro_dataset.npy"), gyro_data)

                matdict = dict()
                matdict['pressure'] = pressure_data
                matdict['acc'] = acc_data
                matdict['gyro'] = gyro_data
                savemat(os.path.join(folder_dir, "matfiles.mat"), matdict)

            f = open(os.path.join(save_dir, 'keymap.txt'), 'w')
            f.write(str(keymap))
            f.close()


# Init
def dataset_init(param, file):
    data_column = csv_columns_names["datasets"]
    pressure_column = csv_columns_names["pressure"]
    left_column = csv_columns_names["left_pressure"]
    right_column = csv_columns_names["right_pressure"]

    total_dataset = pd.read_csv(filepath_or_buffer=file
                                , names=data_column, header=None, skiprows=1, encoding='utf7')

    if param.collect['latency'] != 100:
        latency = int(100 / param.collect['latency'])
        total_dataset = total_dataset[::latency]

    df = total_dataset[pressure_column]

    step_index = get_unit_step(df[left_column])

    return total_dataset, step_index


# Get Unit Step
def get_unit_step(data):
    # get Each unit step
    sum_lp = data.sum(axis=1)
    sample_length = 0
    swing_phase = False

    step_index = list()

    for result_lps in sum_lp:
        if 0 == result_lps and swing_phase is False:
            step_index.append(sample_length)
            swing_phase = True
        elif 0 != result_lps and swing_phase is True:
            swing_phase = False
        sample_length = sample_length + 1

    return step_index


# Get Unit Step
def get_unit_step_v2(data):
    # get Each unit step
    sum_lp = data.sum(axis=1)
    sample_length = 0
    swing_phase = False

    step_index = list()

    for result_lps in sum_lp:
        if 0 == result_lps and swing_phase is False:
            step_index.append(sample_length)
            swing_phase = True
        elif 0 != result_lps and swing_phase is True:
            swing_phase = False
        sample_length = sample_length + 1

    return step_index


# Sampling
def get_index_sampling(param, data, step_index):
    pressure_column = csv_columns_names["pressure"]
    acc_column = csv_columns_names["acc"]
    gyro_column = csv_columns_names["gyro"]

    pressure = data[pressure_column]
    acc = data[acc_column]
    gyro = data[gyro_column]

    out_p = list()
    out_a = list()
    out_g = list()

    if param.method == "add_noise":
        noise = pd.DataFrame(np.random.random((data.shape[0], 16)), columns=pressure_column) * 0.01
        pressure += noise

    for idx in range(0, len(step_index) - 1):
        pressure_loc = pressure.iloc[step_index[idx]:step_index[idx + 1]]
        acc_loc = acc.iloc[step_index[idx]:step_index[idx + 1]]
        gyro_loc = gyro.iloc[step_index[idx]:step_index[idx + 1]]

        out_p.append(pressure_loc)
        out_a.append(acc_loc)
        out_g.append(gyro_loc)

    return [out_p, out_a, out_g]


# Resize
def resize_samples(param, samples):
    sensor_numbers = param.collect["sensor_number"]
    sensor_names = param.sensor_type
    mt = param.collect["minimum_threshold"]

    resized = list()
    for sid, sensor_sample in enumerate(samples):
        sn = sensor_numbers[sensor_names[sid]]
        out = list()
        for sample in sensor_sample:
            sample = np.array(sample.to_numpy(), dtype='float32')
            # new method recommend :  data = cv2.resize(src=sample, dsize=(sn, mt), interpolation=cv2.INTER_CUBIC)
            data = imresize(arr=sample, size=[mt, sn],
                                           interp='bilinear', mode='F')

            out.append(data)
        resized.append(out)

    return resized


# Combine
def combined_samples(samples, comb_degree):
    combined = list()
    for sample in samples:
        combine_list = list()
        for index in range(0, len(sample), comb_degree):
            temp_sample = np.array([])
            try:
                for combine_index in range(0, comb_degree):
                    if 0 == combine_index:
                        temp_sample = sample[index + combine_index]
                    else:
                        temp_sample = np.vstack((temp_sample, sample[index + combine_index]))
                combine_list.append(temp_sample)
            except:
                continue

        combined.append(combine_list)
    return combined


# Vectorize
def vectorized_samples(param, samples, comb_degree):
    sensor_numbers = param.collect["sensor_number"]
    sensor_names = param.sensor_type
    mt = param.collect["minimum_threshold"]

    vectorized = list()
    for sid, data_list in enumerate(samples):
        temp_vector = np.array([])
        sn = sensor_numbers[sensor_names[sid]]
        for i, sample in enumerate(data_list):
            vector = np.reshape(a=sample, newshape=[1, comb_degree * mt * sn])
            if i == 0:
                temp_vector = vector
            else:
                temp_vector = np.vstack([temp_vector, vector])

        vectorized.append(temp_vector)

    return vectorized


# totalize
def label_mapping(pn, key, sensors):
    output = list()
    row, _ = sensors[0].shape
    plabel = np.full((row, 1), pn)
    clabel = np.full((row, 1), key)
    for data in sensors:
        data = np.hstack([data, plabel, clabel])
        output.append(data)
    return output


def save_datasets(param, data_collect, nb_comb):
    if param.object == "custom":
        save_dir = f"../Datasets/{param.folder}"
    else:
        save_dir = f"../Datasets/{param.folder}"
        # save_dir = f"../Datasets/{datetime.datetime.today().strftime('%y%m%d')}_{param.datatype}"

    folder_dir = os.path.join(save_dir, f"Sample_{nb_comb}")
    if os.path.exists(save_dir) is not True:
        os.mkdir(save_dir)
    if os.path.exists(folder_dir) is not True:
        os.mkdir(folder_dir)

    pl = 0
    if param.datatype == "disease_add" or param.datatype == "disease":
        keymap = dict()
        state = False
        pressure_data = np.array([])
        acc_data = np.array([])
        gyro_data = np.array([])
        for key, item in data_collect.items():
            for file_name, datasets in sorted(item):
                file_name = file_name.split('/')[-1].split('.csv')[0]
                pn, cn = file_name.split('_')

                keymap[file_name] = (key, pl, cn)
                try:
                    [pressure, acc, gyro] = label_mapping(pl, int(cn)+1, datasets)
                except:
                    continue

                if state is False:
                    pressure_data = pressure
                    acc_data = acc
                    gyro_data = gyro
                    state = True
                else:
                    pressure_data = np.vstack([pressure_data, pressure])
                    acc_data = np.vstack([acc_data, acc])
                    gyro_data = np.vstack([gyro_data, gyro])

                pl += 1

        if param.model_name == "dat":
            file_name = os.path.join(folder_dir, "pressure_dataset.dat")
            np.savetxt(fname=file_name, X=pressure_data)
            file_name = os.path.join(folder_dir, "acc_dataset.dat")
            np.savetxt(fname=file_name, X=acc_data)
            file_name = os.path.join(folder_dir, "gyro_dataset.dat")
            np.savetxt(fname=file_name, X=gyro_data)

        elif param.model_name == "npy":
            np.save(os.path.join(folder_dir, "pressure_dataset.npy"), pressure_data)
            np.save(os.path.join(folder_dir, "acc_dataset.npy"), acc_data)
            np.save(os.path.join(folder_dir, "gyro_dataset.npy"), gyro_data)
        elif param.model_name == "mat":
            matdict = dict()
            matdict['pressure'] = pressure_data
            matdict['acc'] = acc_data
            matdict['gyro'] = gyro_data
            savemat(os.path.join(folder_dir, "matfiles.mat"), matdict)
        elif param.model_name == "all":
            file_name = os.path.join(folder_dir, "pressure_dataset.dat")
            np.savetxt(fname=file_name, X=pressure_data)
            file_name = os.path.join(folder_dir, "acc_dataset.dat")
            np.savetxt(fname=file_name, X=acc_data)
            file_name = os.path.join(folder_dir, "gyro_dataset.dat")
            np.savetxt(fname=file_name, X=gyro_data)

            np.save(os.path.join(folder_dir, "pressure_dataset.npy"), pressure_data)
            np.save(os.path.join(folder_dir, "acc_dataset.npy"), acc_data)
            np.save(os.path.join(folder_dir, "gyro_dataset.npy"), gyro_data)

            matdict = dict()
            matdict['pressure'] = pressure_data
            matdict['acc'] = acc_data
            matdict['gyro'] = gyro_data
            savemat(os.path.join(folder_dir, "matfiles.mat"), matdict)

        f = open(os.path.join(save_dir, 'keymap.txt'), 'w')
        f.write(str(keymap))
        f.close()

    elif param.datatype == "type":
        keymap = dict()
        pressure_data = np.array([])
        acc_data = np.array([])
        gyro_data = np.array([])

        for key, item in data_collect.items():
            for file_name, datasets in sorted(item):
                file_name = file_name.split('/')[-1].split('.csv')[0]
                pn, cn = file_name.split('_')

                try:
                    [pressure, acc, gyro] = label_mapping(int(pn) - 1, int(cn), datasets)
                    keymap[file_name] = int(pn) - 1
                except:
                    continue

                if int(pn) - 1 == 0 and key == '01':
                    pressure_data = pressure
                    acc_data = acc
                    gyro_data = gyro
                else:
                    pressure_data = np.vstack([pressure_data, pressure])
                    acc_data = np.vstack([acc_data, acc])
                    gyro_data = np.vstack([gyro_data, gyro])

        if param.model_name == "dat":
            np.savetxt(os.path.join(folder_dir, "pressure_dataset.dat"), pressure_data)
            np.savetxt(os.path.join(folder_dir, "acc_dataset.dat"), acc_data)
            np.savetxt(os.path.join(folder_dir, "gyro_dataset.dat"), gyro_data)
        elif param.model_name == "npy":
            np.save(os.path.join(folder_dir, "pressure_dataset.npy"), pressure_data)
            np.save(os.path.join(folder_dir, "acc_dataset.npy"), acc_data)
            np.save(os.path.join(folder_dir, "gyro_dataset.npy"), gyro_data)
        elif param.model_name == "all":
            file_name = os.path.join(folder_dir, "pressure_dataset.dat")
            np.savetxt(fname=file_name, X=pressure_data)
            file_name = os.path.join(folder_dir, "acc_dataset.dat")
            np.savetxt(fname=file_name, X=acc_data)
            file_name = os.path.join(folder_dir, "gyro_dataset.dat")
            np.savetxt(fname=file_name, X=gyro_data)

            np.save(os.path.join(folder_dir, "pressure_dataset.npy"), pressure_data)
            np.save(os.path.join(folder_dir, "acc_dataset.npy"), acc_data)
            np.save(os.path.join(folder_dir, "gyro_dataset.npy"), gyro_data)

        f = open(os.path.join(save_dir, 'keymap.txt'), 'w')
        f.write(str(keymap))
        f.close()


# Init
def dataset_init_v2(param, file):
    data_column = csv_columns_names["datasets"]
    pressure_column = csv_columns_names["pressure"]
    left_column = csv_columns_names["left_pressure"]
    right_column = csv_columns_names["right_pressure"]

    total_dataset = pd.read_csv(filepath_or_buffer=file
                                , names=data_column, header=None, skiprows=1, encoding='utf7')
    df = total_dataset[pressure_column]
    indexes = get_unit_step_v2([df[left_column], df[right_column]])

    return total_dataset, indexes


# Get Unit Step
def get_unit_step_v2(data):
    # get Each unit step
    sum_lp = data[0].sum(axis=1)
    sum_rp = data[1].sum(axis=1)

    left_index = list()
    stance_phase = False
    sample_length = 0
    for result_lps in sum_lp:
        if 0 != result_lps and stance_phase is False:
            left_index.append(sample_length)
            stance_phase = True
        elif 0 == result_lps and stance_phase is True:
            left_index.append(sample_length)
            stance_phase = False
        sample_length = sample_length + 1

    right_index = list()
    stance_phase = False
    sample_length = 0
    for result_lps in sum_rp:
        if 0 != result_lps and stance_phase is False:
            right_index.append(sample_length)
            stance_phase = True
        elif 0 == result_lps and stance_phase is True:
            right_index.append(sample_length)
            stance_phase = False
        sample_length = sample_length + 1

    return [left_index, right_index]


# Sampling
def get_index_sampling_v2(param, data, step_index):
    pressure_column = csv_columns_names["pressure"]
    acc_column = csv_columns_names["acc"]
    gyro_column = csv_columns_names["gyro"]

    pressure = data[pressure_column]
    acc = data[acc_column]
    gyro = data[gyro_column]

    out_p = list()
    out_a = list()
    out_g = list()

    if param.method == "add_noise":
        noise = pd.DataFrame(np.random.random((data.shape[0], 16)), columns=pressure_column) * 0.01
        pressure += noise

    left_index = step_index[0][2:-2]
    right_index = step_index[1][2:-2]

    ltemp = left_index
    rtemp = right_index
    for limit in range(len(right_index)):
        if ltemp[0] > right_index[limit]:
            rtemp = rtemp[limit+1:]
        if len(ltemp) > len(rtemp):
            ltemp = ltemp[:len(ltemp)-1]

    for idx in range(0, len(ltemp) - 1, 2):
        pressure_loc = pressure.iloc[ltemp[idx]:rtemp[idx + 1]]
        acc_loc = acc.iloc[ltemp[idx]:rtemp[idx + 1]]
        gyro_loc = gyro.iloc[ltemp[idx]:rtemp[idx + 1]]

        out_p.append(pressure_loc)
        out_a.append(acc_loc)
        out_g.append(gyro_loc)

    return [out_p, out_a, out_g]


# Resize
def resize_samples_v2(param, samples):
    sensor_numbers = param.collect["sensor_number"]
    sensor_names = param.sensor_type
    mt = param.collect["minimum_threshold"]

    resized = list()
    for sid, sensor_sample in enumerate(samples):
        sn = sensor_numbers[sensor_names[sid]]
        out = list()
        for sample in sensor_sample:
            sample = np.array(sample.to_numpy(), dtype='float32')
            print(sample.shape)
            if sample.shape[0] < 1:
                continue
            data = cv2.resize(src=sample, dsize=(sn, mt), interpolation=cv2.INTER_CUBIC)
            out.append(data)
        resized.append(out)

    return resized


# Combine
def combined_samples_v2(samples, comb_degree):
    combined = list()
    for sample in samples:
        combine_list = list()
        for index in range(0, len(sample), comb_degree):
            temp_sample = np.array([])
            try:
                for combine_index in range(0, comb_degree):
                    if 0 == combine_index:
                        temp_sample = sample[index + combine_index]
                    else:
                        temp_sample = np.vstack((temp_sample, sample[index + combine_index]))
                combine_list.append(temp_sample)
            except:
                continue

        combined.append(combine_list)
    return combined


# Vectorize
def vectorized_samples_v2(param, samples, comb_degree):
    sensor_numbers = param.collect["sensor_number"]
    sensor_names = param.sensor_type
    mt = param.collect["minimum_threshold"]

    vectorized = list()
    for sid, data_list in enumerate(samples):
        temp_vector = np.array([])
        sn = sensor_numbers[sensor_names[sid]]
        for i, sample in enumerate(data_list):
            vector = np.reshape(a=sample, newshape=[1, comb_degree * mt * sn])
            if i == 0:
                temp_vector = vector
            else:
                temp_vector = np.vstack([temp_vector, vector])

        vectorized.append(temp_vector)

    return vectorized


# totalize
def label_mapping_v2(pn, key, sensors):
    output = list()
    row, _ = sensors[0].shape
    plabel = np.full((row, 1), pn)
    clabel = np.full((row, 1), key)
    for data in sensors:
        data = np.hstack([data, plabel, clabel])
        output.append(data)
    return output


def save_datasets_v2(param, data_collect, nb_comb):
    if param.object == "custom":
        save_dir = f"../Datasets/{param.folder}"
    else:
        save_dir = f"../Datasets/{datetime.datetime.today().strftime('%y%m%d')}_{param.datatype}"

    folder_dir = os.path.join(save_dir, f"Sample_{nb_comb}")
    if os.path.exists(save_dir) is not True:
        os.mkdir(save_dir)
    if os.path.exists(folder_dir) is not True:
        os.mkdir(folder_dir)

    pl = 0
    if param.datatype == "disease_add" or param.datatype == "disease":
        keymap = dict()
        state = False
        pressure_data = np.array([])
        acc_data = np.array([])
        gyro_data = np.array([])
        for key, item in data_collect.items():
            for file_name, datasets in sorted(item):
                file_name = file_name.split('/')[-1].split('.csv')[0]
                pn, cn = file_name.split('_')

                keymap[file_name] = (key, pl, cn)
                [pressure, acc, gyro] = label_mapping(pl, int(cn), datasets)

                if state is False:
                    pressure_data = pressure
                    acc_data = acc
                    gyro_data = gyro
                    state = True
                else:
                    pressure_data = np.vstack([pressure_data, pressure])
                    acc_data = np.vstack([acc_data, acc])
                    gyro_data = np.vstack([gyro_data, gyro])

                pl += 1

        if param.model_name == "dat":
            file_name = os.path.join(folder_dir, "pressure_dataset.dat")
            np.savetxt(fname=file_name, X=pressure_data)
            file_name = os.path.join(folder_dir, "acc_dataset.dat")
            np.savetxt(fname=file_name, X=acc_data)
            file_name = os.path.join(folder_dir, "gyro_dataset.dat")
            np.savetxt(fname=file_name, X=gyro_data)

        elif param.model_name == "npy":
            np.save(os.path.join(folder_dir, "pressure_dataset.npy"), pressure_data)
            np.save(os.path.join(folder_dir, "acc_dataset.npy"), acc_data)
            np.save(os.path.join(folder_dir, "gyro_dataset.npy"), gyro_data)
        elif param.model_name == "all":
            file_name = os.path.join(folder_dir, "pressure_dataset.dat")
            np.savetxt(fname=file_name, X=pressure_data)
            file_name = os.path.join(folder_dir, "acc_dataset.dat")
            np.savetxt(fname=file_name, X=acc_data)
            file_name = os.path.join(folder_dir, "gyro_dataset.dat")
            np.savetxt(fname=file_name, X=gyro_data)

            np.save(os.path.join(folder_dir, "pressure_dataset.npy"), pressure_data)
            np.save(os.path.join(folder_dir, "acc_dataset.npy"), acc_data)
            np.save(os.path.join(folder_dir, "gyro_dataset.npy"), gyro_data)

        f = open(os.path.join(save_dir, 'keymap.txt'), 'w')
        f.write(str(keymap))
        f.close()

    elif param.datatype == "type":
        keymap = dict()
        pressure_data = np.array([])
        acc_data = np.array([])
        gyro_data = np.array([])

        for key, item in data_collect.items():
            for file_name, datasets in sorted(item):
                file_name = file_name.split('/')[-1].split('.csv')[0]
                pn, cn = file_name.split('_')

                try:
                    [pressure, acc, gyro] = label_mapping(int(pn) - 1, int(cn), datasets)
                    keymap[file_name] = int(pn) - 1
                except:
                    continue

                if int(pn) - 1 == 0 and key == '01':
                    pressure_data = pressure
                    acc_data = acc
                    gyro_data = gyro
                else:
                    pressure_data = np.vstack([pressure_data, pressure])
                    acc_data = np.vstack([acc_data, acc])
                    gyro_data = np.vstack([gyro_data, gyro])

        if param.model_name == "dat":
            np.savetxt(os.path.join(folder_dir, "pressure_dataset.dat"), pressure_data)
            np.savetxt(os.path.join(folder_dir, "acc_dataset.dat"), acc_data)
            np.savetxt(os.path.join(folder_dir, "gyro_dataset.dat"), gyro_data)
        elif param.model_name == "npy":
            np.save(os.path.join(folder_dir, "pressure_dataset.npy"), pressure_data)
            np.save(os.path.join(folder_dir, "acc_dataset.npy"), acc_data)
            np.save(os.path.join(folder_dir, "gyro_dataset.npy"), gyro_data)
        elif param.model_name == "mat":
            matdict = dict()
            matdict['pressure'] = pressure_data
            matdict['acc'] = acc_data
            matdict['gyro'] = gyro_data
            savemat(os.path.join(folder_dir, "matfiles.npy"), matdict)
        elif param.model_name == "all":
            file_name = os.path.join(folder_dir, "pressure_dataset.dat")
            np.savetxt(fname=file_name, X=pressure_data)
            file_name = os.path.join(folder_dir, "acc_dataset.dat")
            np.savetxt(fname=file_name, X=acc_data)
            file_name = os.path.join(folder_dir, "gyro_dataset.dat")
            np.savetxt(fname=file_name, X=gyro_data)

            np.save(os.path.join(folder_dir, "pressure_dataset.npy"), pressure_data)
            np.save(os.path.join(folder_dir, "acc_dataset.npy"), acc_data)
            np.save(os.path.join(folder_dir, "gyro_dataset.npy"), gyro_data)

            matdict = dict()
            matdict['pressure'] = pressure_data
            matdict['acc'] = acc_data
            matdict['gyro'] = gyro_data
            savemat(os.path.join(folder_dir, "matfiles.mat"), matdict)

        f = open(os.path.join(save_dir, 'keymap.txt'), 'w')
        f.write(str(keymap))
        f.close()


def vti_init(param, file):

    data_column = csv_columns_names["datasets"]

    total_dataset = pd.read_csv(filepath_or_buffer=file
                                , names=data_column, header=None, skiprows=1, encoding='utf7')
    left_pressure = total_dataset[csv_columns_names["left_pressure"]]
    right_pressure = total_dataset[csv_columns_names["right_pressure"]]
    left_acc = total_dataset[csv_columns_names["left_acc"]]
    right_acc = total_dataset[csv_columns_names["right_acc"]]
    left_gyro = total_dataset[csv_columns_names["left_gyro"]]
    right_gyro = total_dataset[csv_columns_names["right_gyro"]]

    return [left_pressure, right_pressure], [left_acc, right_acc], [left_gyro, right_gyro]


def image_pixel_to_eps(img, width, height, value):
    # ###################### using distance and gaussian ##################################
    # self.GaussianMatrix(get_image=img, value=value, sigma=value, width=width, height=height)

    # ################################# elipse - easy mode ################################
    if value == 2:
        fill_out = 255
    elif value == 1:
        fill_out = 170
    elif value == 0:
        fill_out = 85

    draw = ImageDraw.Draw(img)
    draw.ellipse([(width - 36, height - 36), (width + 36, height + 36)], fill=fill_out)

    # img_size = (360, 760)
    # for y in range(img_size[0]):
    #    color = 0
    #    for x in range(img_size[1]):
    #
    #        #distanceToCenter = math.sqrt((x - width)**2 + (y - height)**2)
    #
    #        #distanceToCenter = float(distanceToCenter) / (math.sqrt(2) * width)
    #
    #        #color = 0 * distanceToCenter + value * ( 1 - distanceToCenter)
    #
    #        if color < width - 85 :
    #            color = 85
    #        elif color > 255:
    #            color = 255
    #
    #        img.putpixel(xy=(width, height), value=(int(color)))

    return img


def pressure_vti(dataset):
    # sensor_1 = (241,74) #241 , 74
    # sensor_2 = (256,195) #256, 195
    # sensor_3 = (241,316) #241, 316
    # sensor_4 = (218,762) #218, 762
    # sensor_5 = (100,157) #100, 157
    # sensor_6 = (70,297) #70, 297
    # sensor_7 = (70,437) #70, 437
    # sensor_8 = (96,762) #96, 762
    sensor_points = {
        'left': [(241, 74), (256, 195), (241, 316), (218, 762), (100, 157), (70, 297), (70, 437), (96, 762)],
        'right': [(99, 74), (85, 195), (99, 316), (148, 762), (260, 157), (290, 297), (290, 437), (264, 762)]}

    # left_data = dataset[0].as_matrix()
    # right_data = dataset[1].as_matrix()
    left_data = dataset[0].values
    right_data = dataset[1].values
    if left_data.shape[0] != left_data.shape[0]:
        min_sample = min(left_data.shape[0], right_data.shape[0])
        left_data = left_data[:min_sample]
        right_data = left_data[:min_sample]
    sample_len = len(dataset[0].values)
    sample_col = left_data.shape[1]

    if sample_len < 500:
        return 0

    nan_cleaner = sample_len
    for idx in range(sample_len):
        nan_state = False
        for jdx in range(sample_col):
            if str(left_data[idx, jdx]) == 'nan' or str(right_data[idx, jdx]) == 'nan':
                nan_state = True
        if nan_state is True:
            nan_cleaner -= 1

    foot_dataset = dict()
    foot_dataset['left'] = list()
    foot_dataset['right'] = list()
    for idx in range(nan_cleaner):
        left_list = list()
        right_list = list()
        for jdx in range(sample_col):

            left_list.append(left_data[idx, jdx])
            right_list.append(right_data[idx, jdx])

        left_img = Image.new(mode='L', size=(360, 800), color=0)
        right_img = Image.new(mode='L', size=(360, 800), color=0)

        for sen_idx in range(8):
            left_idx = left_list[sen_idx]
            right_idx = right_list[sen_idx]
            (lw, lh) = sensor_points['left'][sen_idx]
            (rw, rh) = sensor_points['right'][sen_idx]

            left_img = image_pixel_to_eps(img=left_img, width=lw, height=lh, value=left_idx)
            right_img = image_pixel_to_eps(img=right_img, width=rw, height=rh, value=right_idx)

        left_img = np.array(left_img)
        right_img = np.array(right_img)

        left_img = cv2.resize(left_img, dsize=(84, 224), interpolation=cv2.INTER_CUBIC)
        right_img = cv2.resize(right_img, dsize=(84, 224), interpolation=cv2.INTER_CUBIC)

        # left_img = np.asarray(left_img, dtype="int32")
        # left_img = skiresize(image=left_img, output_shape=(84, 224), order=1, mode='reflect', anti_aliasing=False)
        # left_img = toimage(arr=left_img, mode='L')
        # left_img = Image.fromarray(obj=left_img)

        # right_img = np.asarray(right_img, dtype="int32")
        # right_img = skiresize(image=right_img, output_shape=(84, 224), order=1, mode='reflect', anti_aliasing=False)
        # right_img = toimage(arr=right_img, mode='L')
        # right_img = Image.fromarray(obj=right_img)

        foot_dataset['left'].append(left_img)
        foot_dataset['right'].append(right_img)

    return foot_dataset


def gsc_norm(target, smm, vmin, vmax):
    (smin, smax) = smm
    return (target - vmin) * (smax - smin) / (vmax - vmin) + smin


def accgyr_vti(dataset):
    minmax = (-1, 1)
    left_data = dataset[0].values
    right_data = dataset[1].values

    w, h = dataset[0].values.shape
    nan_cleaner = w
    for idx in range(w):
        nan_state = False
        for jdx in range(h):
            if str(left_data[idx, jdx]) == 'nan' or str(right_data[idx, jdx]) == 'nan':
                nan_state = True
        if nan_state is True:
            nan_cleaner -= 1

    left_data = dataset[0].values[:nan_cleaner, :].astype(int)
    right_data = dataset[1].values[:nan_cleaner, :].astype(int)
    if left_data.shape[0] != left_data.shape[0]:
        min_sample = min(left_data.shape[0], right_data.shape[0])
        left_data = left_data[:min_sample]
        right_data = left_data[:min_sample]
    sample_len = left_data.shape[0]
    sample_col = left_data.shape[1]

    if sample_len < 500:
        return 0

    nan_cleaner = sample_len
    for idx in range(sample_len):
        nan_state = False
        for jdx in range(sample_col):
            if str(left_data[idx, jdx]) == 'nan' or str(right_data[idx, jdx]) == 'nan':
                nan_state = True
        if nan_state is True:
            nan_cleaner -= 1

    foot_dataset = dict()
    foot_dataset['left'] = list()
    foot_dataset['right'] = list()

    left_img = np.zeros([nan_cleaner, 3])
    right_img = np.zeros([nan_cleaner, 3])

    for jdx in range(sample_col):
        lmin = min(left_data[:, jdx])
        lmax = max(left_data[:, jdx])
        rmin = min(right_data[:, jdx])
        rmax = max(right_data[:, jdx])
        for idx in range(nan_cleaner):
            left_img[idx, jdx] = gsc_norm(left_data[idx, jdx], minmax, lmin, lmax)
            right_img[idx, jdx] = gsc_norm(right_data[idx, jdx], minmax, rmin, rmax)

    left_img = Image.fromarray(left_img)
    right_img = Image.fromarray(right_img)

    foot_dataset['left'].append(left_img)
    foot_dataset['right'].append(right_img)
    return foot_dataset


def save_vti(dataset, sensor_name, people_nb, class_nb, param):
    folder_dir = '../Datasets/vti'
    if os.path.exists(folder_dir) is not True:
        os.mkdir(folder_dir)

    left_dataset = dataset['left']
    right_dataset = dataset['right']

    for idx, target in enumerate([folder_dir, param.datatype, sensor_name, f'{people_nb}_{class_nb}' ]):
        if idx == 0:
            save_dir = target
            if os.path.exists(save_dir) is not True:
                os.mkdir(save_dir)
        else:
            save_dir = os.path.join(save_dir, target)
            if os.path.exists(save_dir) is not True:
                os.mkdir(save_dir)
    print(f'{idx}: {target}, save directory : {save_dir} state : {os.path.exists(save_dir)}')

    for idx, (left, right) in enumerate(zip(left_dataset, right_dataset)):
        left_dir = os.path.join(save_dir, 'left')
        right_dir = os.path.join(save_dir, 'right')
        if os.path.exists(left_dir) is not True:
            os.mkdir(left_dir)
        if os.path.exists(right_dir) is not True:
            os.mkdir(right_dir)
        cv2.imwrite(os.path.join(left_dir, f'{idx}.png'), left)
        cv2.imwrite(os.path.join(right_dir, f'{idx}.png'), right)


def save_dataset_with_vti(dataset, sensor_name, people_nb, class_nb, param):
    folder_dir = '../Datasets/vti'
    if os.path.exists(folder_dir) is not True:
        os.mkdir(folder_dir)

    left_dataset = dataset['left']
    right_dataset = dataset['right']

    for idx, target in enumerate([folder_dir, param.datatype, sensor_name, f'{people_nb}_{class_nb}' ]):
        if idx == 0:
            save_dir = target
            if os.path.exists(save_dir) is not True:
                os.mkdir(save_dir)
        else:
            save_dir = os.path.join(save_dir, target)
            if os.path.exists(save_dir) is not True:
                os.mkdir(save_dir)
    print(f'{target}, save directory : {save_dir} state : {os.path.exists(save_dir)}')

    for idx, (left, right) in enumerate(zip(left_dataset, right_dataset)):
        left_dir = os.path.join(save_dir, 'left')
        right_dir = os.path.join(save_dir, 'right')
        if os.path.exists(left_dir) is not True:
            os.mkdir(left_dir)
        if os.path.exists(right_dir) is not True:
            os.mkdir(right_dir)
        np.save(os.path.join(left_dir, f'{idx}.npy'), left)
        np.save(os.path.join(right_dir, f'{idx}.npy'), right)
