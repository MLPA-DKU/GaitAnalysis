import os
import numpy as np
import pandas as pd
import skimage.transform
import cv2
import datetime
from Code.utils import dt_printer as dt


# Init
def dataset_init(param, file):
    data_column = param.collect["csv_columns_names"]["datasets"]
    pressure_column = param.collect["csv_columns_names"]["pressure"]
    left_column = param.collect["csv_columns_names"]["left_pressure"]

    total_dataset = pd.read_csv(filepath_or_buffer=file
                                , names=data_column, header=None, skiprows=1, encoding='utf7')
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


# Sampling
def get_index_sampling(param, data, step_index):
    pressure_column = param.collect["csv_columns_names"]["pressure"]
    acc_column = param.collect["csv_columns_names"]["acc"]
    gyro_column = param.collect["csv_columns_names"]["gyro"]

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
            data = cv2.resize(src=sample, dsize=(sn, mt), interpolation=cv2.INTER_CUBIC)
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
