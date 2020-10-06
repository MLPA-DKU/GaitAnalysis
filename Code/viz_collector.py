import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt


def method_extend(param, dataset):
    mode_size = 1000
    data_column = param.collect["csv_columns_names"]["datasets"]
    pressure_column = param.collect["csv_columns_names"]["pressure"]
    left_column = param.collect["csv_columns_names"]["left_pressure"]

    people_number = 0

    for key, data in dataset.items():

        for pn, target in enumerate(data):
            total_dataset = pd.read_csv(filepath_or_buffer=target
                                        , names=data_column, header=None, skiprows=1)

            if param.collect['latency'] != 100:
                latency = int(100 / param.collect['latency'])
                total_dataset = total_dataset[::latency]
                print(latency, ': latency')

            df = total_dataset[pressure_column]
            unit_step = get_unit_step(total_dataset[left_column])

            narr = df.to_numpy()
            people_name, class_name = target.split('/')[-1].split('_')

            # people_name = int(people_name)
            # class_name = int(class_name)

            dimension = int(narr.shape[0] / mode_size) + 1

            converted = np.zeros((mode_size, narr.shape[1], dimension))
            converted_mask = np.zeros((mode_size, narr.shape[1], dimension))

            mask_vector = np.zeros(narr.shape)
            masking = convert_unit_step(mask_vector, unit_step)

            for dim in range(dimension):
                if narr.shape[0] < mode_size:
                    converted[:narr.shape[0], :, dim] = narr[:, :]
                    converted_mask[:narr.shape[0], :, dim] = masking[:, :]
                elif narr.shape[0] > mode_size and dim + 1 == dimension:
                    converted[:narr.shape[0] % mode_size, :, dim] = narr[mode_size * dim:, :]
                    converted_mask[:narr.shape[0] % mode_size, :, dim] = masking[mode_size * dim:, :]
                elif narr.shape[0] > mode_size and dim + 1 < dimension:
                    converted[:, :, dim] = narr[mode_size*dim:mode_size*(dim + 1), :]
                    converted_mask[:, :, dim] = masking[mode_size*dim:mode_size*(dim + 1), :]

            save_dir = '../Result/Viualizer/'
            folder_dir = os.path.join(save_dir, f'{key}')
            if os.path.exists(save_dir) is not True:
                os.mkdir(save_dir)
            if os.path.exists(folder_dir) is not True:
                os.mkdir(folder_dir)

            vector_list = list()

            for dim in range(dimension):
                vic = convert_vector(converted[:, :, dim], [converted_mask[:, :, dim], converted_mask[:, :, dim]])
                vector_list.append(vic)

            line_arr = np.full((mode_size * 14, 10, 3), fill_value=255, dtype=np.uint8)
            for idx, vec in enumerate(vector_list):
                if idx is 0:
                    total_vec = vec
                else:
                    total_vec = np.hstack((total_vec, line_arr))
                    total_vec = np.hstack((total_vec, vec))

            cv2.imwrite(os.path.join(folder_dir, f'{people_number}.png'), total_vec)
            people_number += 1


def method_extend_mk2(param, dataset):
    mode_size = 1000
    data_column = param.collect["csv_columns_names"]["datasets"]
    pressure_column = param.collect["csv_columns_names"]["pressure"]
    left_column = param.collect["csv_columns_names"]["left_pressure"]
    right_column = param.collect["csv_columns_names"]["right_pressure"]

    people_number = 0

    for key, data in dataset.items():

        for pn, target in enumerate(data):
            total_dataset = pd.read_csv(filepath_or_buffer=target
                                        , names=data_column, header=None, skiprows=1, encoding='utf7')

            df = total_dataset[pressure_column]
            left_unit_step = get_unit_step(total_dataset[left_column])
            right_unit_step = get_unit_step(total_dataset[right_column])

            narr = df.to_numpy()
            people_name, class_name = target.split('/')[-1].split('_')

            # people_name = int(people_name)
            # class_name = int(class_name)

            dimension = int(narr.shape[0] / mode_size) + 1

            converted = np.zeros((mode_size, narr.shape[1], dimension))
            left_converted_mask = np.zeros((mode_size, narr.shape[1], dimension))
            right_converted_mask = np.zeros((mode_size, narr.shape[1], dimension))

            mask_vector = np.zeros(narr.shape)
            left_masking = convert_unit_step(mask_vector.copy(), left_unit_step)
            right_masking = convert_unit_step(mask_vector.copy(), right_unit_step)

            for dim in range(dimension):
                if narr.shape[0] < mode_size:
                    converted[:narr.shape[0], :, dim] = narr[:, :]
                    left_converted_mask[:narr.shape[0], :, dim] = left_masking[:, :]
                elif narr.shape[0] > mode_size and dim + 1 == dimension:
                    converted[:narr.shape[0] % mode_size, :, dim] = narr[mode_size * dim:, :]
                    left_converted_mask[:narr.shape[0] % mode_size, :, dim] = left_masking[mode_size * dim:, :]
                elif narr.shape[0] > mode_size and dim + 1 < dimension:
                    converted[:, :, dim] = narr[mode_size*dim:mode_size*(dim + 1), :]
                    left_converted_mask[:, :, dim] = left_masking[mode_size*dim:mode_size*(dim + 1), :]

            for dim in range(dimension):
                if narr.shape[0] < mode_size:
                    converted[:narr.shape[0], :, dim] = narr[:, :]
                    right_converted_mask[:narr.shape[0], :, dim] = right_masking[:, :]
                elif narr.shape[0] > mode_size and dim + 1 == dimension:
                    converted[:narr.shape[0] % mode_size, :, dim] = narr[mode_size * dim:, :]
                    right_converted_mask[:narr.shape[0] % mode_size, :, dim] = right_masking[mode_size * dim:, :]
                elif narr.shape[0] > mode_size and dim + 1 < dimension:
                    converted[:, :, dim] = narr[mode_size*dim:mode_size*(dim + 1), :]
                    right_converted_mask[:, :, dim] = right_masking[mode_size*dim:mode_size*(dim + 1), :]

            save_dir = '../Result/Viualizer/'
            # folder_dir = os.path.join(save_dir, f'{key}')
            # folder_dir = os.path.join(save_dir, f'{key}', f'{pn}')

            for e, added in enumerate([save_dir, 'extended_mk2', f'class{key}', f'pn{pn}']):
                if e == 0:
                    folder_dir = added
                else:
                    folder_dir = os.path.join(folder_dir, added)
                if os.path.exists(folder_dir) is not True:
                    os.mkdir(folder_dir)

            vector_list = list()

            for dim in range(dimension):
                mask = [left_converted_mask[:, :, dim], right_converted_mask[:, :, dim]]
                vic = convert_vector(converted[:, :, dim], mask)
                vector_list.append(vic)

            line_arr = np.full((mode_size * 14, 10, 3), fill_value=255, dtype=np.uint8)
            for idx, vec in enumerate(vector_list):
                if idx is 0:
                    total_vec = vec
                else:
                    total_vec = np.hstack((total_vec, line_arr))
                    total_vec = np.hstack((total_vec, vec))

            cv2.imwrite(os.path.join(folder_dir, f'{people_number}.png'), total_vec)
            people_number += 1


def method_create(param, dataset):
    mode_size = 1000
    data_column = param.collect["csv_columns_names"]["datasets"]
    pressure_column = param.collect["csv_columns_names"]["pressure"]
    left_column = param.collect["csv_columns_names"]["left_pressure"]

    pl = 0
    # mapping labeling
    keymap = dict()
    for key, data in dataset.items():
        # data = sorted(data)
        for pn, target in enumerate(data):
            total_dataset = pd.read_csv(filepath_or_buffer=target
                                        , names=data_column, header=None, skiprows=1)
            df = total_dataset[pressure_column]
            unit_step = get_unit_step(total_dataset[left_column])

            narr = df.to_numpy()
            people_name, class_name = target.split('/')[-1].rstrip('.csv').split('_')
            # class_name = int(class_name)

            keymap[target.split('/')[-1].rstrip('.csv')] = (key, pl, class_name)
            # people_name = int(people_name)

            dimension = int(narr.shape[0] / mode_size) + 1

            converted = np.zeros((mode_size, narr.shape[1], dimension))
            converted_mask = np.zeros((mode_size, narr.shape[1], dimension))

            mask_vector = np.zeros(narr.shape)
            masking = convert_unit_step(mask_vector, unit_step)

            for dim in range(dimension):
                if narr.shape[0] < mode_size:
                    converted[:narr.shape[0], :, dim] = narr[:, :]
                    converted_mask[:narr.shape[0], :, dim] = masking[:, :]
                elif narr.shape[0] > mode_size and dim + 1 == dimension:
                    converted[:narr.shape[0] % mode_size, :, dim] = narr[mode_size * dim:, :]
                    converted_mask[:narr.shape[0] % mode_size, :, dim] = masking[mode_size * dim:, :]
                elif narr.shape[0] > mode_size and dim + 1 < dimension:
                    converted[:, :, dim] = narr[mode_size*dim:mode_size*(dim + 1), :]
                    converted_mask[:, :, dim] = masking[mode_size*dim:mode_size*(dim + 1), :]

            save_dir = f"../Result/Viualizer/{param.folder}"
            folder_dir = os.path.join(save_dir, f'{key}')
            if os.path.exists(save_dir) is not True:
                os.mkdir(save_dir)
            if os.path.exists(folder_dir) is not True:
                os.mkdir(folder_dir)

            vector_list = list()

            for dim in range(dimension):
                vic = convert_vector(converted[:, :, dim], converted_mask[:, :, dim])
                vector_list.append(vic)

            line_arr = np.full((mode_size * 14, 10, 3), fill_value=255, dtype=np.uint8)
            for idx, vec in enumerate(vector_list):
                if idx is 0:
                    total_vec = vec
                else:
                    total_vec = np.hstack((total_vec, line_arr))
                    total_vec = np.hstack((total_vec, vec))

            cv2.imwrite(os.path.join(folder_dir, f"p{pl}_c{int(class_name)}.png"), total_vec)
            pl += 1
    f = open(os.path.join(save_dir, 'keymap.txt'), 'w')
    f.write(str(keymap))
    f.close()


def convert_vector(data, mask):
    wid, hei = data.shape
    vector = np.zeros((wid * 14, hei * 14, 3), dtype=np.uint8)

    for w in range(wid):
        for h in range(int(hei/2)):
            if data[w, h] == 0:
                vector[w*14:(w+1)*14, h*14:(h+1)*14, :] = (0, 0, 0)
            elif data[w, h] == 1:
                vector[w*14:(w+1)*14, h*14:(h+1)*14, :] = (0, 0, 128)
            elif data[w, h] == 2:
                vector[w*14:(w+1)*14, h*14:(h+1)*14, :] = (0, 0, 255)

        for h in range(int(hei/2), hei):
            if data[w, h] == 0:
                vector[w*14:(w+1)*14, h*14:(h+1)*14, :] = (0, 0, 0)
            elif data[w, h] == 1:
                vector[w*14:(w+1)*14, h*14:(h+1)*14, :] = (128, 0, 0)
            elif data[w, h] == 2:
                vector[w*14:(w+1)*14, h*14:(h+1)*14, :] = (255, 0, 0)

    for w in range(wid):
        for h in range(int(hei/2)):
            if mask[0][w, h] == 0 and data[w, h] == 0:
                continue
            elif mask[0][w, h] == 1 and data[w, h] == 1:
                continue
            elif mask[0][w, h] == 2:
                vector[w * 14:(w + 1) * 14, h * 14:(h + 1) * 14, :] = (0, 255, 0)

        for h in range(int(hei/2), hei):
            if mask[1][w, h] == 0 and data[w, h] == 0:
                continue
            elif mask[1][w, h] == 1 and data[w, h] == 1:
                continue
            elif mask[1][w, h] == 2:
                vector[w * 14:(w + 1) * 14, h * 14:(h + 1) * 14, :] = (0, 255, 0)
    return vector


def method_normalized(param, dataset):
    mode_size = 1000
    data_column = param.collect["csv_columns_names"]["datasets"]
    pressure_column = param.collect["csv_columns_names"]["pressure"]
    left_column = param.collect["csv_columns_names"]["left_pressure"]

    people_number = 0

    for key, data in dataset.items():

        for pn, target in enumerate(data):
            total_dataset = pd.read_csv(filepath_or_buffer=target
                                        , names=data_column, header=None, skiprows=1)
            df = total_dataset[pressure_column]
            narr = df.to_numpy()
            unit_step = get_unit_step(total_dataset[left_column])
            resized_list = get_index_sampling(param, [narr], unit_step)
            narr = get_index_vectorized(resized_list)
            # narr = resized_list[0]
            people_name, class_name = target.split('/')[-1].split('_')

            # people_name = int(people_name)
            # class_name = int(class_name)

            dimension = int(narr.shape[0] / mode_size) + 1

            converted = np.zeros((mode_size, narr.shape[1], dimension))
            converted_mask = np.zeros((mode_size, narr.shape[1], dimension))

            mask_vector = np.zeros(narr.shape)
            masking = convert_resized_step(mask_vector, 63)

            for dim in range(dimension):
                if narr.shape[0] < mode_size:
                    converted[:narr.shape[0], :, dim] = narr[:, :]
                    converted_mask[:narr.shape[0], :, dim] = masking[:, :]
                elif narr.shape[0] > mode_size and dim + 1 == dimension:
                    converted[:narr.shape[0] % mode_size, :, dim] = narr[mode_size * dim:, :]
                    converted_mask[:narr.shape[0] % mode_size, :, dim] = masking[mode_size * dim:, :]
                elif narr.shape[0] > mode_size and dim + 1 < dimension:
                    converted[:, :, dim] = narr[mode_size*dim:mode_size*(dim + 1), :]
                    converted_mask[:, :, dim] = masking[mode_size*dim:mode_size*(dim + 1), :]

            save_dir = '../Result/Viualizer/'
            folder_dir = os.path.join(save_dir, f'resized_{key}')
            if os.path.exists(save_dir) is not True:
                os.mkdir(save_dir)
            if os.path.exists(folder_dir) is not True:
                os.mkdir(folder_dir)

            vector_list = list()

            for dim in range(dimension):
                vic = convert_vector(converted[:, :, dim], converted_mask[:, :, dim])
                vector_list.append(vic)

            line_arr = np.full((mode_size * 14, 10, 3), fill_value=255, dtype=np.uint8)
            for idx, vec in enumerate(vector_list):
                if idx is 0:
                    total_vec = vec
                else:
                    total_vec = np.hstack((total_vec, line_arr))
                    total_vec = np.hstack((total_vec, vec))

            cv2.imwrite(os.path.join(folder_dir, f'{people_number}.png'), total_vec)
            people_number += 1


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


def convert_unit_step(vector, unit_list):
    for index in range(0, len(unit_list)-1):
        vector[unit_list[index]:unit_list[index + 1], :] = 1
    for index in range(0, len(unit_list) - 1):
        vector[unit_list[index], :] = 2

    return vector


def convert_resized_step(mask_vector, m=63):
    for index in range(0, mask_vector.shape[0]):
        if index % (64-1) == 0:
            mask_vector[index, :] = 1
        else:
            mask_vector[index, :] = 1

    return mask_vector


def convert_sampling(param, data, unit_list):
    samples = get_index_sampling(param, data, unit_list)
    resized_list = list()
    for sample in samples:
        resized = resize_samples(param, sample)
        resized_array = np.array([])
        for i, temp in enumerate(resized):
            if i == 0:
                resized_array = temp
            else:
                resized_array = np.vstack((resized_array, temp))

        resized_list.append(resized_array)

    return resized_list


# Sampling
def get_index_sampling(param, data, step_index):
    # pressure_column = param.collect["csv_columns_names"]["pressure"]
    # acc_column = param.collect["csv_columns_names"]["acc"]
    # gyro_column = param.collect["csv_columns_names"]["gyro"]

    # pressure = data[pressure_column]
    # acc = data[acc_column]
    # gyro = data[gyro_column]

    out_p = list()
    # out_a = list()
    # out_g = list()
    pressure = data

    for idx in range(0, len(step_index) - 1):
        pressure_loc = pressure[0][step_index[idx]:step_index[idx + 1]]
        # acc_loc = acc.iloc[step_index[idx]:step_index[idx + 1]]
        # gyro_loc = gyro.iloc[step_index[idx]:step_index[idx + 1]]

        out_p.append(pressure_loc)
        # out_a.append(acc_loc)
        # out_g.append(gyro_loc)

    # return [out_p, out_a, out_g]
    return out_p


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


def get_index_vectorized(dataset):
    vec = np.array([])
    for i, data in enumerate(dataset):
        if i == 0 :
            vec = data
        else:
            vec = np.vstack((vec, data))

    return vec


def method_mapping(param, dataset):
    data_column = param.collect["csv_columns_names"]["datasets"]
    pressure_column = param.collect["csv_columns_names"]["pressure"]
    acc_column = param.collect["csv_columns_names"]["acc"]
    gyro_column = param.collect["csv_columns_names"]["gyro"]
    left_column = param.collect["csv_columns_names"]["left_pressure"]
    right_column = param.collect["csv_columns_names"]["right_pressure"]

    pl = 0
    # mapping labeling
    keymap = dict()
    for key, data in dataset.items():

        for pn, target in enumerate(data):
            total_dataset = pd.read_csv(filepath_or_buffer=target
                                        , names=data_column, header=None, skiprows=1, encoding='utf7')

            peo_nb = int(target.split('/')[-1].split('_')[0])
            pre = np.array(total_dataset[pressure_column].to_numpy())
            acc = np.array(total_dataset[acc_column].to_numpy())
            gyro = np.array(total_dataset[gyro_column].to_numpy())

            left_unit_step = get_unit_step(total_dataset[left_column])
            right_unit_step = get_unit_step(total_dataset[right_column])

            print("Done?")
