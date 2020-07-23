import os
import argparse
import json
import time
import datetime

import numpy as np
import tensorflow as tf
from keras.backend import set_session
from keras import backend as K

from Code import loader, preprocessing, model_compactor, visualizer
from Code import create_collector as cc
from Code.utils import dt_printer as dt
from Code.result_collector import column_info, directory, DataStore

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser(description="Gait Analysis Project")
parser.add_argument('--json', type=str, default="create_collector", help='collector file')
parser.add_argument('--Header', type=str, default="200630_type", help='output header')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size default=64')
parser.add_argument('--epochs', type=int, default=20, help='epochs default=20')
args = parser.parse_args()

method_info = {
    "repeat": ['mdpi', 'half', 'dhalf', 'MCCV', 'base'],
    "people": 'LeaveOne',
    "CrossValidation": ['7CV', 'SCV']
}


def get_args(arguments_parser, parameter_store):
    parameter_store.Header = arguments_parser.Header
    parameter_store.batch_size = arguments_parser.batch_size
    parameter_store.epochs = arguments_parser.epochs
    return parameter_store


class SetProject:
    def __init__(self, target):
        self.Header = None
        with open(target, 'r') as f:
            ps = json.load(f)
        self.object = ps["object"]
        self.datatype = ps["dataset"]
        self.method = ps["method"]
        self.folder = ps["folder"]
        self.collect = ps["collect"]
        self.sensor_type = self.collect["sensor_type"]
        self.model_name = ps["model"]
        print(f"{dt()} :: --collected items :{[*self.collect.keys()]}")


def chosen_object(object_name):
    if object_name == "experiment":
        experiment(params, comb_degree=5)
    elif object_name == "create":
        create(params)
    elif object_name == "create_v2":
        create_v2(params)
    elif object_name == "tsne":
        tsne(params, comb_degree=1)
    elif object_name == "visualize":
        visualize(params)
    elif object_name == "create_and_visualize":
        create_and_visualize(params)
    elif object_name == "custom2":
        custom2(params, comb_degree=3)
    elif object_name == "cropping":
        cropping(params)


# experiment
def experiment(param, comb_degree=5):
    print(f"{dt()} :: Experiments Initialize")

    for nb_combine in range(1, comb_degree+1):
        print(f"{dt()} :: {nb_combine} sample experiments")
        param.nb_combine = nb_combine
        if nb_combine != 1:
            continue

        datasets = loader.data_loader(param, target=nb_combine)
        train, test, nb_class, nb_people = preprocessing.chosen_method(param=param, comb=nb_combine, datasets=datasets)
        if param.model_name in model_compactor.model_info['dl']:
            deep_learning_experiment_configuration(param, train, test, [nb_class, nb_people])
            ds.save_result(param)
        elif param.model_name in model_compactor.model_info['ml']:
            machine_learning_experiment_configuration(param, train, test, [nb_class, nb_people])


def deep_learning_experiment_configuration(param, train, test, label_info):
    nb_class = label_info[0]
    nb_people = label_info[1]
    param.nb_modal = 3

    if param.method == method_info['people']:
        nb_repeat = nb_people
    elif param.method in method_info['repeat']:
        nb_repeat = 20
    elif param.method in method_info["CrossValidation"]:
        nb_repeat = param.collect["CrossValidation"] * 5

    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    for repeat in range(nb_repeat):
        # config = tf.compat.v1.ConfigProto()

        sess = tf.Session(config=config)
        # use GPU memory in the available GPU memory capacity

        # sess = tf.compat.v1.Session(config=config)
        set_session(sess)

        print(f"{dt()} :: {repeat+1}/{nb_repeat} experiment progress")

        tartr = train[repeat]
        tarte = test[repeat]

        tr_data = [tartr["data_0"], tartr["data_1"], tartr["data_2"]]
        te_data = [tarte["data_0"], tarte["data_1"], tarte["data_2"]]
        if param.datatype == "type":
            tr_label = tartr["tag"] - 1
            te_label = tarte["tag"] - 1
            nb_class = label_info[0]
        elif param.datatype == "disease":
            tr_label = tartr["tag"]
            te_label = tarte["tag"]
            nb_class = label_info[0]

        cat_tr = preprocessing.to_categorical(tr_label, nb_class)
        cat_te = preprocessing.to_categorical(te_label, nb_class)

        model = model_compactor.model_setting(param, train[repeat], test[repeat], [nb_class, nb_people])
        print(f"{dt()} :: MODEL={param.model_name}, METHOD={param.method}")

        log_dir = f"../Log/{param.model_name}_{param.method}"
        # log_dir = f"/home/blackcow/mlpa/workspace/gait-rework/gait-rework/Log/{param.model_name}_{param.method}"

        tb_hist = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)

        model.summary()
        model_result = model.fit(x=tr_data, y=cat_tr, epochs=param.epochs, batch_size=param.batch_size
                                 , validation_data=(te_data, cat_te), verbose=2, callbacks=[tb_hist])

        model_score = model.evaluate(x=te_data, y=cat_te, verbose=0)


        print(f"{dt()} :: Test Loss :{model_score[0]}")
        print(f"{dt()} :: Test Accuracy :{model_score[1]}")

        if repeat == 0:
            tracking = [dt(), param.method, param.model_name, param.nb_combine, repeat, model_score[0], model_score[1]]
            ds.stock_result(tracking)
        else:
            tracking = [dt(), repeat, model_score[0], model_score[1]]
            ds.stock_result(tracking)

        ds.save_result_obo(param, tracking)

        model_result = None
        model_score = None
        tracking = None
        tr_data = None
        te_date = None
        K.clear_session()
        tf.keras.backend.clear_session()
        sess.close()


def machine_learning_experiment_configuration(param, train, test, label_info):
    nb_class = label_info[0]
    nb_people = label_info[1]
    param.nb_modal = 3

    if param.method == 'LeaveOne':
        nb_repeat = nb_people
    elif param.method in ['mdpi', 'half', 'dhalf', 'MCCV']:
        nb_repeat = 20
    elif param.method in ['7CV', 'SCV']:
        nb_repeat = param.collect["CrossValidation"] * 10

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    for repeat in range(nb_repeat):
        # config = tf.compat.v1.ConfigProto()
        #
        # sess = tf.Session(config=config)
        # # use GPU memory in the available GPU memory capacity
        #
        # # sess = tf.compat.v1.Session(config=config)
        # set_session(sess)

        print(f"{dt()} :: {repeat+1}/{nb_repeat} experiment progress")

        tartr = train[repeat]
        tarte = test[repeat]

        if param.datatype == "type":
            nb_class = label_info[0]
        elif param.datatype == "disease":
            nb_class = label_info[0] + 1

        # Machine Learning Models will need to labeling info.
        model_score = model_compactor.model_setting(param, tartr, tarte, [nb_class, nb_people])
        print(f"{dt()} :: MODEL={param.model_name}, METHOD={param.method}")

        # log_dir = f"../Log/{param.model_name}_{param.method}"
        # log_dir = f"/home/blackcow/mlpa/workspace/gait-rework/gait-rework/Log/{param.model_name}_{param.method}"

        # tb_hist = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)

        # if repeat == 0:
        #     tracking = [dt(), param.method, param.model_name, param.nb_combine, repeat, model_score[0], model_score[1]]
        #     ds.stock_result(tracking)
        # else:
        #     tracking = [dt(), repeat, model_score[0], model_score[1]]
        #     ds.stock_result(tracking)


# create
def create(param):
    print(f"{dt()} :: Create Initialize")
    datasets = loader.create_loader(param)

    for nb_comb in range(1, 6):
        print(f"{dt()} :: Combine Number : {nb_comb}")
        dc = dict()
        for key, files in datasets.items():
            files = sorted(files)
            param.key = key
            class_collect = list()

            for file in files:
                output = create_configuration(param, file, nb_comb)
                class_collect.append(output)
            dc[key] = class_collect
        print(f"{dt()} :: --Combine Number : {nb_comb} Save Initialize")
        cc.save_datasets(param, dc, nb_comb)
        print(f"{dt()} :: --Done")


def create_configuration(param, file, nb_comb):
    df, step_index = cc.dataset_init(param, file)
    sampled = cc.get_index_sampling(param, df, step_index)
    resized = cc.resize_samples(param, sampled)
    combined = cc.combined_samples(resized, comb_degree=nb_comb)
    vectorized = cc.vectorized_samples(param, combined, nb_comb)
    return [file, vectorized]


# create
def create_v2(param):
    print(f"{dt()} :: Create Initialize")
    datasets = loader.create_loader(param)

    for nb_comb in range(1, 6):
        print(f"{dt()} :: Combine Number : {nb_comb}")
        dc = dict()
        for key, files in datasets.items():
            files = sorted(files)
            param.key = key
            class_collect = list()

            for file in files:
                if file.split('/')[-1] == '13_04.csv' or file.split('/')[-1] == '14_05.csv' and \
                        (param.datatype == 'disease' or param.datatype == 'disease_add'):
                    continue
                print(file)
                output = create_configuration_v2(param, file, nb_comb)
                class_collect.append(output)
            dc[key] = class_collect
        print(f"{dt()} :: --Combine Number : {nb_comb} Save Initialize")
        cc.save_datasets(param, dc, nb_comb)
        print(f"{dt()} :: --Done")


def create_configuration_v2(param, file, nb_comb):
    df, step_index = cc.dataset_init_v2(param, file)
    sampled = cc.get_index_sampling_v2(param, df, step_index)
    resized = cc.resize_samples_v2(param, sampled)
    combined = cc.combined_samples_v2(resized, comb_degree=nb_comb)
    vectorized = cc.vectorized_samples_v2(param, combined, nb_comb)
    return [file, vectorized]


# custom : create module + visualize module
def create_and_visualize(param):
    print(f"{dt()} :: Create And Visualize Initialize")
    datasets = loader.create_loader(param)

    for nb_comb in range(1, 6):
        print(f"{dt()} :: --Create Combine Number : {nb_comb}")
        dc = dict()
        for key, files in datasets.items():
            files = sorted(files)
            param.key = key
            class_collect = list()

            for file in files:
                output = create_configuration(param, file, nb_comb)
                class_collect.append(output)
            dc[key] = class_collect

        print(f"{dt()} :: --Combine Number : {nb_comb} Save Initialize")
        cc.save_datasets(param, dc, nb_comb)
        print(f"{dt()} :: --Done")

        if nb_comb == 1:
            print(f"{dt()} :: Visualize Initialize")
            visdat = loader.viz_loader(param)
            print(f"{dt()} :: --Visualize Dataset Path Load")
            visualizer.chosen_viz(param, visdat)
            print(f"{dt()} :: --Done")


# tsne
def tsne(param, comb_degree=1):
    NotImplemented


# custom: cropping network
def cropping(param):
    print(f"{dt()} :: Cropping Network Initialize")
    datasets = loader.create_loader(param)
    data_list = preprocessing.normalize_all_of_length(param, datasets)
    for i, data in enumerate(data_list):
        data[:, -2] = data[:, -2] - 1
        data_list[i] = data[data[:, -2].argsort()]

    param.nb_modal = 3
    train, test, nb_class, nb_people = preprocessing.chosen_method(param=param, comb=1, datasets=data_list)
    nb_repeat = len(train)
    for repeat in range(nb_repeat):
        model = model_compactor.model_setting(param, train[repeat], test[repeat], [nb_class, nb_people])
    print('Done?')


# custom 2 Binary Classification without PA(0)
def custom2(param, comb_degree=3):
    print(f"{dt()} :: Experiments Initialize")

    for nb_combine in range(1, comb_degree+1):
        print(f"{dt()} :: {nb_combine} sample experiments")
        param.nb_combine = nb_combine

        datasets = loader.data_loader(param, target=nb_combine)
        datasets = preprocessing.del_subject(param, datasets, target="PA")
        train, test, nb_class, nb_people = preprocessing.chosen_method(param=param, comb=nb_combine, datasets=datasets)
        deep_learning_experiment_configuration(param, train, test, [nb_class, nb_people])

        ds.save_result(param)


def visualize(param):
    print(f"{dt()} :: Visualize Initialize")
    datasets = loader.viz_loader(param)
    visualizer.chosen_viz(param, datasets)
    NotImplemented


def visualize_configuration():
    NotImplemented


if __name__ == "__main__":
    project_time = time.time()

    # test1 = np.loadtxt('/home/mlpa/Desktop/Untitled Folder/pressure_dataset.dat')
    # test1 = np.load('/home/mlpa/Documents/mlpa/Github_WorkSpace/gait-rework/Datasets/200626_type/Sample_1/pressure_dataset.npy')
    # test2 = np.load('/home/mlpa/Documents/mlpa/Github_WorkSpace/gait-rework/Datasets/200529/Sample_1/pressure_dataset.npy')
    # test3 = np.load('/home/mlpa/Documents/mlpa/pastWorks/CNN/datasets/Tri-Modal/MM_1/pressure_xdatasets.npy')
    # test3_rigi = np.load('/home/mlpa/Documents/mlpa/Python_WorkSpace/190718_PAOA_UserIden/Datasets/Sample_1/pressure_dataset.npy')
    json_file = args.json
    loaddir = f'../Collector/{json_file}.json'
    # loaddir = f'/home/blackcow/mlpa/workspace/gait-rework/gait-rework/Collector/{json_file}.json'

    print(f"{dt()} :: Project Initialize")
    print(f"{dt()} :: Collect Parameter from {loaddir}")

    params = SetProject(target=loaddir)
    params = get_args(args, params)
    ds = DataStore(info=column_info, savedir=directory)

    print(f"{dt()} :: --Header :{params.Header}, Method:{params.method}, Type:{params.datatype}")
    print(f"{dt()} :: --Object :{params.object}")
    chosen_object(params.object)


