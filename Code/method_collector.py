from random import random, seed, sample
import numpy as np
import datetime
import time
import Code.preprocessing as pp


def remove_subject(rsub):
    pn_list = list()
    for target in rsub:
        pn, cn = target.endswith('.csv').spliat('_')
        pn_list.append((pn, cn))
    return pn_list


def method_base(param, comb, datasets):
    BaseDivideProcess(param.method, param.model_name, dataset=datasets)
    if param.datatype == "disease":
        BaseDivideProcess.nb_class += 1

    divide_process = baseDP(param.method, param.model_name, dataset=datasets, rsub=None)

    sampling_data = divide_process.sampling()

    sample_train = sampling_data["train"]
    sample_test = sampling_data["test"]

    for repeat in range(20):
        train = sample_train[repeat]
        test = sample_test[repeat]

        for nb in range(3):
            train[f"data_{nb}"] = divide_process.convert(data=train[f"data_{nb}"],
                                                         mt=param.collect["minimum_threshold"], comb=comb)
            test[f"data_{nb}"] = divide_process.convert(data=test[f"data_{nb}"],
                                                        mt=param.collect["minimum_threshold"], comb=comb)
        sample_train[repeat] = train
        sample_test[repeat] = test

    nb_tag = divide_process.nb_class
    nb_people = divide_process.nb_people

    return sample_train, sample_test, nb_tag, nb_people


def method_sn(param, comb, datasets):
    BaseDivideProcess(param.method, param.model_name, dataset=datasets)
    if param.datatype == "disease":
        BaseDivideProcess.nb_class += 1

    divide_process = snDP(param.method, param.model_name, dataset=datasets)

    sampling_data = divide_process.sampling()

    sample_train = sampling_data["train"]
    sample_test = sampling_data["test"]

    for repeat in range(20):
        train = sample_train[repeat]
        test = sample_test[repeat]

        for nb in range(3):
            train[f"data_{nb}"] = divide_process.convert(data=train[f"data_{nb}"],
                                                         mt=param.collect["minimum_threshold"], comb=comb)
            test[f"data_{nb}"] = divide_process.convert(data=test[f"data_{nb}"],
                                                        mt=param.collect["minimum_threshold"], comb=comb)
        sample_train[repeat] = train
        sample_test[repeat] = test

    nb_tag = divide_process.nb_class
    nb_people = divide_process.nb_people

    return sample_train, sample_test, nb_tag, nb_people


def method_leaveone(param, comb, datasets):
    BaseDivideProcess(param.method, param.model_name, dataset=datasets)
    if param.method == "cropping" or param.method == "convert":
        divide_process = LeaveOneDP_ns(param.method, param.model_name, dataset=datasets, rsub=None)
        tot_repeat = divide_process.nb_people
        if param.datatype == "disease":
            divide_process.nb_class += 1
    elif param.method == "sleaveone":
        divide_process = LeaveOneDP_select(param.method, param.model_name, dataset=datasets, rsub=None)
        tot_repeat = 20
    else:
        divide_process = LeaveOneDP(param.method, param.model_name, dataset=datasets, rsub=None)
        tot_repeat = divide_process.nb_people
        if param.datatype == "disease":
            divide_process.nb_class += 1
    sampling_data = divide_process.sampling()

    sample_train = sampling_data["train"]
    sample_test = sampling_data["test"]

    for repeat in range(tot_repeat):
        train = sample_train[repeat]
        test = sample_test[repeat]

        for nb in range(3):
            train[f"data_{nb}"] = divide_process.convert(data=train[f"data_{nb}"],
                                                         mt=param.collect["minimum_threshold"], comb=comb)
            test[f"data_{nb}"] = divide_process.convert(data=test[f"data_{nb}"],
                                                        mt=param.collect["minimum_threshold"], comb=comb)

        sample_train[repeat] = train
        sample_test[repeat] = test

    nb_tag = divide_process.nb_class
    nb_people = divide_process.nb_people

    return sample_train, sample_test, nb_tag, nb_people


def method_fa_leaveone(param, comb, datasets):
    BaseDivideProcess(param.method, param.model_name, dataset=datasets)
    if param.datatype == "disease":
        BaseDivideProcess.nb_class += 1
    divide_process = LeaveOneDP(param.method, param.model_name, dataset=datasets
                                , rsub=None)

    sampling_data = divide_process.sampling()

    sample_train = sampling_data["train"]
    sample_test = sampling_data["test"]

    for repeat in range(divide_process.nb_people):
        train = sample_train[repeat]
        test = sample_test[repeat]

        for nb in range(3):
            train[f"data_{nb}"] = divide_process.convert(data=train[f"data_{nb}"],
                                                         mt=param.collect["minimum_threshold"], comb=comb)
            test[f"data_{nb}"] = divide_process.convert(data=test[f"data_{nb}"],
                                                        mt=param.collect["minimum_threshold"], comb=comb)

        sample_train[repeat] = train
        sample_test[repeat] = test

    nb_tag = divide_process.nb_class
    nb_people = divide_process.nb_people

    return sample_train, sample_test, nb_tag, nb_people


def method_mdpi(param, comb, datasets):
    BaseDivideProcess(param.method, param.model_name, dataset=datasets)
    if param.datatype == "disease":
        BaseDivideProcess.nb_class += 1
    divide_process = mdpiDP(param.method, param.model_name, dataset=datasets
                            , rsub=None)

    sampling_data = divide_process.sampling()

    sample_train = sampling_data["train"]
    sample_test = sampling_data["test"]

    for repeat in range(20):
        train = sample_train[repeat]
        test = sample_test[repeat]

        for nb in range(3):
            train[f"data_{nb}"] = divide_process.convert(data=train[f"data_{nb}"],
                                                         mt=param.collect["minimum_threshold"], comb=comb)
            test[f"data_{nb}"] = divide_process.convert(data=test[f"data_{nb}"],
                                                        mt=param.collect["minimum_threshold"], comb=comb)

        sample_train[repeat] = train
        sample_test[repeat] = test

    nb_tag = divide_process.nb_class
    nb_people = divide_process.nb_people

    return sample_train, sample_test, nb_tag, nb_people


def method_smdpi(param, comb, datasets):
    BaseDivideProcess(param.method, param.model_name, dataset=datasets)
    divide_process = mdpiDP(param.method, param.model_name, dataset=datasets
                            , rsub=None)

    sampling_data = divide_process.sampling(s1=param.collect["select"][0], s2=param.collect["select"][1])

    sample_train = sampling_data["train"]
    sample_test = sampling_data["test"]

    for repeat in range(20):
        train = sample_train[repeat]
        test = sample_test[repeat]

        for nb in range(3):
            train[f"data_{nb}"] = divide_process.convert(data=train[f"data_{nb}"],
                                                         mt=param.collect["minimum_threshold"], comb=comb)
            test[f"data_{nb}"] = divide_process.convert(data=test[f"data_{nb}"],
                                                        mt=param.collect["minimum_threshold"], comb=comb)

        sample_train[repeat] = train
        sample_test[repeat] = test

    nb_tag = divide_process.nb_class
    nb_people = divide_process.nb_people

    return sample_train, sample_test, nb_tag, nb_people


def method_dhalf(param, comb, datasets):
    BaseDivideProcess(param.method, param.model_name, dataset=datasets)
    if param.datatype == "disease":
        BaseDivideProcess.nb_class += 1
    divide_process = mdpi_dhalfDP(param.method, param.model_name, dataset=datasets
                                , rsub=None)

    sampling_data = divide_process.sampling()

    sample_train = sampling_data["train"]
    sample_test = sampling_data["test"]

    for repeat in range(20):
        train = sample_train[repeat]
        test = sample_test[repeat]

        for nb in range(3):
            train[f"data_{nb}"] = divide_process.convert(data=train[f"data_{nb}"],
                                                         mt=param.collect["minimum_threshold"], comb=comb)
            test[f"data_{nb}"] = divide_process.convert(data=test[f"data_{nb}"],
                                                        mt=param.collect["minimum_threshold"], comb=comb)

        sample_train[repeat] = train
        sample_test[repeat] = test

    nb_tag = divide_process.nb_class
    nb_people = divide_process.nb_people

    return sample_train, sample_test, nb_tag, nb_people


def method_half(param, comb, datasets):
    BaseDivideProcess(param.method, param.model_name, dataset=datasets)
    if param.datatype == "disease":
        BaseDivideProcess.nb_class += 1
    divide_process = mdpi_halfDP(param.method, param.model_name, dataset=datasets
                                , rsub=None)

    sampling_data = divide_process.sampling()

    sample_train = sampling_data["train"]
    sample_test = sampling_data["test"]

    for repeat in range(20):
        train = sample_train[repeat]
        test = sample_test[repeat]

        for nb in range(3):
            train[f"data_{nb}"] = divide_process.convert(data=train[f"data_{nb}"],
                                                         mt=param.collect["minimum_threshold"], comb=comb)
            test[f"data_{nb}"] = divide_process.convert(data=test[f"data_{nb}"],
                                                        mt=param.collect["minimum_threshold"], comb=comb)

        sample_train[repeat] = train
        sample_test[repeat] = test

    nb_tag = divide_process.nb_class
    nb_people = divide_process.nb_people

    return sample_train, sample_test, nb_tag, nb_people


def method_MCCV(param, comb, datasets):
    BaseDivideProcess(param.method, param.model_name, dataset=datasets)
    if param.datatype == "disease":
        BaseDivideProcess.nb_class += 1
    divide_process = mdpi_MCCVDP(param.method, param.model_name, dataset=datasets
                                , rsub=None)

    sampling_data = divide_process.sampling()

    sample_train = sampling_data["train"]
    sample_test = sampling_data["test"]

    for repeat in range(20):
        train = sample_train[repeat]
        test = sample_test[repeat]

        for nb in range(3):
            train[f"data_{nb}"] = divide_process.convert(data=train[f"data_{nb}"],
                                                         mt=param.collect["minimum_threshold"], comb=comb)
            test[f"data_{nb}"] = divide_process.convert(data=test[f"data_{nb}"],
                                                        mt=param.collect["minimum_threshold"], comb=comb)

        sample_train[repeat] = train
        sample_test[repeat] = test

    nb_tag = divide_process.nb_class
    nb_people = divide_process.nb_people

    return sample_train, sample_test, nb_tag, nb_people


def method_CV(param, comb, datasets):
    BaseDivideProcess(param.method, param.model_name, dataset=datasets)
    if param.datatype == "disease":
        BaseDivideProcess.nb_class += 1
    if param.collect["CrossValidation"] == 7:
        divide_process = seven_CVDP(param.method, param.model_name, dataset=datasets
                                    , rsub=None)
    else:
        param.cv_ratio = param.collect["CrossValidation"]
        divide_process = select_CVDP(param.method, param.model_name, dataset=datasets
                                    , rsub=None)

    sampling_data = divide_process.sampling()

    sample_train = sampling_data["train"]
    sample_test = sampling_data["test"]

    for repeat in range(len(sample_train)):
        train = sample_train[repeat]
        test = sample_test[repeat]

        for nb in range(3):
            train[f"data_{nb}"] = divide_process.convert(data=train[f"data_{nb}"],
                                                         mt=param.collect["minimum_threshold"], comb=comb)
            test[f"data_{nb}"] = divide_process.convert(data=test[f"data_{nb}"],
                                                        mt=param.collect["minimum_threshold"], comb=comb)

        sample_train[repeat] = train
        sample_test[repeat] = test

    nb_tag = divide_process.nb_class
    nb_people = divide_process.nb_people

    return sample_train, sample_test, nb_tag, nb_people


def method_vec(param, comb, datasets):

    BaseDivideProcess(param.method, param.model_name, dataset=datasets)
    if param.datatype == "disease":
        BaseDivideProcess.nb_class += 1
    divide_process = NotImplemented
    sampling_data = divide_process.sampling()

    sample_train = sampling_data["train"]
    sample_test = sampling_data["test"]

    for repeat in range(len(sample_train)):
        train = sample_train[repeat]
        test = sample_test[repeat]

        for nb in range(3):
            train[f"data_{nb}"] = divide_process.convert(data=train[f"data_{nb}"],
                                                         mt=param.collect["minimum_threshold"], comb=comb)
            test[f"data_{nb}"] = divide_process.convert(data=test[f"data_{nb}"],
                                                        mt=param.collect["minimum_threshold"], comb=comb)

        sample_train[repeat] = train
        sample_test[repeat] = test

    nb_tag = divide_process.nb_class
    nb_people = divide_process.nb_people

    return sample_train, sample_test, nb_tag, nb_people


# Base Divide Process Class
class BaseDivideProcess:
    def __init__(self, mode, model_name, dataset):
        assert len(dataset) == 3, "dataset must be 3 arguments"
        data1, data2, data3 = dataset

        # [data1, data2, data3] = pp.sort_by_people(dataset)
        data1 = data1[data1[:, -2].argsort()]
        data2 = data2[data2[:, -2].argsort()]
        data3 = data3[data3[:, -2].argsort()]
        # sampling func name
        self.mode = mode
        # used model name
        self.model_name = model_name

        self.dataset = dataset
        self.plabel = data1[:, -2]
        self.tlabel = data1[:, -1]

        # dataset index
        self.data1 = data1[:, :-2]
        self.data2 = data2[:, :-2]
        self.data3 = data3[:, :-2]

        self.nb_class = int(max(self.tlabel))
        self.nb_people = int(max(self.plabel)) + 1

    def sampling(self):
        pass

    def convert(self, data, mt, comb):
        drow, dcol = data.shape
        input_shape = (int(mt * comb), int((dcol) / (mt * comb)))
        if self.model_name in method_info['4columns']:
            converted = data.reshape(-1, input_shape[0], input_shape[1], 1)
        elif self.model_name == "pVGG":
            data = data.reshape(-1, input_shape[0], input_shape[1])
            converted = np.zeros((data.shape[0], data.shape[1], data.shape[2], 3))
            for idx in range(3):
                converted[:, :, :, idx] = data
        elif self.model_name in method_info['3columns']:
            converted = data.reshape(-1, input_shape[0], input_shape[1])
        elif self.model_name in method_info['2columns']:
            converted = data
        elif self.model_name in method_info['specific']:
            converted = data
        elif self.model_name in method_info['vector']:
            converted = data
        elif self.model_name in method_info['5columns']:
            if input_shape[1] == 6:
                left_data = data[:, :3]
                right_data = data[:, 3:]
                converted = [left_data.reshape(-1, input_shape[0], 3), right_data.reshape(-1, input_shape[0], 3)]
        return converted


# 1000, 1000 sampling Class
class baseDP(BaseDivideProcess):
    """
        Sn 600-900 sampling
    """
    def __init__(self, mode, model_name, dataset, rsub):
        super().__init__(mode, model_name, dataset)
        print(f"{datetime.datetime.now()} :: Divide Process : {self.mode}")

    def sampling(self):

        total_dataset = dict()
        total_dataset["train"] = list()
        total_dataset["test"] = list()

        for repeat in range(20):
            seed(repeat)
            drow, _ = self.data1.shape

            train_dict = dict()
            test_dict = dict()
            dataset_list = list()
            random_list = sample(range(drow), drow)

            for dataset in [self.data1, self.data2, self.data3]:
                dataset_list.append(dataset[random_list])

            targetp = self.plabel[random_list]
            targetc = self.tlabel[random_list]

            for i, dataset in enumerate(dataset_list):
                train_dict[f"data_{i}"] = dataset[:1000, :]
                test_dict[f"data_{i}"] = dataset[1000:2000, :]
            train_dict["people"] = targetp[:1000]
            train_dict["tag"] = targetc[:1000]
            test_dict["people"] = targetp[1000:2000]
            test_dict["tag"] = targetc[1000:2000]

            total_dataset["train"].append(train_dict)
            total_dataset["test"].append(test_dict)

        return total_dataset


# 600-900 sampling Class
class snDP(BaseDivideProcess):
    """
        Sn 600-900 sampling
    """
    def __init__(self, mode, model_name, dataset, rsub):
        super().__init__(mode, model_name, dataset)
        print(f"{datetime.datetime.now()} :: Divide Process : {self.mode}")

    def sampling(self):

        total_dataset = dict()
        total_dataset["train"] = list()
        total_dataset["test"] = list()

        for repeat in range(20):
            seed(repeat)
            drow, _ = self.data1.shape

            train_dict = dict()
            test_dict = dict()

            for class_target in range(self.nb_class):
                find_idx = []
                count_idx = 0
                for idx in range(drow):
                    if self.tlabel[idx] == class_target:
                        find_idx.append(idx)
                        count_idx += 1

                dataset_list = list()
                for dataset in [self.data1, self.data2, self.data3]:
                    target = dataset[find_idx[0]:find_idx[-1] + 1, :]
                    dataset_list.append(target)

                targetp = self.plabel[find_idx[0]:find_idx[-1] + 1]
                targetc = self.tlabel[find_idx[0]:find_idx[-1] + 1]

                random_list = sample(range(count_idx), count_idx)

                for i, target in enumerate(dataset_list):
                    dataset_list[i] = target[random_list]
                targetp = targetp[random_list]
                targetc = targetc[random_list]

                if class_target == 0:
                    for i, dataset in enumerate(dataset_list):
                        train_dict[f"data_{i}"] = dataset[:200, :]
                        test_dict[f"data_{i}"] = dataset[200:, :]
                    train_dict["people"] = targetp[:200]
                    train_dict["tag"] = targetc[:200]
                    test_dict["people"] = targetp[200:]
                    test_dict["tag"] = targetc[200:]
                else:
                    for i, dataset in enumerate(dataset_list):
                        train_dict[f"data_{i}"] = np.vstack([train_dict[f"data_{i}"], dataset[:200, :]])
                        test_dict[f"data_{i}"] = np.vstack([test_dict[f"data_{i}"], dataset[200:, :]])
                    train_dict["people"] = np.concatenate([train_dict["people"], targetp[:200]])
                    train_dict["tag"] = np.concatenate([train_dict["tag"], targetc[:200]])
                    test_dict["people"] = np.concatenate([test_dict["people"], targetp[200:]])
                    test_dict["tag"] = np.concatenate([test_dict["tag"], targetc[200:]])

            other_samples, _ = train_dict["data_0"].shape
            random_list = sample(range(other_samples), 600)
            train_dict["people"] = train_dict["people"][random_list]
            train_dict["tag"] = train_dict["tag"][random_list]
            for i in range(3):
                train_dict[f"data_{i}"] = train_dict[f"data_{i}"][random_list]

            other_samples, _ = test_dict["data_0"].shape
            random_list = sample(range(other_samples), 900)
            test_dict["people"] = test_dict["people"][random_list]
            test_dict["tag"] = test_dict["tag"][random_list]
            for i in range(3):
                test_dict[f"data_{i}"] = test_dict[f"data_{i}"][random_list]

            total_dataset["train"].append(train_dict)
            total_dataset["test"].append(test_dict)

        return total_dataset


# LeaveOne sampling Class
class LeaveOneDP(BaseDivideProcess):
    """
        LeaveOne sampling
    """
    def __init__(self, mode, model_name, dataset, rsub):
        super().__init__(mode, model_name, dataset)
        print(f"{datetime.datetime.now()} :: Divide Process : {self.mode}")

    def sampling(self):
        total_dataset = dict()
        total_dataset["train"] = list()
        total_dataset["test"] = list()

        for peo_target in range(self.nb_people):
            train_dict = dict()
            test_dict = dict()

            dataset_list = list()
            train_list = list()

            find_idx = []
            count_idx = 0
            drow, _ = self.data1.shape

            for idx in range(drow):
                if self.plabel[idx] == peo_target:
                    find_idx.append(idx)
                    count_idx += 1

            for dataset in [self.data1, self.data2, self.data3]:
                target = dataset[find_idx[0]:find_idx[-1] + 1, :]

                if find_idx[0] == 0:
                    train = dataset[find_idx[-1] + 1:, :]
                elif find_idx[0] != 0 and find_idx[-1] + 1 != drow:
                    temp1 = dataset[:find_idx[0], :]
                    temp2 = dataset[find_idx[-1] + 1:, :]
                    train = np.vstack([temp1, temp2])
                elif find_idx[-1] + 1 == drow:
                    train = dataset[:find_idx[-1] + 1, :]

                dataset_list.append(target)
                train_list.append(train)

            targetp = self.plabel[find_idx[0]:find_idx[-1] + 1]
            targetc = self.tlabel[find_idx[0]:find_idx[-1] + 1]

            if find_idx[0] == 0:
                trainp = self.plabel[find_idx[-1] + 1:]
                trainc = self.tlabel[find_idx[-1] + 1:]
            elif find_idx[0] != 0 and find_idx[-1] + 1 != drow:
                temp1 = self.plabel[:find_idx[0]]
                temp2 = self.plabel[find_idx[-1] + 1:]
                trainp = np.concatenate([temp1, temp2])

                temp1 = self.tlabel[:find_idx[0]]
                temp2 = self.tlabel[find_idx[-1] + 1:]
                trainc = np.concatenate([temp1, temp2])
            elif find_idx[-1] + 1 == drow:
                trainp = self.plabel[:find_idx[-1] + 1]
                trainc = self.tlabel[:find_idx[-1] + 1]

            target_indexes, _ = dataset_list[0].shape
            train_indexes, _ = train_list[0].shape
            random_list1 = sample(range(target_indexes), target_indexes)
            random_list2 = sample(range(train_indexes), train_indexes)

            for i, dataset in enumerate(dataset_list):
                test_dict[f"data_{i}"] = dataset[random_list1]
            test_dict["people"] = targetp[random_list1]
            test_dict["tag"] = targetc[random_list1]

            for i, dataset in enumerate(train_list):
                train_dict[f"data_{i}"] = dataset[random_list2]
            train_dict["people"] = trainp[random_list2]
            train_dict["tag"] = trainc[random_list2]

            total_dataset["train"].append(train_dict)
            total_dataset["test"].append(test_dict)

        return total_dataset


# LeaveOne sampling Class no shuffle
class LeaveOneDP_ns(BaseDivideProcess):
    """
        LeaveOne sampling
    """
    def __init__(self, mode, model_name, dataset, rsub):
        super().__init__(mode, model_name, dataset)
        print(f"{datetime.datetime.now()} :: Divide Process : {self.mode}")

    def sampling(self):
        total_dataset = dict()
        total_dataset["train"] = list()
        total_dataset["test"] = list()

        for peo_target in range(self.nb_people):
            train_dict = dict()
            test_dict = dict()

            dataset_list = list()
            train_list = list()

            find_idx = []
            count_idx = 0
            drow, _ = self.data1.shape

            for idx in range(drow):
                if self.plabel[idx] == peo_target:
                    find_idx.append(idx)
                    count_idx += 1

            for dataset in [self.data1, self.data2, self.data3]:
                target = dataset[find_idx[0]:find_idx[-1] + 1, :]

                if find_idx[0] == 0:
                    train = dataset[find_idx[-1] + 1:, :]
                elif find_idx[0] != 0 and find_idx[-1] + 1 != drow:
                    temp1 = dataset[:find_idx[0], :]
                    temp2 = dataset[find_idx[-1] + 1:, :]
                    train = np.vstack([temp1, temp2])
                elif find_idx[-1] + 1 == drow:
                    train = dataset[:find_idx[-1] + 1, :]

                dataset_list.append(target)
                train_list.append(train)

            targetp = self.plabel[find_idx[0]:find_idx[-1] + 1]
            targetc = self.tlabel[find_idx[0]:find_idx[-1] + 1]

            if find_idx[0] == 0:
                trainp = self.plabel[find_idx[-1] + 1:]
                trainc = self.tlabel[find_idx[-1] + 1:]
            elif find_idx[0] != 0 and find_idx[-1] + 1 != drow:
                temp1 = self.plabel[:find_idx[0]]
                temp2 = self.plabel[find_idx[-1] + 1:]
                trainp = np.concatenate([temp1, temp2])

                temp1 = self.tlabel[:find_idx[0]]
                temp2 = self.tlabel[find_idx[-1] + 1:]
                trainc = np.concatenate([temp1, temp2])
            elif find_idx[-1] + 1 == drow:
                trainp = self.plabel[:find_idx[-1] + 1]
                trainc = self.tlabel[:find_idx[-1] + 1]

            for i, dataset in enumerate(dataset_list):
                test_dict[f"data_{i}"] = dataset
            test_dict["people"] = targetp
            test_dict["tag"] = targetc

            for i, dataset in enumerate(train_list):
                train_dict[f"data_{i}"] = dataset
            train_dict["people"] = trainp
            train_dict["tag"] = trainc

            total_dataset["train"].append(train_dict)
            total_dataset["test"].append(test_dict)

        return total_dataset


# LeaveOne sampling Class
class LeaveOneDP(BaseDivideProcess):
    """
        LeaveOne sampling
    """
    def __init__(self, mode, model_name, dataset, rsub):
        super().__init__(mode, model_name, dataset)
        print(f"{datetime.datetime.now()} :: Divide Process : {self.mode}")

    def sampling(self):
        total_dataset = dict()
        total_dataset["train"] = list()
        total_dataset["test"] = list()

        for peo_target in range(self.nb_people):
            train_dict = dict()
            test_dict = dict()

            dataset_list = list()
            train_list = list()

            find_idx = []
            count_idx = 0
            drow, _ = self.data1.shape

            for idx in range(drow):
                if self.plabel[idx] == peo_target:
                    find_idx.append(idx)
                    count_idx += 1

            for dataset in [self.data1, self.data2, self.data3]:
                target = dataset[find_idx[0]:find_idx[-1] + 1, :]

                if find_idx[0] == 0:
                    train = dataset[find_idx[-1] + 1:, :]
                elif find_idx[0] != 0 and find_idx[-1] + 1 != drow:
                    temp1 = dataset[:find_idx[0], :]
                    temp2 = dataset[find_idx[-1] + 1:, :]
                    train = np.vstack([temp1, temp2])
                elif find_idx[-1] + 1 == drow:
                    train = dataset[:find_idx[-1] + 1, :]

                dataset_list.append(target)
                train_list.append(train)

            targetp = self.plabel[find_idx[0]:find_idx[-1] + 1]
            targetc = self.tlabel[find_idx[0]:find_idx[-1] + 1]

            if find_idx[0] == 0:
                trainp = self.plabel[find_idx[-1] + 1:]
                trainc = self.tlabel[find_idx[-1] + 1:]
            elif find_idx[0] != 0 and find_idx[-1] + 1 != drow:
                temp1 = self.plabel[:find_idx[0]]
                temp2 = self.plabel[find_idx[-1] + 1:]
                trainp = np.concatenate([temp1, temp2])

                temp1 = self.tlabel[:find_idx[0]]
                temp2 = self.tlabel[find_idx[-1] + 1:]
                trainc = np.concatenate([temp1, temp2])
            elif find_idx[-1] + 1 == drow:
                trainp = self.plabel[:find_idx[-1] + 1]
                trainc = self.tlabel[:find_idx[-1] + 1]

            target_indexes, _ = dataset_list[0].shape
            train_indexes, _ = train_list[0].shape
            random_list1 = sample(range(target_indexes), target_indexes)
            random_list2 = sample(range(train_indexes), train_indexes)

            for i, dataset in enumerate(dataset_list):
                test_dict[f"data_{i}"] = dataset[random_list1]
            test_dict["people"] = targetp[random_list1]
            test_dict["tag"] = targetc[random_list1]

            for i, dataset in enumerate(train_list):
                train_dict[f"data_{i}"] = dataset[random_list2]
            train_dict["people"] = trainp[random_list2]
            train_dict["tag"] = trainc[random_list2]

            total_dataset["train"].append(train_dict)
            total_dataset["test"].append(test_dict)

        return total_dataset


# LeaveOne sampling Class
class LeaveOneDP_select(BaseDivideProcess):
    """
        LeaveOne sampling
    """
    def __init__(self, mode, model_name, dataset, rsub):
        super().__init__(mode, model_name, dataset)
        print(f"{datetime.datetime.now()} :: Divide Process : {self.mode}")

    def sampling(self):
        total_dataset = dict()
        total_dataset["train"] = list()
        total_dataset["test"] = list()
        seed_num = 0

        for repeat in range(20):
            train_dict = dict()
            test_dict = dict()
            class_collect = dict()

            for target_class in range(1, self.nb_class+1):

                # per label collect
                data1 = self.dataset[0][target_class == self.dataset[0][:, -1]]
                data2 = self.dataset[1][target_class == self.dataset[0][:, -1]]
                data3 = self.dataset[2][target_class == self.dataset[0][:, -1]]

                per_people = list()
                for peo_target in range(self.nb_people+1):

                    find_idx = []
                    count_idx = 0
                    drow, _ = data1.shape

                    for idx in range(drow):
                        if data1[idx, -2] == peo_target:
                            find_idx.append(idx)
                            count_idx += 1

                    if len(find_idx) == 0:
                        continue

                    dataset_list = list()
                    for dataset in [data1, data2, data3]:
                        target = dataset[find_idx[0]:find_idx[-1] + 1, :]
                        dataset_list.append(target)

                    per_people.append(dataset_list)

                class_collect[target_class] = per_people

            test_list = list()
            train_list = list()
            for key, datalist in class_collect.items():
                class_len = len(datalist)
                seed(seed_num)
                seed_num += 1
                ridx = sample(range(class_len), class_len)
                temp_test = datalist.pop(ridx[0])
                temp_train = datalist

                test_list.append(temp_test)
                train_list.extend(temp_train)

            for sens in range(3):
                for i, data in enumerate(test_list):
                    if i == 0:
                        test_dict[f"data_{sens}"] = data[sens][:, :-2]
                        if sens == 0:
                            test_dict["people"] = data[sens][:, -2]
                            test_dict["tag"] = data[sens][:, -1]
                    else:
                        test_dict[f"data_{sens}"] = np.vstack([test_dict[f"data_{sens}"], data[sens][:, :-2]])
                        if sens == 0:
                            test_dict["people"] = np.concatenate([test_dict["people"], data[sens][:, -2]])
                            test_dict["tag"] = np.concatenate([test_dict["tag"], data[sens][:, -1]])

                for i, data in enumerate(train_list):
                    if i == 0:
                        train_dict[f"data_{sens}"] = data[sens][:, :-2]
                        if sens == 0:
                            train_dict["people"] = data[sens][:, -2]
                            train_dict["tag"] = data[sens][:, -1]
                    else:
                        train_dict[f"data_{sens}"] = np.vstack([train_dict[f"data_{sens}"], data[sens][:, :-2]])
                        if sens == 0:
                            train_dict["people"] = np.concatenate([train_dict["people"], data[sens][:, -2]])
                            train_dict["tag"] = np.concatenate([train_dict["tag"], data[sens][:, -1]])

            total_dataset["train"].append(train_dict)
            total_dataset["test"].append(test_dict)

        return total_dataset


# LeaveOne sampling Class
class mdpiDP(BaseDivideProcess):
    """
        mdpi sampling
    """
    def __init__(self, mode, model_name, dataset, rsub):
        super().__init__(mode, model_name, dataset)
        print(f"{datetime.datetime.now()} :: Divide Process : {self.mode}")

    def sampling(self, s1=3, s2=50):
        total_dataset = dict()
        total_dataset["train"] = list()
        total_dataset["test"] = list()

        for repeat in range(20):
            seed(repeat)
            drow, _ = self.data1.shape

            train_dict = dict()
            test_dict = dict()

            for people_target in range(self.nb_people):
                find_idx = []
                for idx in range(drow):
                    if self.plabel[idx] == people_target:
                        find_idx.append(idx)

                dataset_list = list()
                for dataset in [self.data1, self.data2, self.data3]:
                    target = dataset[find_idx[0]:find_idx[-1] + 1, :]
                    dataset_list.append(target)

                targetp = self.plabel[find_idx[0]:find_idx[-1] + 1]
                targetc = self.tlabel[find_idx[0]:find_idx[-1] + 1]

                for class_target in range(self.nb_class):
                    find_idx = []
                    count_idx = 0
                    for idx in range(dataset_list[0].shape[0]):
                        if targetc[idx] == class_target + 1:
                            find_idx.append(idx)
                            count_idx += 1

                    class_list = list()
                    try:
                        for dataset in dataset_list:
                            target = dataset[find_idx[0]:find_idx[-1] + 1, :]
                            class_list.append(target)

                        sec_targetp = targetp[find_idx[0]:find_idx[-1] + 1]
                        sec_targetc = targetc[find_idx[0]:find_idx[-1] + 1]
                    except:
                        class_list = list()
                        continue

                    random_list = sample(range(count_idx), count_idx)

                    for i, target in enumerate(class_list):
                        class_list[i] = target[random_list]
                    sec_targetp = sec_targetp[random_list]
                    sec_targetc = sec_targetc[random_list]

                    if s2 != -1:
                        if people_target == 0:
                            for i, dataset in enumerate(class_list):
                                train_dict[f"data_{i}"] = dataset[:s1, :]
                                test_dict[f"data_{i}"] = dataset[s1:s2, :]
                            train_dict["people"] = sec_targetp[:s1]
                            train_dict["tag"] = sec_targetc[:s1]
                            test_dict["people"] = sec_targetp[s1:s2]
                            test_dict["tag"] = sec_targetc[s1:s2]
                        else:
                            for i, dataset in enumerate(class_list):
                                train_dict[f"data_{i}"] = np.vstack([train_dict[f"data_{i}"], dataset[:s1, :]])
                                test_dict[f"data_{i}"] = np.vstack([test_dict[f"data_{i}"], dataset[s1:s2, :]])
                            train_dict["people"] = np.concatenate([train_dict["people"], sec_targetp[:s1]])
                            train_dict["tag"] = np.concatenate([train_dict["tag"], sec_targetc[:s1]])
                            test_dict["people"] = np.concatenate([test_dict["people"], sec_targetp[s1:s2]])
                            test_dict["tag"] = np.concatenate([test_dict["tag"], sec_targetc[s1:s2]])
                    else:
                        if people_target == 0:
                            for i, dataset in enumerate(class_list):
                                train_dict[f"data_{i}"] = dataset[:s1, :]
                                test_dict[f"data_{i}"] = dataset[s1:, :]
                            train_dict["people"] = sec_targetp[:s1]
                            train_dict["tag"] = sec_targetc[:s1]
                            test_dict["people"] = sec_targetp[s1:]
                            test_dict["tag"] = sec_targetc[s1:]
                        else:
                            for i, dataset in enumerate(class_list):
                                train_dict[f"data_{i}"] = np.vstack([train_dict[f"data_{i}"], dataset[:s1, :]])
                                test_dict[f"data_{i}"] = np.vstack([test_dict[f"data_{i}"], dataset[s1:, :]])
                            train_dict["people"] = np.concatenate([train_dict["people"], sec_targetp[:s1]])
                            train_dict["tag"] = np.concatenate([train_dict["tag"], sec_targetc[:s1]])
                            test_dict["people"] = np.concatenate([test_dict["people"], sec_targetp[s1:]])
                            test_dict["tag"] = np.concatenate([test_dict["tag"], sec_targetc[s1:]])

            other_samples, _ = train_dict["data_0"].shape
            random_list = sample(range(other_samples), other_samples)
            train_dict["people"] = train_dict["people"][random_list]
            train_dict["tag"] = train_dict["tag"][random_list]
            for i in range(3):
                train_dict[f"data_{i}"] = train_dict[f"data_{i}"][random_list]

            other_samples, _ = test_dict["data_0"].shape
            random_list = sample(range(other_samples), other_samples)
            test_dict["people"] = test_dict["people"][random_list]
            test_dict["tag"] = test_dict["tag"][random_list]
            for i in range(3):
                test_dict[f"data_{i}"] = test_dict[f"data_{i}"][random_list]

            total_dataset["train"].append(train_dict)
            total_dataset["test"].append(test_dict)

        return total_dataset


# LeaveOne sampling Class
class mdpi_dhalfDP(BaseDivideProcess):
    """
        mdpi sampling
    """
    def __init__(self, mode, model_name, dataset, rsub):
        super().__init__(mode, model_name, dataset)
        print(f"{datetime.datetime.now()} :: Divide Process : {self.mode}")


    def sampling(self):

        total_dataset = dict()
        total_dataset["train"] = list()
        total_dataset["test"] = list()

        for repeat in range(20):
            seed(repeat)
            drow, _ = self.data1.shape

            train_dict = dict()
            test_dict = dict()

            rindx_list = sample(range(drow), drow)

            dataset_list = list()
            for dataset in [self.data1, self.data2, self.data3]:
                randomized = dataset[rindx_list]
                dataset_list.append(randomized)

            targetc = self.tlabel[rindx_list]
            targetp = self.plabel[rindx_list]

            half_idx = int(drow / 2)

            # get decimal
            result = 0
            previous = 0
            n = 10
            while result == 0:
                output = round(half_idx // n)
                if output == 0:
                    n = n / 10
                    result = previous * n
                else:
                    previous = output
                    n = n * 10

            drop_idx = int(result)

            # drop_idx = 10**(len(half_idx) - 1)

            for i, dataset in enumerate(dataset_list):
                train_dict[f"data_{i}"] = dataset[:drop_idx, :]
                test_dict[f"data_{i}"] = dataset[drop_idx:2*drop_idx, :]

            train_dict["people"] = targetp[:drop_idx]
            train_dict["tag"] = targetc[:drop_idx]
            test_dict["people"] = targetp[drop_idx:2*drop_idx]
            test_dict["tag"] = targetc[drop_idx:2*drop_idx]

            total_dataset["train"].append(train_dict)
            total_dataset["test"].append(test_dict)

        return total_dataset


# LeaveOne sampling Class
class mdpi_halfDP(BaseDivideProcess):
    """
        mdpi sampling
    """
    def __init__(self, mode, model_name, dataset, rsub):
        super().__init__(mode, model_name, dataset)
        print(f"{datetime.datetime.now()} :: Divide Process : {self.mode}")

    def sampling(self):

        total_dataset = dict()
        total_dataset["train"] = list()
        total_dataset["test"] = list()

        for repeat in range(20):
            seed(repeat)
            drow, _ = self.data1.shape

            train_dict = dict()
            test_dict = dict()

            rindx_list = sample(range(drow), drow)

            dataset_list = list()
            for dataset in [self.data1, self.data2, self.data3]:
                randomized = dataset[rindx_list]
                dataset_list.append(randomized)

            targetc = self.tlabel[rindx_list]
            targetp = self.plabel[rindx_list]

            half_idx = int(drow/2)

            for i, dataset in enumerate(dataset_list):
                train_dict[f"data_{i}"] = dataset[:half_idx, :]
                test_dict[f"data_{i}"] = dataset[half_idx:, :]

            train_dict["people"] = targetp[:half_idx]
            train_dict["tag"] = targetc[:half_idx]
            test_dict["people"] = targetp[half_idx:]
            test_dict["tag"] = targetc[half_idx:]

            total_dataset["train"].append(train_dict)
            total_dataset["test"].append(test_dict)

        return total_dataset


# LeaveOne sampling Class
class mdpi_MCCVDP(BaseDivideProcess):
    """
        mdpi sampling
    """
    def __init__(self, mode, model_name, dataset, rsub):
        super().__init__(mode, model_name, dataset)
        print(f"{datetime.datetime.now()} :: Divide Process : {self.mode}")

    def sampling(self):

        total_dataset = dict()
        total_dataset["train"] = list()
        total_dataset["test"] = list()

        for repeat in range(20):
            seed(repeat)
            drow, _ = self.data1.shape

            train_dict = dict()
            test_dict = dict()

            rindx_list = sample(range(drow), drow)

            dataset_list = list()
            for dataset in [self.data1, self.data2, self.data3]:
                randomized = dataset[rindx_list]
                dataset_list.append(randomized)

            targetc = self.tlabel[rindx_list]
            targetp = self.plabel[rindx_list]

            mcv_rate = int(drow * 0.7)

            for i, dataset in enumerate(dataset_list):
                train_dict[f"data_{i}"] = dataset[:mcv_rate, :]
                test_dict[f"data_{i}"] = dataset[mcv_rate:, :]

            train_dict["people"] = targetp[:mcv_rate]
            train_dict["tag"] = targetc[:mcv_rate]
            test_dict["people"] = targetp[mcv_rate:]
            test_dict["tag"] = targetc[mcv_rate:]

            total_dataset["train"].append(train_dict)
            total_dataset["test"].append(test_dict)

        return total_dataset


# 7 - Cross Validation sampling Class
class seven_CVDP(BaseDivideProcess):
    """
        mdpi sampling
    """
    def __init__(self, mode, model_name, dataset, rsub):
        super().__init__(mode, model_name, dataset)
        print(f"{datetime.datetime.now()} :: Divide Process : {self.mode}")

    def sampling(self):

        total_dataset = dict()
        total_dataset["train"] = list()
        total_dataset["test"] = list()

        for repeat in range(5):
            seed(repeat)
            drow, _ = self.data1.shape
            rindx_list = sample(range(drow), drow)

            dataset_list = list()
            for dataset in [self.data1, self.data2, self.data3]:
                randomized = dataset[rindx_list]
                dataset_list.append(randomized)

            targetc = self.tlabel[rindx_list]
            targetp = self.plabel[rindx_list]

            cv_rate = int(drow / 7)
            for cvi in range(7):
                train_dict = dict()
                test_dict = dict()

                for i, dataset in enumerate(dataset_list):
                    test_dict[f"data_{i}"] = dataset[cv_rate*cvi: cv_rate*(cvi+1), :]
                test_dict["people"] = targetp[cv_rate*cvi: cv_rate*(cvi+1)]
                test_dict["tag"] = targetc[cv_rate*cvi: cv_rate*(cvi+1)]

                indexing = np.arange(cv_rate*cvi, cv_rate*(cvi+1))
                for i, dataset in enumerate(dataset_list):
                    train_dict[f"data_{i}"] = np.array([element for idx, element in enumerate(dataset) if idx not in indexing])

                train_dict["people"] = np.array([element for idx, element in enumerate(targetp) if idx not in indexing])
                train_dict["tag"] = np.array([element for idx, element in enumerate(targetc) if idx not in indexing])

                # if cvi == 0:
                #     for i, dataset in enumerate(dataset_list):
                #         test_dict[f"data_{i}"] = dataset[cv_rate:, :]
                #     test_dict["people"] = targetp[cv_rate:]
                #     test_dict["tag"] = targetc[cv_rate:]
                # elif cvi == 6:
                #     for i, dataset in enumerate(dataset_list):
                #         test_dict[f"data_{i}"] = dataset[:cv_rate*cvi, :]
                #     test_dict["people"] = targetp[:cv_rate*cvi]
                #     test_dict["tag"] = targetc[:cv_rate*cvi]
                # else:
                #     for i, dataset in enumerate(dataset_list):
                #         temp1 = dataset[:cv_rate*cvi, :]
                #         temp2 = dataset[cv_rate*(cvi+1):, :]
                #         test_dict[f"data_{i}"] = np.vstack([temp1, temp2])
                #     test_dict["people"] = np.vstack([targetp[:cv_rate*cvi], targetp[cv_rate*(cvi+1):]])
                #     test_dict["tag"] = np.vstack([targetc[:cv_rate*cvi], targetc[cv_rate*(cvi+1):]])

                total_dataset["train"].append(train_dict)
                total_dataset["test"].append(test_dict)

        return total_dataset


# Selected Cross Validation sampling Class
class select_CVDP(BaseDivideProcess):
    """
        mdpi sampling
    """
    def __init__(self, mode, model_name, dataset, rsub):
        super().__init__(mode, model_name, dataset)
        print(f"{datetime.datetime.now()} :: Divide Process : {self.mode}")

    def sampling(self):

        total_dataset = dict()
        total_dataset["train"] = list()
        total_dataset["test"] = list()

        for repeat in range(10):
            seed(repeat)
            drow, _ = self.data1.shape

            train_dict = dict()
            test_dict = dict()

            rindx_list = sample(range(drow), drow)

            dataset_list = list()
            for dataset in [self.data1, self.data2, self.data3]:
                randomized = dataset[rindx_list]
                dataset_list.append(randomized)

            targetc = self.tlabel[rindx_list]
            targetp = self.plabel[rindx_list]

            cv_rate = int(drow / 7)
            for cvi in range(7):
                for i, dataset in enumerate(dataset_list):
                    train_dict[f"data_{i}"] = dataset[cv_rate*cvi: cv_rate*cvi+1, :]
                train_dict["people"] = targetp[:cv_rate]
                train_dict["tag"] = targetc[:cv_rate]

                if cvi == 0:
                    for i, dataset in enumerate(dataset_list):
                        test_dict[f"data_{i}"] = dataset[cv_rate:, :]
                    test_dict["people"] = targetp[cv_rate:]
                    test_dict["tag"] = targetc[cv_rate:]
                elif cvi == 6:
                    for i, dataset in enumerate(dataset_list):
                        test_dict[f"data_{i}"] = dataset[:cv_rate*cvi, :]
                    test_dict["people"] = targetp[:cv_rate*cvi]
                    test_dict["tag"] = targetc[:cv_rate*cvi]
                else:
                    for i, dataset in enumerate(dataset_list):
                        temp1 = dataset[:cv_rate*cvi, :]
                        temp2 = dataset[cv_rate*cvi+1:, :]
                        test_dict[f"data_{i}"] = np.vstack([temp1, temp2])
                    test_dict["people"] = np.vstack([targetp[:cv_rate*cvi], targetp[cv_rate*cvi+1]])
                    test_dict["tag"] = np.vstack([targetc[:cv_rate*cvi], targetc[cv_rate*cvi+1]])

                total_dataset["train"].append(train_dict)
                total_dataset["test"].append(test_dict)

        return total_dataset


# Base Divide Process Class
class BaseVectorDivideProcess:
    def __init__(self, mode, model_name, dataset):
        assert len(dataset) == 3, "dataset must be 3 arguments"
        pressure, accl, accr, gyrl, gyrr, info = dataset

        # [data1, data2, data3] = pp.sort_by_people(dataset)

        # sampling func name
        self.mode = mode
        # used model name
        self.model_name = model_name

        self.dataset = dataset
        self.real_plabel = info[:, 0]
        self.plabel = info[:, 1]
        self.tlabel = info[:, 2]

        # dataset index
        self.pressure = pressure
        self.acc = [accl, accr]
        self.gyro = [gyrl, gyrr]

        self.nb_class = int(max(self.tlabel))
        self.nb_people = int(max(self.plabel)) + 1

    def sampling(self):
        pass

    def convert(self, data, mt, comb):
        # need to update
        drow, dcol = data.shape
        input_shape = (int(mt * comb), int((dcol) / (mt * comb)))
        if self.model_name in method_info['4columns']:
            converted = data.reshape(-1, input_shape[0], input_shape[1], 1)
        elif self.model_name == "pVGG":
            data = data.reshape(-1, input_shape[0], input_shape[1])
            converted = np.zeros((data.shape[0], data.shape[1], data.shape[2], 3))
            for idx in range(3):
                converted[:, :, :, idx] = data
        elif self.model_name in method_info['3columns']:
            converted = data.reshape(-1, input_shape[0], input_shape[1])
        elif self.model_name in method_info['2columns']:
            converted = data
        elif self.model_name in method_info['specific']:
            converted = data
        elif self.model_name in method_info['vector']:
            converted = data
        return converted


class method_as_vector(BaseDivideProcess):
    """
            convert vector method
    """

    def __init__(self, mode, model_name, dataset, rsub):
        super().__init__(mode, model_name, dataset)
        print(f"{datetime.datetime.now()} :: Divide Process : {self.mode}")

    def sampling(self):

        total_dataset = dict()
        total_dataset["train"] = list()
        total_dataset["test"] = list()

        return total_dataset