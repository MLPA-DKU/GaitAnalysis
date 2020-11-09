import lightgbm as lgb
import numpy as np


def lgbm_construct(param, dataset, label):
    nb_class = label[0]
    nb_people = label[1]

    tartr = dataset[0]
    tarte = dataset[1]

    tr_data = [tartr["data_0"], tartr["data_1"], tartr["data_2"]]
    te_data = [tarte["data_0"], tarte["data_1"], tarte["data_2"]]
    if param.datatype == "type":
        if param.lable == "people":
            tr_label = tartr[param.lable]
            te_label = tarte[param.lable]
        elif param.lable == "tag":
            tr_label = tartr[param.lable]
            te_label = tarte[param.lable]
        # nb_class = label_info[0]
    elif param.datatype == "disease":
        tr_label = tartr[param.lable]
        te_label = tarte[param.lable]
        # nb_class = label_info[0]

    modal_list = list()
    for modality in [0, 1, 2]:
        print(f"modality : {modality} experiment")
        train = tr_data[modality]
        test = te_data[modality]

        train_data = lgb.Dataset(train, label=tr_label)
        test_data = lgb.Dataset(test, label=te_label)

        # Hyper Parameter for Classification
        lgb_param = {'learning_rate': 0.01, 'objective': 'multiclass',
                     'metric': ['multi_logloss'], 'max_depth': -1, 'num_class': nb_class,
                     'boosting_type': 'gbdt', 'num_leaves': 512, 'verbose': -1}

        num_round = 2000

        # SINGULAR : bst = lgb.train(lgb_param, train_data, num_round, valid_sets=[test_data])
        # CROSS VALIDATION : bst = lgb.cv(lgb_param, train_data, num_round, nfold=5)
        bst = lgb.train(lgb_param, train_data, num_round, valid_sets=[test_data], early_stopping_rounds=50)
        ypred = bst.predict(test)

        pred_match = np.argmax(ypred, axis=1)
        # precision_score = np.mean(pred_match, te_label)

        nb_sample, _ = test.shape

        counting = 0
        for sample in range(nb_sample):
            if pred_match[sample] == int(te_label[sample]):
                counting += 1

        accuracy = counting / nb_sample
        modal_list.append(accuracy)
        print(f"modal {modality} test accuracy: {accuracy}")

    return modal_list
