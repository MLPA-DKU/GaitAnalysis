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
        tr_label = tartr["tag"] - 1
        te_label = tarte["tag"] - 1
    elif param.datatype == "disease":
        tr_label = tartr["tag"]
        te_label = tarte["tag"]

    modal_list = list()
    for modality in range(3):
        train = tr_data[modality]
        test = te_data[modality]

        train_data = lgb.Dataset(train, label=tr_label)
        test_data = lgb.Dataset(test, label=te_label)

        # Hyper Parameter for Classification
        lgb_param = {'learning_rate': 0.01, 'objective': 'multiclass',
                     'metric': ['multi_logloss'], 'max_depth': 10, 'num_class': nb_class,
                     'boosting_type': 'gbdt', 'num_leaves': 1024, 'verbose': -1}

        num_round = 500

        # SINGULAR : bst = lgb.train(lgb_param, train_data, num_round, valid_sets=[test_data])
        # CROSS VALIDATION : bst = lgb.cv(lgb_param, train_data, num_round, nfold=5)
        bst = lgb.train(lgb_param, train_data, num_round, valid_sets=[test_data])
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
        print(f"modal{modality}: {accuracy}")

    return modal_list
