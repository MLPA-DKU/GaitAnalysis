import tensorflow as tf
from keras.backend import set_session
from keras import backend as K
import keras
from Code.Model import basic_10_network as BasicNet
from Code.Model import resnet as ResNet
from Code.Model import vgg_network as VGGNet
from Code.Model import base_model_v2 as dcnn_v2
from Code.Model import base_model as dcnn
from Code.Model import lstm_model as lstm
from Code.Model import cnn_lstm_model as clstm
from Code.Model import bidirectinal_lstm_model as bilstm
from Code.Model import lightGBM_model as lgbm
from Code.Model import cropNet_model as crop
from Code.Model import self_similarity as ss
from Code.Model import divide_vector_model as div_vec
from Code.Model import divide_base_model as div_dcnn

model_info = {
    'dl': ['BasicNet', 'ResNet', 'VGG', 'pVGG', 'base', 'lstm', 'bi-lstm', 'cnn_lstm', 'base_v2', 'div_base'],
    'c_dl': ['similarity', 'lstm_attention'],
    'v_dl': ['div_vec'],
    'ml': ['lgbm'],
    'ensemble': ['ensemble']
}


def model_setting(param, train, test, label_info):
    model = param.model_name
    nb_class = label_info[0]
    nb_people = label_info[1]
    nb_modal = param.nb_modal

    if model == 'BasicNet':
        shape_list = list()
        for i in range(nb_modal):
            _, row, col, _ = train[f"data_{i}"].shape
            shape_list.append((row, col))
        model = BasicNet.basic_10_network(shape_list=shape_list, nb_class=nb_class)

    elif model == 'ResNet':
        shape_list = list()
        for i in range(nb_modal):
            _, row, col, ch = train[f"data_{i}"].shape
            shape_list.append((row, col, ch))
        model = ResNet.resnet_builder(shape_list=shape_list, nb_class=nb_class)

    elif model == 'VGG':
        shape_list = list()
        for i in range(nb_modal):
            _, row, col, ch = train[f"data_{i}"].shape
            shape_list.append((row, col, ch))
        model = VGGNet.trained_vgg_builder(shape_list=shape_list, nb_class=nb_class, trainable=False)

    elif model == 'pVGG':
        shape_list = list()
        for i in range(nb_modal):
            _, row, col, ch = train[f"data_{i}"].shape
            shape_list.append((row, col, ch))
        model = VGGNet.trained_vgg_builder(shape_list=shape_list, nb_class=nb_class, trainable=False)

    elif model == 'base':
        shape_list = list()
        for i in range(nb_modal):
            _, row, col = train[f"data_{i}"].shape
            shape_list.append((row, col))
        model = dcnn.dcnn_network(shape_list=shape_list, nb_class=nb_class, comb_degree=param.nb_combine)

    elif model == 'base_v2':
        shape_list = list()
        for i in range(nb_modal):
            _, row, col = train[f"data_{i}"].shape
            shape_list.append((row, col))
        model = dcnn_v2.dcnn_network(shape_list=shape_list, nb_class=nb_class, comb_degree=param.nb_combine)

    elif model == 'div_base':
        shape_list = list()
        for i in range(nb_modal):
            if nb_modal != 0:
                _, row, col = train[f"data_{i}"][0].shape
                shape_list.append((row, col))
                _, row, col = train[f"data_{i}"][1].shape
                shape_list.append((row, col))
            else:
                _, row, col = train[f"data_{i}"].shape
                shape_list.append((row, col))
        model = div_dcnn.dcnn_network(shape_list=shape_list, nb_class=nb_class, comb_degree=param.nb_combine)

    elif model == 'lstm':
        shape_list = list()
        for i in range(nb_modal):
            _, row, col = train[f"data_{i}"].shape
            shape_list.append((row, col))
        model = lstm.lstm_network(shape_list=shape_list, nb_class=nb_class, comb_degree=param.nb_combine)
    elif model == 'cnn_lstm':
        shape_list = list()
        for i in range(nb_modal):
            _, row, col = train[f"data_{i}"].shape
            shape_list.append((row, col))
        model = clstm.cnn_lstm_network(shape_list=shape_list, nb_class=nb_class, comb_degree=param.nb_combine)
    elif model == 'bi-lstm':
        shape_list = list()
        for i in range(nb_modal):
            _, row, col = train[f"data_{i}"].shape
            shape_list.append((row, col))
        model = bilstm.bilstm_network(shape_list=shape_list, nb_class=nb_class, comb_degree=param.nb_combine)
    elif model == 'lstm_attention':
        NotImplemented
    elif model == 'lgbm':
        model = lgbm.lgbm_construct(param, dataset=[train, test], label=[nb_class, nb_people])
    elif model == 'cropping':
        model = crop.cropping_network(dataset=[train, test], nb_class=nb_class)
    elif model == 'similarity':
        shape_list = list()
        for i in range(nb_modal):
            _, row, col = train[f"data_{i}"].shape
            shape_list.append((row, col))
        model = ss.self_similarity_network_conv2d(shape_list=shape_list, comb_degree=param.nb_combine)
    elif model == 'div_vec':
        model = div_vec.divide_network(shape_list=None, comb_degree=param.nb_combine)

    elif model == 'ensemble':
        shape_list = list()
        for i in range(nb_modal):
            _, row, col = train[f"data_{i}"].shape
            shape_list.append((row, col))
        model1 = dcnn.dcnn_network(shape_list=shape_list, nb_class=nb_class, comb_degree=param.nb_combine)
        model2 = lstm.lstm_network(shape_list=shape_list, nb_class=nb_class, comb_degree=param.nb_combine)
        return model1, model2

    return model
