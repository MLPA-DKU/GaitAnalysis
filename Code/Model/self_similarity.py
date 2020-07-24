import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam

from tensorflow.python.framework import ops

def modified_cnn_block(comb_degree, input_layer):
    nb_filter = 32
    nb_strides = 1

    if int(comb_degree) == 3:
        nb_filter = nb_filter * 2
        nb_strides = 2
    elif int(comb_degree) == 4:
        nb_filter = nb_filter * 2
        nb_strides = 2
    elif int(comb_degree) == 5:
        nb_filter = nb_filter * 2
        nb_strides = 2

    input_layer = input_layer
    # input1 = keras.layers.Input(shape=shape_list[0])
    input1_cnn1 = keras.layers.Conv1D(filters=nb_filter, kernel_size=20,
                                      strides=nb_strides, activation='relu')(input_layer)
    input1_batchnorm1 = keras.layers.BatchNormalization()(input1_cnn1)

    input1_cnn2 = keras.layers.Conv1D(filters=nb_filter * 2, kernel_size=20,
                                      strides=nb_strides, activation='relu')(input1_batchnorm1)
    input1_batchnorm2 = keras.layers.BatchNormalization()(input1_cnn2)

    input1_cnn3 = keras.layers.Conv1D(filters=nb_filter * 4, kernel_size=20,
                                      strides=nb_strides, activation='relu')(input1_batchnorm2)
    input1_batchnorm3 = keras.layers.BatchNormalization()(input1_cnn3)
    # input1_flatten = keras.layers.Flatten()(input1_batchnorm3)

    # dense_layer1 = keras.layers.Dense(units=256, activation='relu')(input1_flatten)
    # dense_layer = keras.layers.Dense(units=256, activation='relu')(input1_flatten)
    # dense_layer = keras.layers.Dense(units=64, activation='relu')(dense_layer)
    permute_layer = keras.layers.Permute((2, 1))(input1_batchnorm3)
    similarity_layer = keras.layers.Dot(axes=(1, 2))([input1_batchnorm3, permute_layer])

    similarity_layer = keras.layers.Conv1D(filters=nb_filter * 2, kernel_size=20,
                                           strides=1, activation='relu')(similarity_layer)
    # dense_layer2 = keras.layers.Dense(units=64, activation='relu')(similarity_layer)
    # softmax_layer = keras.activations.softmax(similarity_layer, axis=1)

    # model = keras.models.Model(input=input_layer, output=dense_layer)

    return similarity_layer


def self_similarity_network(shape_list, comb_degree):

    input1 = keras.layers.Input(shape=shape_list[0])
    input2 = keras.layers.Input(shape=shape_list[1])
    input3 = keras.layers.Input(shape=shape_list[2])

    input1_cnn = modified_cnn_block(comb_degree, input1)
    input2_cnn = modified_cnn_block(comb_degree, input2)
    input3_cnn = modified_cnn_block(comb_degree, input3)

    input1_per = keras.layers.Permute((2, 1))(input1_cnn)
    input2_per = keras.layers.Permute((2, 1))(input2_cnn)
    input3_per = keras.layers.Permute((2, 1))(input3_cnn)

    pa = keras.layers.Dot(axes=(1, 2))([input1_cnn, input2_per])
    pg = keras.layers.Dot(axes=(1, 2))([input2_cnn, input3_per])
    ag = keras.layers.Dot(axes=(1, 2))([input3_cnn, input1_per])

    pa = keras.layers.Flatten()(pa)
    pa = keras.layers.Dense(units=256, activation='relu')(pa)
    pa = keras.layers.Dense(units=7, activation='softmax')(pa)

    pg = keras.layers.Flatten()(pg)
    pg = keras.layers.Dense(units=256, activation='relu')(pg)
    pg = keras.layers.Dense(units=7, activation='softmax')(pg)

    ag = keras.layers.Flatten()(ag)
    ag = keras.layers.Dense(units=256, activation='relu')(ag)
    ag = keras.layers.Dense(units=7, activation='softmax')(ag)

    # summation = keras.layers.Add()([pa, pg, ag])
    # summation = keras.layers.Lambda(lambda inputs: inputs[0] / inputs[1])([summation, 3.0])  # ensemble
    # summation = keras.layers.Flatten()(summation)
    # summation = keras.layers.Dense(units=256, activation='relu')(summation)
    # summation = keras.layers.Dense(units=7, activation='softmax')(summation)

    model = keras.models.Model([input1, input2, input3], [pa, pg, ag])
    # model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss=keras.losses.categorical_crossentropy,
    #               metrics=['accuracy'])

    # for layer in model.layers:
    #     print(layer.name)

    # merged_layer = keras.layers.concatenate([input1_cnn, input2_cnn, input3_cnn])
    # merged_dense1 = keras.layers.Dense(units=256, activation='relu')(merged_layer)
    # merged_batchnorm1 = keras.layers.BatchNormalization()(merged_dense1)
    # merged_dense2 = keras.layers.Dense(units=nb_class, activation='softmax')(merged_batchnorm1)
    #
    # model = keras.models.Model([input1, input2, input3], merged_dense2)
    # model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss=keras.losses.categorical_crossentropy,
    #               metrics=['accuracy'])

    return model
