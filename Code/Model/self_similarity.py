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


def modified_cnn_block(comb_degree, input_layer, sensor_index):
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
    similarity_layer1 = keras.layers.Dot(axes=(1, 2), name=f"similarity_block{sensor_index}")([input1_batchnorm3, permute_layer])
    # (w, h) = tf.shape(similarity_layer1)
    # reshaped = keras.layers.Reshape((128, 128, 1))(similarity_layer1)
    # similarity_layer2 = keras.layers.Conv2D(nb_filter * 2, kernel_size=(1, 1), activation='relu')(reshaped)
    similarity_layer2 = keras.layers.Conv1D(filters=nb_filter * 2, kernel_size=20,
                                           strides=1, activation='relu')(similarity_layer1)
    # dense_layer2 = keras.layers.Dense(units=64, activation='relu')(similarity_layer)
    # softmax_layer = keras.activations.softmax(similarity_layer, axis=1)

    # model = keras.models.Model(input=input_layer, output=dense_layer)

    return similarity_layer2, similarity_layer1


def self_similarity_network(shape_list, comb_degree):

    input1 = keras.layers.Input(shape=shape_list[0])
    input2 = keras.layers.Input(shape=shape_list[1])
    input3 = keras.layers.Input(shape=shape_list[2])

    input1_cnn, sim_block1 = modified_cnn_block(comb_degree, input1, sensor_index=1)
    input2_cnn, sim_block2 = modified_cnn_block(comb_degree, input2, sensor_index=2)
    input3_cnn, sim_block3 = modified_cnn_block(comb_degree, input3, sensor_index=3)

    input1_per = keras.layers.Permute((2, 1))(input1_cnn)
    input2_per = keras.layers.Permute((2, 1))(input2_cnn)
    input3_per = keras.layers.Permute((2, 1))(input3_cnn)

    pa = keras.layers.Dot(axes=(1, 2))([input1_cnn, input2_per])
    pg = keras.layers.Dot(axes=(1, 2))([input2_cnn, input3_per])
    ag = keras.layers.Dot(axes=(1, 2))([input3_cnn, input1_per])

    pa = keras.layers.Flatten()(pa)
    pa = keras.layers.Dense(units=256, activation='relu')(pa)
    pa = keras.layers.Dense(units=7)(pa)
    pa = keras.layers.Softmax()(pa)

    pg = keras.layers.Flatten()(pg)
    pg = keras.layers.Dense(units=256, activation='relu')(pg)
    pg = keras.layers.Dense(units=7)(pg)
    pg = keras.layers.Softmax()(pg)

    ag = keras.layers.Flatten()(ag)
    ag = keras.layers.Dense(units=256, activation='relu')(ag)
    ag = keras.layers.Dense(units=7)(ag)
    ag = keras.layers.Softmax()(ag)


    # summation = keras.layers.Add()([pa, pg, ag])
    # summation = keras.layers.Lambda(lambda inputs: inputs[0] / inputs[1])([summation, 3.0])  # ensemble
    # summation = keras.layers.Flatten()(summation)
    # summation = keras.layers.Dense(units=256, activation='relu')(summation)
    # summation = keras.layers.Dense(units=7, activation='softmax')(summation)

    model = keras.models.Model([input1, input2, input3], [pa, pg, ag, sim_block1, sim_block2, sim_block3])
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


def modified_cnn2d_block(comb_degree, input_layer, sensor_index):
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

    input1_cnn3 = keras.layers.Conv1D(filters=nb_filter * 2, kernel_size=20,
                                      strides=nb_strides, activation='relu')(input1_batchnorm2)
    input1_batchnorm3 = keras.layers.BatchNormalization()(input1_cnn3)
    input1_batchnorm3 = keras.layers.GlobalAveragePooling1D()(input1_batchnorm3)

    input1_batchnorm3 = keras.layers.Flatten()(input1_batchnorm3)
    input1_batchnorm3 = keras.layers.Reshape((1, 64))(input1_batchnorm3)

    permute_layer = keras.layers.Permute((2, 1))(input1_batchnorm3)
    similarity_layer1 = keras.layers.Dot(axes=(1, 2), name=f"similarity_block{sensor_index}")([input1_batchnorm3, permute_layer])

    reshaped = keras.layers.Reshape((64, 64, 1))(similarity_layer1)
    similarity_layer2 = keras.layers.Conv2D(nb_filter * 4, kernel_size=(3, 3), activation='relu')(reshaped)
    similarity_layer2 = keras.layers.BatchNormalization()(similarity_layer2)
    pool1 = keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(similarity_layer2)
    similarity_layer3 = keras.layers.Conv2D(nb_filter * 2, kernel_size=(1, 1), activation='relu')(pool1)
    similarity_layer3 = keras.layers.BatchNormalization()(similarity_layer3)

    # similarity_layer 1 will be visualized as similarity matrix
    return similarity_layer3, similarity_layer1


def self_similarity_network_conv2d(shape_list, comb_degree):

    input1 = keras.layers.Input(shape=shape_list[0])
    input2 = keras.layers.Input(shape=shape_list[1])
    input3 = keras.layers.Input(shape=shape_list[2])

    # (32, 32, 128)
    input1_cnn, sim_block1 = modified_cnn2d_block(comb_degree, input1, sensor_index=1)
    input2_cnn, sim_block2 = modified_cnn2d_block(comb_degree, input2, sensor_index=2)
    input3_cnn, sim_block3 = modified_cnn2d_block(comb_degree, input3, sensor_index=3)

    pa = keras.layers.Flatten()(input1_cnn)
    pa = keras.layers.Dropout(0.3)(pa)
    con_pa = keras.layers.Dense(units=256, activation='relu')(pa)
    pa = keras.layers.Dense(units=7)(con_pa)
    pa = keras.layers.Softmax()(pa)

    pg = keras.layers.Flatten()(input2_cnn)
    pg = keras.layers.Dropout(0.3)(pg)
    con_pg = keras.layers.Dense(units=256, activation='relu')(pg)
    pg = keras.layers.Dense(units=7)(con_pg)
    pg = keras.layers.Softmax()(pg)

    ag = keras.layers.Flatten()(input3_cnn)
    ag = keras.layers.Dropout(0.3)(ag)
    con_ag = keras.layers.Dense(units=256, activation='relu')(ag)
    ag = keras.layers.Dense(units=7)(con_ag)
    ag = keras.layers.Softmax()(ag)

    merged_layer = keras.layers.concatenate([con_pa, con_pg, con_ag])
    merged_layer = keras.layers.Dense(units=7, activation='relu')(merged_layer)

    merged_layer = keras.layers.Dense(units=7, activation='softmax')(merged_layer)

    # summation = keras.layers.Add()([pa, pg, ag])
    # summation = keras.layers.Lambda(lambda inputs: inputs[0] / inputs[1])([summation, 3.0])  # ensemble
    # summation = keras.layers.Flatten()(summation)
    # summation = keras.layers.Dense(units=256, activation='relu')(summation)
    # summation = keras.layers.Dense(units=7, activation='softmax')(summation)

    model = keras.models.Model([input1, input2, input3], [pa, pg, ag, sim_block1, sim_block2, sim_block3, merged_layer])
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