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


def lstm_block(comb_degree, input_layer):

    # input_layer = keras.layers.Input(input_shape)
    hidden_cells = 64 * comb_degree
    lstm1 = keras.layers.LSTM(hidden_cells, return_sequences=True, recurrent_dropout=0.2)(input_layer)
    lstm2 = keras.layers.LSTM(hidden_cells, return_sequences=True, recurrent_dropout=0.2)(lstm1)
    flatten = keras.layers.Flatten()(lstm2)

    # model = keras.models.Model(inputs=input_layer, output=flatten)
    # model = Sequential()
    # model.add(LSTM(hidden_cells, input_shape=input_shape, return_sequences=True, recurrent_dropout=0.2))
    # model.add(LSTM(hidden_cells))    #A second layer of LSTM
    # model.add(Flatten())
    return flatten


def cnn_block(comb_degree, input_layer):
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
    input1_flatten = keras.layers.Flatten()(input1_batchnorm3)

    dense_layer1 = keras.layers.Dense(units=256, activation='relu')(input1_flatten)
    dense_layer2 = keras.layers.Dense(units=64, activation='relu')(dense_layer1)

    # model = keras.models.Model(input=input_layer, output=dense_layer)

    return dense_layer2


def cnn_lstm_network(shape_list, nb_class, comb_degree):

    input1 = keras.layers.Input(shape=shape_list[0])
    input2 = keras.layers.Input(shape=shape_list[1])
    input3 = keras.layers.Input(shape=shape_list[2])

    input1_lstm = lstm_block(comb_degree, input1)
    input2_lstm = lstm_block(comb_degree, input2)
    input3_lstm = lstm_block(comb_degree, input3)

    input1_cnn = cnn_block(comb_degree, input1)
    input2_cnn = cnn_block(comb_degree, input2)
    input3_cnn = cnn_block(comb_degree, input3)

    input1_merged = keras.layers.concatenate([input1_lstm, input1_cnn])
    input2_merged = keras.layers.concatenate([input2_lstm, input2_cnn])
    input3_merged = keras.layers.concatenate([input3_lstm, input3_cnn])

    input1_dense = keras.layers.Dense(256)(input1_merged)
    input2_dense = keras.layers.Dense(256)(input2_merged)
    input3_dense = keras.layers.Dense(256)(input3_merged)

    merged_layer = keras.layers.concatenate([input1_dense, input2_dense, input3_dense])
    merged_dense1 = keras.layers.Dense(units=256, activation='relu')(merged_layer)
    merged_batchnorm1 = keras.layers.BatchNormalization()(merged_dense1)
    merged_dense2 = keras.layers.Dense(units=nb_class, activation='softmax')(merged_batchnorm1)

    model = keras.models.Model([input1, input2, input3], merged_dense2)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])

    return model
