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
import numpy as np

from Code.preprocessing import to_categorical_unit


class binary_blocks:
    def __init__(self):
        self.stack = None
        self.state = False

    def stack_block(self, binary):
        if self.state is False:
            self.stack = binary
            self.state = True
        else:
            self.stack = tf.stack(binary)


class cropnet:
    def __init__(self):
        self.batch_size = 256
        self.row_point = 128
        self.in_col = dict()
        self.channels = 1

        self.input_shape = list()

        self.nb_class = None
        self.batch_dataset = None
        self.generator = None
        self.discriminator = None

    def network_init(self, train, nb_class):
        self.nb_class = nb_class

        for i in range(3):
            data = train[f"data_{i}"]
            row, col = data.shape
            self.in_col[f"{i}"] = col
            self.input_shape.append((self.row_point, col, self.channels))

        batch_per_epoch = row // self.batch_size
        batch = dict()
        for i in range(3):
            batch[f"data_{i}"] = list()
        batch["label"] = list()

        for i in range(3):
            target = train[f"data_{i}"]
            for batch_time in range(batch_per_epoch):
                temp_batch = np.zeros((batch_per_epoch, self.batch_size, self.in_col[f"{i}"]))
                temp_label = np.zeros((batch_per_epoch, self.batch_size, nb_class))

                pre = batch_time * self.batch_size
                aft = (batch_time + 1) * self.batch_size
                for n, batch_item in enumerate(target[pre:aft]):
                    onehot_label = to_categorical_unit(train["tag"][n], nb_class)
                    temp_batch[batch_time, n, :] = batch_item
                    temp_label[batch_time, n, :] = onehot_label

                batch[f"data_{i}"].append(temp_batch)
                batch["label"].append(temp_label)

        self.batch_dataset = batch
        # self.generator = self.build_generator()

    def build_generator(self):
        input1 = Input(self.input_shape[0])
        input2 = Input(self.input_shape[1])
        input3 = Input(self.input_shape[2])
        cnn_block = gen_cnn_block(input1)
        attention_block1 = gen_lstm_block(1, input2)
        attention_block2 = gen_lstm_block(2, input3)

        cnn_block = keras.backend.sum(attention_block1)(cnn_block)
        summation = keras.backend.sum(attention_block2)(cnn_block)
        bc = Dense(units=1, activation='tanh')(summation)

        model = Model(inputs=[input1, input2, input3], output=bc)

        return model

    def train(self, dataset, epochs=20, batch_size=128):
        train_x, train_y, test_x, test_y = dataset

        bb = binary_blocks()

        batch_data1 = np.zeros((batch_size, self.row_point, 16))
        batch_data2 = np.zeros((batch_size, self.row_point, 6))
        batch_data3 = np.zeros((batch_size, self.row_point, 6))
        batch_label = np.zeros((batch_size, 2))


def gen_lstm_block(idx, input_layer):
    hidden_cells = 64
    lstm1 = keras.layers.LSTM(hidden_cells, return_sequences=True, recurrent_dropout=0.2, name=f'{idx}_gen_lstm_layer1')(input_layer)
    lstm2 = keras.layers.LSTM(hidden_cells, return_sequences=True, recurrent_dropout=0.2, name=f'{idx}_gen_lstm_layer2')(lstm1)
    lstm_flatten = Flatten(lstm2, name=f'{idx}_gen_lstm_flatten')
    lstm_dense1 = Dense(units=256, name=f'{idx}_gen_lstm_dense1')(lstm_flatten)
    lstm_dense2 = Dense(units=64, name=f'{idx}_gen_lstm_dense2')(lstm_dense1)
    return lstm_dense2


def gen_cnn_block(input_layer):
    nb_filter = 32
    nb_strides = 1

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
    return dense_layer2


def generative_block(shape_list, train):
    input1 = Input(shape_list[0])
    input2 = Input(shape_list[1])
    input3 = Input(shape_list[2])
    cnn_block = gen_cnn_block(input1)
    attention_block1 = gen_lstm_block(1, input2)
    attention_block2 = gen_lstm_block(2, input3)

    cnn_block = keras.backend.sum(attention_block1)(cnn_block)
    summation = keras.backend.sum(attention_block2)(cnn_block)
    bc = Dense(units=1, activation='tanh')(summation)

    model = Model(inputs=[input1, input2, input3], output=bc)

    return model(train)


def validate_block():
    NotImplemented


def cropping_network(dataset, nb_class):
    bb = binary_blocks()
    framework = cropnet()

    train, test = dataset
    framework.network_init(train, nb_class)
    # train_x = train[:, :-2]
    # train_y = train[:, -1]
    # test_x = test[:, :-2]
    # test_y = test[:, -1]





