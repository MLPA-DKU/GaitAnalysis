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


def get_model(input_shape):
    hidden_cells = 32

    input_model = Input(shape=input_shape)

    # fit = keras.layers.Conv1D(filters=hidden_cells * 2, kernel_size=1, strides=1, padding='valid')(input_model)
    # fit_batchnorm = keras.layers.BatchNormalization()(fit)

    def bidirectional_block(input_layer, hidden_cells):
        forward_layer = LSTM(hidden_cells, activation='relu', return_sequences=True)
        backward_layer = LSTM(hidden_cells, activation='relu', return_sequences=True,
                              go_backwards=True)
        bidirectional_layer = Bidirectional(forward_layer, backward_layer=backward_layer)(input_layer)
        return bidirectional_layer

    # modify = bidirectional_block(input_model, hidden_cells)
    # hidden_cells = int(hidden_cells/2)
    # skip = keras.layers.Add()([fit_batchnorm, modify])
    #
    # fit = keras.layers.Conv1D(filters=hidden_cells * 2, kernel_size=1, strides=1, padding='valid')(skip)
    # fit_batchnorm = keras.layers.BatchNormalization()(fit)
    #
    # modify2 = bidirectional_block(skip, hidden_cells)
    # skip2 = keras.layers.Add()([fit_batchnorm, modify2])

    modify = bidirectional_block(input_model, hidden_cells)

    flatten = Flatten()(modify)
    model = Model(inputs=input_model, outputs=flatten)
    return model


# multi modal DCNN
def bilstm_network(shape_list, nb_class, comb_degree):

    input1 = get_model((shape_list[0]))
    input2 = get_model((shape_list[1]))
    input3 = get_model((shape_list[2]))

    merged_layer = concatenate([input1.output, input2.output, input3.output])
    dense_layer1 = Dense(3000, activation="relu")(merged_layer)
    dropout_layer1 = Dropout(0.25)(dense_layer1)
    dense_layer2 = Dense(256, activation="relu")(dropout_layer1)
    dropout_layer2 = Dropout(0.25)(dense_layer2)
    dense_layer3 = Dense(nb_class, activation="softmax")(dropout_layer2)

    model = Model(inputs=[input1.input, input2.input, input3.input], outputs=dense_layer3)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])

    return model
