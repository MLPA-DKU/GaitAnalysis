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


def get_model(input_shape, comb_degree):
    model = Sequential()
    hidden_cells = 64 * comb_degree
    model.add(LSTM(hidden_cells, input_shape=input_shape, return_sequences=True, recurrent_dropout=0.2))
    model.add(LSTM(hidden_cells))    #A second layer of LSTM
    model.add(Flatten())
    return model


# multi modal DCNN
def lstm_network(shape_list, nb_class, comb_degree):

    input1 = get_model((shape_list[0]),comb_degree)
    input2 = get_model((shape_list[1]),comb_degree)
    input3 = get_model((shape_list[2]),comb_degree)

    merged_layer = concatenate([input1.output, input2.output, input3.output])
    dense_layer1 = Dense(256, activation="relu")(merged_layer)
    dropout_layer = Dropout(0.7)(dense_layer1)
    dense_layer2 = Dense(nb_class, activation="softmax")(dropout_layer)

    model = Model(inputs=[input1.input, input2.input, input3.input], outputs=dense_layer2)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])

    return model
