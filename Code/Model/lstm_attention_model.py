import keras
from keras.layers import Dense, Lambda, dot, Activation, Concatenate


def custom_lstm_model(param, shape_list):
    NotImplemented


def lstm_attention_model(param, shape_list):
    # Basic 10 Models
    input1 = keras.layers.Input(shape=(shape_list[0][0], shape_list[0][1], 1))
    input2 = keras.layers.Input(shape=(shape_list[1][0], shape_list[1][1], 1))
    input3 = keras.layers.Input(shape=(shape_list[2][0], shape_list[2][1], 1))

    temp_unit = 64
    attention_blcok1 = keras.layers.LSTM(temp_unit, return_sequences=True)(input2)
    attention_blcok2 = keras.layers.LSTM(temp_unit, return_sequences=True)(input3)


    # pre_model = basic_block(shape_list[0])
    # acc_model = basic_block(shape_list[1])
    # gyr_model = basic_block(shape_list[2])

    # preflatten1 = pre_model(input1)
    # preflatten2 = acc_model(input2)
    # preflatten3 = gyr_model(input3)
    #
    # flatten1 = keras.layers.Flatten()(preflatten1)
    # flatten2 = keras.layers.Flatten()(preflatten2)
    # flatten3 = keras.layers.Flatten()(preflatten3)
    #
    # # merge feature map
    # merged_layer = keras.layers.concatenate([flatten1, flatten2, flatten3])
    # merged_dense = keras.layers.Dense(units=1000, activation='relu')(merged_layer)
    # merged_batchnorm = keras.layers.BatchNormalization()(merged_dense)
    # merged_dropout = keras.layers.Dropout(0.7)(merged_batchnorm)
    # merged_class_layer = keras.layers.Dense(units=nb_class, activation='softmax')(merged_dropout)
    #
    # model = keras.models.Model(inputs=[input1, input2, input3], output=merged_class_layer)
    # model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
    #               loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])


def attention_block(hidden_states):
    hidden_size = int(hidden_states.shape[2])
    # Inside dense layer
    #              hidden_states            dot               W            =>           score_first_part
    # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
    # W is the trainable weight matrix of attention Luong's multiplicative style score
    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
    #            score_first_part           dot        last_hidden_state     => attention_weights
    # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
    score = dot([score_first_part, h_t], [2, 1], name='attention_score')
    attention_weights = Activation('softmax', name='attention_weight')(score)
    # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
    context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
    pre_activation = Concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(128, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
    return attention_vector