import keras


def basic_block(input_shape):
    model = keras.models.Sequential()
    channel_size = 8

    # encoding phase 1
    # model.add(keras.layers.Conv2D(channel_size, kernel_size=(1, 3), strides=(1, 2),
    #           input_shape=(batch_size, input_shape[0], input_shape[1]), padding='same'))
    model.add(keras.layers.Conv2D(channel_size, kernel_size=(1, 3), strides=(1, 2),
                                  input_shape=(input_shape[0], input_shape[1], 1), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    channel_size *= 2
    model.add(keras.layers.Conv2D(channel_size, kernel_size=(1, 3), strides=(1, 2), padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    # encoding phase 2
    channel_size *= 2
    model.add(keras.layers.Conv2D(channel_size, kernel_size=(3, 1), strides=(2, 1), padding='same'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(channel_size, kernel_size=(3, 1), strides=(2, 1), padding='same'))

    # model.add(keras.layers.Flatten())

    # choose this layer
    # model.add(keras.layers.Dense(125, activation='relu'))

    # output layer
    # model.add(keras.layers.Dense(target_class, activation='softmax'))

    return model


def basic_10_network(shape_list, nb_class):
    # Basic 10 Models
    input1 = keras.layers.Input(shape=(shape_list[0][0], shape_list[0][1], 1))
    input2 = keras.layers.Input(shape=(shape_list[1][0], shape_list[1][1], 1))
    input3 = keras.layers.Input(shape=(shape_list[2][0], shape_list[2][1], 1))

    pre_model = basic_block(shape_list[0])
    acc_model = basic_block(shape_list[1])
    gyr_model = basic_block(shape_list[2])

    preflatten1 = pre_model(input1)
    preflatten2 = acc_model(input2)
    preflatten3 = gyr_model(input3)

    flatten1 = keras.layers.Flatten()(preflatten1)
    flatten2 = keras.layers.Flatten()(preflatten2)
    flatten3 = keras.layers.Flatten()(preflatten3)

    # merge feature map
    merged_layer = keras.layers.concatenate([flatten1, flatten2, flatten3])
    merged_dense = keras.layers.Dense(units=1000, activation='relu')(merged_layer)
    merged_batchnorm = keras.layers.BatchNormalization()(merged_dense)
    merged_dropout = keras.layers.Dropout(0.7)(merged_batchnorm)
    merged_class_layer = keras.layers.Dense(units=nb_class, activation='softmax')(merged_dropout)

    model = keras.models.Model(inputs=[input1, input2, input3], output=merged_class_layer)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
                  loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    return model
