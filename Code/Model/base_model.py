import tensorflow as tf
import tensorflow.keras as keras


# multi modal DCNN
def dcnn_network(shape_list, nb_class, comb_degree):
    nb_filter = 32
    nb_strides = 1
    kernel_size = 20  # default = 20
    
    if int(comb_degree) < 3:
        fc_unit = 3000
    elif int(comb_degree) == 3:
        # nb_filter = nb_filter * 2
        fc_unit = 3000
        nb_strides = 2
    elif int(comb_degree) == 4:
        # nb_filter = nb_filter * 2
        fc_unit = 5000
        nb_strides = 2
    elif int(comb_degree) == 5:
        # nb_filter = nb_filter * 2
        fc_unit = 8000
        nb_strides = 2
    
    # first network
    input1 = keras.layers.Input(shape=shape_list[0])
    input1_cnn1 = keras.layers.Conv1D(filters=nb_filter, kernel_size=kernel_size,
                                      strides=nb_strides, activation='relu')(input1)
    input1_batchnorm1 = keras.layers.BatchNormalization()(input1_cnn1)
    # input1_batchnorm1 = keras.layers.MaxPooling1D(pool_size=2, strides=nb_strides)(input1_batchnorm1)


    input1_cnn2 = keras.layers.Conv1D(filters=nb_filter * 2, kernel_size=kernel_size,
                                      strides=nb_strides, activation='relu')(input1_batchnorm1)
    input1_batchnorm2 = keras.layers.BatchNormalization()(input1_cnn2)
    # input1_batchnorm2 = keras.layers.MaxPooling1D(pool_size=2, strides=nb_strides)(input1_batchnorm2)

    input1_cnn3 = keras.layers.Conv1D(filters=nb_filter * 4, kernel_size=kernel_size,
                                      strides=nb_strides, activation='relu')(input1_batchnorm2)
    input1_batchnorm3 = keras.layers.BatchNormalization()(input1_cnn3)
    # input1_batchnorm3 = keras.layers.MaxPooling1D(pool_size=2, strides=nb_strides)(input1_batchnorm3)
    input1_flatten = keras.layers.Flatten()(input1_batchnorm3)

    # second network
    input2 = keras.layers.Input(shape=shape_list[1])
    input2_cnn1 = keras.layers.Conv1D(filters=nb_filter, kernel_size=kernel_size,
                                      strides=nb_strides, activation='relu')(input2)
    input2_batchnorm1 = keras.layers.BatchNormalization()(input2_cnn1)
    # input2_batchnorm1 = keras.layers.MaxPooling1D(pool_size=2, strides=nb_strides)(input2_batchnorm1)

    input2_cnn2 = keras.layers.Conv1D(filters=nb_filter * 2, kernel_size=kernel_size,
                                      strides=nb_strides, activation='relu')(input2_batchnorm1)
    input2_batchnorm2 = keras.layers.BatchNormalization()(input2_cnn2)
    # input2_batchnorm2 = keras.layers.MaxPooling1D(pool_size=2, strides=nb_strides)(input2_batchnorm2)

    input2_cnn3 = keras.layers.Conv1D(filters=nb_filter * 4, kernel_size=kernel_size,
                                      strides=nb_strides, activation='relu')(input2_batchnorm2)
    input2_batchnorm3 = keras.layers.BatchNormalization()(input2_cnn3)
    # input2_batchnorm3 = keras.layers.MaxPooling1D(pool_size=2, strides=nb_strides)(input2_batchnorm3)
    input2_flatten = keras.layers.Flatten()(input2_batchnorm3)

    # third network
    input3 = keras.layers.Input(shape=shape_list[2])
    input3_cnn1 = keras.layers.Conv1D(filters=nb_filter, kernel_size=kernel_size,
                                      strides=nb_strides, activation='relu')(input3)
    input3_batchnorm1 = keras.layers.BatchNormalization()(input3_cnn1)
    # input3_batchnorm1 = keras.layers.MaxPooling1D(pool_size=2, strides=nb_strides)(input3_batchnorm1)

    input3_cnn2 = keras.layers.Conv1D(filters=nb_filter * 2, kernel_size=kernel_size,
                                      strides=nb_strides, activation='relu')(input3_batchnorm1)
    input3_batchnorm2 = keras.layers.BatchNormalization()(input3_cnn2)
    # input3_batchnorm2 = keras.layers.MaxPooling1D(pool_size=2, strides=nb_strides)(input3_batchnorm2)

    input3_cnn3 = keras.layers.Conv1D(filters=nb_filter * 4, kernel_size=kernel_size,
                                      strides=nb_strides, activation='relu')(input3_batchnorm2)
    input3_batchnorm3 = keras.layers.BatchNormalization()(input3_cnn3)
    # input3_batchnorm3 = keras.layers.MaxPooling1D(pool_size=2, strides=nb_strides)(input3_batchnorm3)
    input3_flatten = keras.layers.Flatten()(input3_batchnorm3)

    # merge feature map
    merged_layer = keras.layers.concatenate([input1_flatten, input2_flatten, input3_flatten])
    # merged_layer = keras.layers.Dense(units=fc_unit, activation='relu')(merged_layer)
    # merged_layer = keras.layers.BatchNormalization()(merged_layer)
    # merged_layer = keras.layers.Dense(units=fc_unit, activation='relu')(merged_layer)
    merged_layer = keras.layers.Dense(units=256, activation='relu')(merged_layer)
    # merged_layer = keras.layers.BatchNormalization()(merged_layer)
    merged_layer = keras.layers.Dropout(0.7)(merged_layer)
    merged_class_layer = keras.layers.Dense(units=nb_class, activation='softmax')(merged_layer)

    model = keras.models.Model([input1, input2, input3], merged_class_layer)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss=keras.losses.categorical_crossentropy,
                       metrics=['accuracy'])
    return model