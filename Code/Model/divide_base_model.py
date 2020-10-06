import tensorflow as tf
import tensorflow.keras as keras


# multi modal DCNN
def dcnn_network(shape_list, nb_class, comb_degree):
    nb_filter = 32
    nb_strides = 1
    
    if int(comb_degree) < 3:
        fc_unit = 3000
    elif int(comb_degree) == 3:
        nb_filter = nb_filter * 2
        fc_unit = 3000
        nb_strides = 2
    elif int(comb_degree) == 4:
        nb_filter = nb_filter * 2
        fc_unit = 5000
        nb_strides = 2
    elif int(comb_degree) == 5:
        nb_filter = nb_filter * 2
        fc_unit = 8000
        nb_strides = 2
    
    # first network
    input1 = keras.layers.Input(shape=shape_list[0])
    input1_cnn1 = keras.layers.Conv1D(filters=nb_filter, kernel_size=20,
                                      strides=nb_strides, activation='relu')(input1)
    input1_cnn1 = keras.layers.MaxPooling1D(pool_size=2, strides=nb_strides)(input1_cnn1)
    # input1_batchnorm1 = keras.layers.BatchNormalization()(input1_cnn1)

    input1_cnn2 = keras.layers.Conv1D(filters=nb_filter * 2, kernel_size=20,
                                      strides=nb_strides, activation='relu')(input1_cnn1)
    input1_cnn2 = keras.layers.MaxPooling1D(pool_size=2, strides=nb_strides)(input1_cnn2)
    # input1_batchnorm2 = keras.layers.BatchNormalization()(input1_cnn2)

    input1_cnn3 = keras.layers.Conv1D(filters=nb_filter * 4, kernel_size=20,
                                      strides=nb_strides, activation='relu')(input1_cnn2)
    input1_cnn3 = keras.layers.MaxPooling1D(pool_size=2, strides=nb_strides)(input1_cnn3)
    input1_flatten = keras.layers.Flatten()(input1_cnn3)
    # input1_batchnorm3 = keras.layers.BatchNormalization()(input1_cnn3)
    # input1_flatten = keras.layers.Flatten()(input1_batchnorm3)

    # second network
    left_input2 = keras.layers.Input(shape=shape_list[1])
    left_input2_cnn1 = keras.layers.Conv1D(filters=nb_filter, kernel_size=20,
                                      strides=nb_strides, activation='relu')(left_input2)
    left_input2_cnn1 = keras.layers.MaxPooling1D(pool_size=2, strides=nb_strides)(left_input2_cnn1)
    # input2_batchnorm1 = keras.layers.BatchNormalization()(input2_cnn1)

    left_input2_cnn2 = keras.layers.Conv1D(filters=nb_filter * 2, kernel_size=20,
                                      strides=nb_strides, activation='relu')(left_input2_cnn1)
    left_input2_cnn2 = keras.layers.MaxPooling1D(pool_size=2, strides=nb_strides)(left_input2_cnn2)
    # input2_batchnorm2 = keras.layers.BatchNormalization()(input2_cnn2)

    left_input2_cnn3 = keras.layers.Conv1D(filters=nb_filter * 4, kernel_size=20,
                                      strides=nb_strides, activation='relu')(left_input2_cnn2)
    left_input2_cnn3 = keras.layers.MaxPooling1D(pool_size=2, strides=nb_strides)(left_input2_cnn3)
    # input2_batchnorm3 = keras.layers.BatchNormalization()(input2_cnn3)
    left_input2_flatten = keras.layers.Flatten()(left_input2_cnn3)

    right_input2 = keras.layers.Input(shape=shape_list[2])
    right_input2_cnn1 = keras.layers.Conv1D(filters=nb_filter, kernel_size=20,
                                      strides=nb_strides, activation='relu')(right_input2)
    right_input2_cnn1 = keras.layers.MaxPooling1D(pool_size=2, strides=nb_strides)(right_input2_cnn1)
    # input2_batchnorm1 = keras.layers.BatchNormalization()(input2_cnn1)

    right_input2_cnn2 = keras.layers.Conv1D(filters=nb_filter * 2, kernel_size=20,
                                      strides=nb_strides, activation='relu')(right_input2_cnn1)
    right_input2_cnn2 = keras.layers.MaxPooling1D(pool_size=2, strides=nb_strides)(right_input2_cnn2)
    # input2_batchnorm2 = keras.layers.BatchNormalization()(input2_cnn2)

    right_input2_cnn3 = keras.layers.Conv1D(filters=nb_filter * 4, kernel_size=20,
                                      strides=nb_strides, activation='relu')(right_input2_cnn2)
    right_input2_cnn3 = keras.layers.MaxPooling1D(pool_size=2, strides=nb_strides)(right_input2_cnn3)
    # input2_batchnorm3 = keras.layers.BatchNormalization()(input2_cnn3)
    right_input2_flatten = keras.layers.Flatten()(right_input2_cnn3)

    # third network
    left_input3 = keras.layers.Input(shape=shape_list[3])
    left_input3_cnn1 = keras.layers.Conv1D(filters=nb_filter, kernel_size=20,
                                      strides=nb_strides, activation='relu')(left_input3)
    left_input3_cnn1 = keras.layers.MaxPooling1D(pool_size=2, strides=nb_strides)(left_input3_cnn1)
    # input3_batchnorm1 = keras.layers.BatchNormalization()(input3_cnn1)

    left_input3_cnn2 = keras.layers.Conv1D(filters=nb_filter * 2, kernel_size=20,
                                      strides=nb_strides, activation='relu')(left_input3_cnn1)
    left_input3_cnn2 = keras.layers.MaxPooling1D(pool_size=2, strides=nb_strides)(left_input3_cnn2)
    # input3_batchnorm2 = keras.layers.BatchNormalization()(input3_cnn2)

    left_input3_cnn3 = keras.layers.Conv1D(filters=nb_filter * 4, kernel_size=20,
                                      strides=nb_strides, activation='relu')(left_input3_cnn2)
    left_input3_cnn3 = keras.layers.MaxPooling1D(pool_size=2, strides=nb_strides)(left_input3_cnn3)
    # input3_batchnorm3 = keras.layers.BatchNormalization()(input3_cnn3)
    left_input3_flatten = keras.layers.Flatten()(left_input3_cnn3)

    right_input3 = keras.layers.Input(shape=shape_list[4])
    right_input3_cnn1 = keras.layers.Conv1D(filters=nb_filter, kernel_size=20,
                                      strides=nb_strides, activation='relu')(right_input3)
    right_input3_cnn1 = keras.layers.MaxPooling1D(pool_size=2, strides=nb_strides)(right_input3_cnn1)
    # input3_batchnorm1 = keras.layers.BatchNormalization()(input3_cnn1)

    right_input3_cnn2 = keras.layers.Conv1D(filters=nb_filter * 2, kernel_size=20,
                                      strides=nb_strides, activation='relu')(right_input3_cnn1)
    right_input3_cnn2 = keras.layers.MaxPooling1D(pool_size=2, strides=nb_strides)(right_input3_cnn2)
    # input3_batchnorm2 = keras.layers.BatchNormalization()(input3_cnn2)

    right_input3_cnn3 = keras.layers.Conv1D(filters=nb_filter * 4, kernel_size=20,
                                      strides=nb_strides, activation='relu')(right_input3_cnn2)
    right_input3_cnn3 = keras.layers.MaxPooling1D(pool_size=2, strides=nb_strides)(right_input3_cnn3)
    # input3_batchnorm3 = keras.layers.BatchNormalization()(input3_cnn3)
    right_input3_flatten = keras.layers.Flatten()(right_input3_cnn3)

    # merge feature map

    input2_flatten = keras.layers.concatenate([left_input2_flatten, right_input2_flatten])
    input3_flatten = keras.layers.concatenate([left_input3_flatten, right_input3_flatten])

    merged_layer = keras.layers.concatenate([input1_flatten, input2_flatten, input3_flatten])
    # merged_dense1 = keras.layers.Dense(units=fc_unit, activation='relu')(merged_layer)
    # merged_batchnorm1 = keras.layers.BatchNormalization()(merged_dense1)
    merged_dense2 = keras.layers.Dense(units=fc_unit, activation='relu')(merged_layer)
    # merged_dense2 = keras.layers.Dense(units=256, activation='relu')(merged_layer)
    # merged_batchnorm2 = keras.layers.BatchNormalization()(merged_dense2)
    merged_dropout = keras.layers.Dropout(0.7)(merged_dense2)
    merged_class_layer = keras.layers.Dense(units=nb_class, activation='softmax')(merged_dropout)

    model = keras.models.Model([input1, [left_input2, right_input2], [left_input3, right_input3]], merged_class_layer)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss=keras.losses.categorical_crossentropy,
                       metrics=['accuracy'])
    return model