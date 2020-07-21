import tensorflow as tf
import numpy as np
import keras
from keras.applications import VGG16
from keras_applications.imagenet_utils import _obtain_input_shape


def trained_vgg_builder(shape_list, nb_class, trainable):

    input_layers = list()
    resnet_layers = list()

    for input_shape in shape_list:

        input_layer = keras.layers.Input(shape=(input_shape[0], input_shape[1], 1))
        one_conv_layer = keras.layers.Conv2D(3, (1, 1), padding='same', activation='linear')(input_layer)
        one_conv_layer = keras.layers.Dense(units=256, activation='linear')(one_conv_layer)
        input_layers.append(one_conv_layer)

        pvgg16 = VGG16(weights='imagenet', include_top=False)
        pvgg16.trainable = trainable
        resnet_layers.append(pvgg16(one_conv_layer))

    merged_layer = keras.layers.concatenate(resnet_layers)
    merged_dense = keras.layers.Dense(units=1000, activation='relu')(merged_layer)
    merged_batchnorm = keras.layers.BatchNormalization()(merged_dense)
    merged_dropout = keras.layers.Dropout(0.7)(merged_batchnorm)
    merged_class_layer = keras.layers.Dense(units=nb_class, activation='softmax')(merged_dropout)

    model = keras.models.Model(inputs=input_layers, output=merged_class_layer)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
                  loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    return model


def vgg_builder(shape_list, nb_class):

    input_layers = list()
    vgg_layers = list()

    for input_shape in shape_list:
        input_layer = keras.layers.Input(shape=input_shape)
        input_layers.append(input_layer)
        one_conv_layer = keras.layers.Conv2D(3, (1, 1), padding='same', activation='linear')(input_layer)
        pvgg16 = VGG16(weights='imagenet', include_top=False)(one_conv_layer)
        pvgg16.trainable = False
        vgg_layers.append(pvgg16)

    merged_layer = keras.layers.concatenate(vgg_layers)
    merged_dense = keras.layers.Dense(units=1000, activation='relu')(merged_layer)
    merged_batchnorm = keras.layers.BatchNormalization()(merged_dense)
    merged_dropout = keras.layers.Dropout(0.7)(merged_batchnorm)
    merged_class_layer = keras.layers.Dense(units=nb_class, activation='softmax')(merged_dropout)

    model = keras.models.Model(inputs=input_layers, output=merged_class_layer)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
                  loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    return model
