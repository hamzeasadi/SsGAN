import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, Flatten, Input
from keras.layers import LeakyReLU, Reshape, ReLU, BatchNormalization
import numpy as np
import os
import argparse
from matplotlib import pyplot as plt


def setparams():
    parser = argparse.ArgumentParser(description='set input and output shape of our network')
    parser.add_argument('-i', '--input_shape', nargs='+', type=int, default=(28, 28, 1), help='input shape')
    parser.add_argument('-o', '--output_shape', type=int, default=11, help='output shape')
    args = parser.parse_args()
    return args


class SSGAN():
    """This is a simplified version of semi-supervised GAN"""
    def __init__(self, inputshape, outputshape):
        self.inshp = inputshape
        self.outshp = outputshape

    # def __call__(self, *args, **kwargs):

    def __call__(self, inputshape, outputshape):
        self.inshp = inputshape
        self.outshp = outputshape

    def buildModel(self):
        x = Input(shape=self.inshp, name='dis_input')
        dis_conv1 = Conv2D(filters=128, kernel_size=3, strides=2, padding ='same', name='dis_conv1')(x)
        dis_act = LeakyReLU(alpha=0.2, name='leakyrelu1')(dis_conv1)
        dis_conv2 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', name='dis_conv2')(dis_act)
        dis_act = LeakyReLU(alpha=0.2, name='leakyrelu2')(dis_conv2)
        dis_conv3 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', name="dis_conv3")(dis_act)
        dis_act = LeakyReLU(alpha=0.2, name='leakyrelu3')(dis_conv3)

        flatten = Flatten()(dis_act)
        dropout = keras.layers.Dropout(rate=0.4)(flatten)
        dis_output = Dense(units=1, activation='sigmoid', name='dis_output')(dropout)

        dis_model = keras.models.Model(inputs=x, outputs=dis_output, name='dis_model')
        dis_opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        dis_loss = keras.losses.BinaryCrossentropy()
        dis_model.compile(loss=dis_loss, optimizer=dis_opt, metrics=['accuracy'])

        c_output = Dense(units=self.outshp, activation='softmax', name='c_output')(dropout)
        c_model = keras.models.Model(inputs=x, outputs=c_output, name='c_model')
        c_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=dis_opt, metrics=['accuracy'])

        return dis_model, c_model

    








def main():
    params = setparams()
    # print(params.input_shape, type(params.input_shape), type(params.input_shape[0]), params.input_shape[0])
    # print(params.output_shape)
    ssgan = SSGAN(inputshape=params.input_shape, outputshape=params.output_shape)
    d_model, c_model = ssgan.buildModel()
    d_model.summary()
    path = os.path.join(os.getcwd(), 'pics')
    if not os.path.exists(path=path):
        os.mkdir(path=path)

    keras.utils.plot_model(model=d_model, to_file=os.path.join(path, 'discriminative.png'))
    keras.utils.plot_model(model=c_model, to_file=os.path.join(path, 'classifier_model.png'))

if __name__ == '__main__':
    main()















