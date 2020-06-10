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
    parser.add_argument('-ld', '--latent_dim', type=int, default=100, help='shape of latent dim')
    args = parser.parse_args()
    return args


class SSGAN():
    """This is a simplified version of semi-supervised GAN"""
    def __init__(self, inputshape, outputshape, latent_dim):
        self.inshp = inputshape
        self.outshp = outputshape
        self.ld = latent_dim

    # def __call__(self, *args, **kwargs):

    def __call__(self, inputshape, outputshape, latent_dim):
        self.inshp = inputshape
        self.outshp = outputshape
        self.ld = latent_dim

    def builddiscriminator(self):
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


    def buildgenerator(self):
        x = Input(shape=(self.ld, ), name='gen_input')
        gen_dense = Dense(units=7*7*128, name='gen_dense')(x)
        gen_act = LeakyReLU(alpha=0.2, name='LeakyRelu_1')(gen_dense)
        reshape = Reshape(target_shape=(7, 7, 128), name='reshape')(gen_act)
        gen_tconv1 = keras.layers.Conv2DTranspose(filters=128, strides=2, kernel_size=3,
                                                  padding='same', name='gen_conv1')(reshape)
        gen_act = LeakyReLU(alpha=0.2, name='LeakyRelu_2')(gen_tconv1)
        gen_tconv2 = keras.layers.Conv2DTranspose(filters=128, strides=2, kernel_size=3,
                                                  padding='same', name='gen_conv2')(gen_act)
        gen_act = LeakyReLU(alpha=0.2, name='LeakyRelu_3')(gen_tconv2)
        gen_out = Conv2D(filters=1, strides=1, kernel_size=5, padding='same', name='gen_output')(gen_act)

        gen_model = keras.models.Model(inputs=x, outputs=gen_out)
        return gen_model

    def buildgan(self):
        pass











def main():
    params = setparams()
    # print(params.input_shape, type(params.input_shape), type(params.input_shape[0]), params.input_shape[0])
    # print(params.output_shape)
    ssgan = SSGAN(inputshape=params.input_shape, outputshape=params.output_shape, latent_dim=params.latent_dim)
    d_model, c_model = ssgan.builddiscriminator()
    gen_model = ssgan.buildgenerator()
    gen_model.summary()
    path = os.path.join(os.getcwd(), 'pics')
    if not os.path.exists(path=path):
        os.mkdir(path=path)

    keras.utils.plot_model(model=d_model, to_file=os.path.join(path, 'discriminative.png'))
    keras.utils.plot_model(model=c_model, to_file=os.path.join(path, 'classifier_model.png'))
    keras.utils.plot_model(model=gen_model, to_file=os.path.join(path, 'generator.png'), show_shapes=True)

if __name__ == '__main__':
    main()















