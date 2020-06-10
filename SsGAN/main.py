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
    parser.add_argument('-o', '--output_shape', type=int, default=10, help='output shape')
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
        c_model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=dis_opt, metrics=['accuracy'])

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
        d_model, c_model = self.builddiscriminator()
        g_model = self.buildgenerator()
        d_model.trainable = False
        gan_input = g_model.input
        d_input = g_model.output
        gan_output = d_model(d_input)
        gan_model = keras.models.Model(inputs=gan_input, outputs=gan_output, name='gan_model')
        gan_opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        gan_loss = keras.losses.BinaryCrossentropy()
        gan_model.compile(loss=gan_loss, optimizer=gan_opt, metrics=['accuracy'])
        return gan_model

    def load_data(self):
        mnist = keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        X = x_train.astype('float32')
        X = np.expand_dims(X, axis=-1)
        X = (X - 127.5)/127.5
        return X, y_train

    def select_supervised_samples(self, dataset, n_samples=100, n_classes=10):
        X, y = dataset
        X_list, y_list = list(), list()
        n_per_class = int(n_samples / n_classes)
        for i in range(n_classes):
            X_with_class = X[y == i]
            ix = np.random.randint(0, len(X_with_class), n_per_class)
            [X_list.append(X_with_class[j]) for j in ix]
            [y_list.append(i) for j in ix]
        return np.asarray(X_list), np.asarray(y_list)

    def generate_real_samples(self, dataset, n_samples):
        images, labels = dataset
        ix = np.random.randint(0, images.shape[0], n_samples)
        X, labels = images[ix], labels[ix]
        y = np.ones((n_samples, 1))
        return [X, labels], y

    def generate_latent_points(self, latent_dim, n_samples):
        z_input = np.random.randn(latent_dim * n_samples)
        z_input = z_input.reshape(n_samples, latent_dim)
        return z_input

    def generate_fake_samples(self, generator, latent_dim, n_samples):
        z_input = self.generate_latent_points(latent_dim, n_samples)
        images = generator.predict(z_input)
        y = np.zeros((n_samples, 1))
        return images, y

    def train(self, g_model, d_model, c_model, gan_model, n_epochs=20, n_batch=100):
        path = os.path.join(os.getcwd(), 'models')
        if not os.path.exists(path):
            os.mkdir(path=path)
        dataset = self.load_data()
        latent_dim = self.ld
        X_sup, y_sup = self.select_supervised_samples(dataset)
        bat_per_epo = int(dataset[0].shape[0] / n_batch)
        n_steps = bat_per_epo * n_epochs
        half_batch = int(n_batch / 2)
        for i in range(n_steps):
            [Xsup_real, ysup_real], _ = self.generate_real_samples([X_sup, y_sup], half_batch)
            # ysup_real = keras.utils.to_categorical(ysup_real, num_classes=10)
            c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)
            [X_real, _], y_real = self.generate_real_samples(dataset, half_batch)
            d_loss1 = d_model.train_on_batch(X_real, y_real)
            X_fake, y_fake = self.generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss2 = d_model.train_on_batch(X_fake, y_fake)
            X_gan, y_gan = self.generate_latent_points(latent_dim, n_batch), np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i + 1, c_loss, c_acc * 100, d_loss1, d_loss2, g_loss))
            # # evaluate the model performance every so often
            if (i + 1) % (bat_per_epo * 1) == 0:
                print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i + 1, c_loss, c_acc * 100, d_loss1, d_loss2, g_loss))
                # summarize_performance(i, g_model, c_model, latent_dim, dataset)

        g_model.save(filepath=os.path.join(path, 'gen_model.h5'))
        d_model.save(filepath=os.path.join(path, 'd_model.h5'))
        c_model.save(filepath=os.path.join(path, 'c_model.h5'))












def main():
    params = setparams()
    # print(params.input_shape, type(params.input_shape), type(params.input_shape[0]), params.input_shape[0])
    # print(params.output_shape)
    ssgan = SSGAN(inputshape=params.input_shape, outputshape=params.output_shape, latent_dim=params.latent_dim)
    d_model, c_model = ssgan.builddiscriminator()
    gen_model = ssgan.buildgenerator()
    gan_model = ssgan.buildgan()
    gan_model.summary()
    path = os.path.join(os.getcwd(), 'pics')
    if not os.path.exists(path=path):
        os.mkdir(path=path)

    ssgan.train(gen_model, d_model, c_model, gan_model, n_epochs=20, n_batch=100)
    #
    # keras.utils.plot_model(model=d_model, to_file=os.path.join(path, 'discriminative.png'))
    # keras.utils.plot_model(model=c_model, to_file=os.path.join(path, 'classifier_model.png'))
    # keras.utils.plot_model(model=gen_model, to_file=os.path.join(path, 'generator.png'), show_shapes=True)
    # keras.utils.plot_model(model=gan_model, to_file=os.path.join(path, 'gan_model.png'), show_shapes=True)

if __name__ == '__main__':
    main()















