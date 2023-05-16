import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.initializers import RandomNormal, truncated_normal, constant, he_normal, GlorotNormal
from keras.layers import LeakyReLU, Concatenate

import numpy as np


class NetBlocks(object):
    @staticmethod
    def build_baseline_discriminator(img_shape, nfilters=32, batch_norm=None, layer_norm=False, summary=True):
        """ basic discriminator structure
        :param img_shape: dimension of the input image (nx,ny,nc)
        :param nfilters: number of base filters
        :param batch_norm: if is not None, batch normalization with given momentum
        :param layer_norm: True to apply layer normalization
        :param summary: display network summary if True
        :return: keras Functional model
        """
        input_layer = layers.Input(shape=img_shape)  # input layer from the img shape

        d = layers.Conv2D(nfilters, (5, 5), padding='same', use_bias=True,
                          bias_initializer=constant(0.), kernel_initializer=RandomNormal(stddev=0.02))(input_layer)
        d = layers.LeakyReLU(0.2)(d)

        d = layers.Conv2D(2 * nfilters, (3, 3), padding='same', strides=(2, 2),
                          use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(d)
        if batch_norm:
            d = layers.BatchNormalization(momentum=batch_norm)(d)
        if layer_norm is True:
            d = layers.LayerNormalization()(d)
        d = layers.LeakyReLU(0.2)(d)

        d = layers.Conv2D(4 * nfilters, (3, 3), padding='same', strides=(2, 2),
                          use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(d)
        if batch_norm:
            d = layers.BatchNormalization(momentum=batch_norm)(d)
        if layer_norm is True:
            d = layers.LayerNormalization()(d)
        d = layers.LeakyReLU(0.2)(d)

        d = layers.Conv2D(8 * nfilters, (3, 3), padding='same', strides=(2, 2),
                          use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(d)
        if batch_norm:
            d = layers.BatchNormalization(momentum=batch_norm)(d)
        if layer_norm is True:
            d = layers.LayerNormalization()(d)
        d = layers.LeakyReLU(0.2)(d)

        d = layers.Flatten()(d)
        out = layers.Dense(1, activation='linear', kernel_initializer=RandomNormal(stddev=0.02))(d)
        discriminator = Model(inputs=[input_layer], outputs=[out], name='Discriminator')
        if summary is True:
            discriminator.summary()
        return discriminator

    @staticmethod
    def build_baseline_generator(latent_dim, img_shape, nfilters=32, batch_norm=None, summary=True):
        """ basic generator structure
        :param latent_dim: dimension of the latent vector (nz)
        :param img_shape: dimension of the input image (nx,ny,nc)
        :param nfilters: number of base filters
        :param batch_norm: if is not None, batch normalization with given momentum
        :param summary: display network summary if True
        :return: keras Functional model
        """
        ni, nj = img_shape[0], img_shape[1]
        nchannels = img_shape[2]

        input_layer = layers.Input(shape=(latent_dim,))
        g = layers.Dense(np.ceil(ni / 8).astype(int) * np.ceil(nj / 8).astype(int) * 8 * nfilters,
                         use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(input_layer)
        if batch_norm:
            g = layers.BatchNormalization(momentum=batch_norm)(g)
        g = layers.Reshape((np.ceil(ni / 8).astype(int), np.ceil(nj / 8).astype(int), 8 * nfilters))(g)

        g = layers.Conv2D(8 * nfilters, (3, 3), padding='same',
                          use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(g)
        if batch_norm:
            g = layers.BatchNormalization(momentum=batch_norm)(g)
        g = layers.ReLU()(g)

        g = tf.image.resize(g, size=(np.ceil(ni / 4).astype(int), np.ceil(nj / 4).astype(int)), method='nearest')
        g = layers.Conv2D(4 * nfilters, (3, 3), padding='same',
                          use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(g)
        if batch_norm:
            g = layers.BatchNormalization(momentum=batch_norm)(g)
        g = layers.ReLU()(g)

        g = tf.image.resize(g, size=(np.ceil(ni / 2).astype(int), np.ceil(nj / 2).astype(int)), method='nearest')
        g = layers.Conv2D(2 * nfilters, (3, 3), padding='same',
                          use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(g)
        if batch_norm:
            g = layers.BatchNormalization(momentum=batch_norm)(g)
        g = layers.ReLU()(g)

        g = tf.image.resize(g, size=(np.ceil(ni / 1).astype(int), np.ceil(nj / 1).astype(int)), method='nearest')
        g = layers.Conv2D(1 * nfilters, (3, 3), padding='same',
                          use_bias=False, kernel_initializer=RandomNormal(stddev=0.02))(g)
        if batch_norm:
            g = layers.BatchNormalization(momentum=batch_norm)(g)
        g = layers.ReLU()(g)

        image = layers.Conv2D(nchannels, (5, 5), padding='same', activation='tanh', name='last_conv')(g)
        image = tf.image.resize(image, size=img_shape[0:2])  # assert correct output shape
        generator = Model(inputs=[input_layer], outputs=[image], name='Generator')
        if summary is True:
            generator.summary()
        return generator

