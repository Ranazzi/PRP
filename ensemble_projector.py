import numpy as np
import tensorflow as tf
from GAN.netblocks import NetBlocks
from tensorflow.keras import Model, layers
from da_methods import Esmda

from math_utils import normalized_data_mismatch


class EnsembleProjector(object):
    def __init__(self, net, img_shape, latent_dim, size=5000, upscaling_rate=4,
                 loss_net_folder='./classifier_weights'):
        self.generator = net
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.N = size

        self.upscaling_rate = upscaling_rate

        print('loading loss network')
        RS = NetBlocks.build_classifier(img_shape=(48, 48, 1), nclasses=6, nfilters=16)
        RS.load_weights('{}/weights'.format(loss_net_folder))

        self.Loss_net = Model(inputs=RS.input,
                              outputs=[layers.GlobalAveragePooling2D()(RS.layers[3].output),
                                       layers.GlobalAveragePooling2D()(RS.layers[9].output),
                                       layers.GlobalAveragePooling2D()(RS.layers[16].output),
                                       layers.GlobalAveragePooling2D()(RS.layers[39].output),
                                       layers.GlobalAveragePooling2D()(RS.layers[53].output),
                                       layers.GlobalAveragePooling2D()(RS.layers[-5].output)])  # first dense

    def project(self, sample, steps, m_error):
        def step(z, dout, dr, steps, Cd):
            dreal = np.repeat(dr, dout.shape[-1], axis=1)
            # za = esmda(z, dout, dreal, Cd).subspace(steps, 0.9)  # Nd > Ne
            za = Esmda(z, dout, dreal, Cd).explicit(steps)  # Nd > Ne
            return za

        obj = []

        z = np.random.normal(0, 2, (self.N, self.latent_dim)).T  # generate initial latent ensemble

        sample = tf.image.resize(sample, [48, 48])

        feature_real = self.Loss_net(sample)
        feature_real = np.concatenate((feature_real[0].numpy(), feature_real[1].numpy(),
                                       feature_real[2].numpy(), feature_real[3].numpy(),
                                       feature_real[4].numpy(), feature_real[5].numpy()), axis=-1)

        d_real_0 = tf.image.resize(sample, [46 // self.upscaling_rate, 69 // self.upscaling_rate]).numpy().T.reshape(
            (46 // self.upscaling_rate) * (69 // self.upscaling_rate), 1)

        d_real_1 = feature_real.T.reshape(-1, feature_real.T.shape[-1])  # convert feature map to measurements vector

        d_real = np.concatenate((d_real_0, d_real_1), axis=0)

        c_array = np.ones(d_real.shape[0]) * m_error if m_error.__len__() == 1 else c_array = m_error

        Cd = np.diag(c_array)

        for i in range(steps):
            d_pred = self.generator(z.T).numpy()  # run outputs = generator(latents)
            d_pred_0 = tf.image.resize(d_pred,
                                       [46 // self.upscaling_rate, 69 // self.upscaling_rate]).numpy().T.reshape(
                (46 // self.upscaling_rate) * (69 // self.upscaling_rate),
                d_pred.shape[0])
            d_pred = tf.image.resize(d_pred, [48, 48])

            feature_pred = self.Loss_net(d_pred)  # get predicted feature map
            feature_pred = np.concatenate((feature_pred[0].numpy(), feature_pred[1].numpy(),
                                           feature_pred[2].numpy(), feature_pred[3].numpy(),
                                           feature_pred[4].numpy(), feature_pred[5].numpy()), axis=-1)

            d_pred_1 = feature_pred.T.reshape(-1, feature_pred.T.shape[-1])  # convert feature map to data vector

            d_pred = np.concatenate((d_pred_0, d_pred_1), axis=0)

            # d_pred = d_pred.reshape((self.N, 46*69)).T


            # ofunc = tf.reduce_mean(tf.square(d_pred - d_real)).numpy()
            ofunc = normalized_data_mismatch(d_pred, d_real, Cd)
            obj.append(ofunc)


            # alpha = steps
            # alpha = np.min([800 * ofunc, 800])
            # beta = beta + (1/alpha)
            # if (beta > (1-1/800)) or (i == (steps-1)):
            #     alpha = 1/(1-beta)
            #     beta = 1

            # print('assimilating')
            z = step(z, d_pred, d_real, steps, Cd)  # run esmda step

            # for _1 in range(z.shape[0]):
            #     for _2 in range(z.shape[1]):
            #         if z[_1, _2] > 5 or z[_1, _2] < -5:
            #             z[_1, _2] = np.random.normal()

            # print('step {} of {} done'.format(i + 1, steps))
        return z, obj, d_real, Cd
