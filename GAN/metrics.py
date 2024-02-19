"""Metrics to validate GANs:
    FID: Frechet Inception Distance
    FRD: Frechet Reservoir Distance
"""
import numpy as np
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3

from tensorflow.keras import layers, Model
from tensorflow.keras.initializers import RandomNormal, he_normal, GlorotNormal
from tensorflow.image import resize
from scipy.linalg import sqrtm
from GAN.netblocks import NetBlocks

class GanMetrics(object):
    @staticmethod
    def frechet_reservoir_distance(x1, x2, folder='./classifier_weights'):
        x1 = resize(x1, [48, 48])
        x2 = resize(x2, [48, 48])

        print('loading network')
        RS = NetBlocks.build_classifier(img_shape=(48, 48, 1), nclasses=6, nfilters=16)
        RS.load_weights('{}/weights'.format(folder))
        FRD = Model(inputs=RS.input,
                    outputs=layers.GlobalAveragePooling2D()(RS.layers[-5].output))  # first dense

        # RS = tf.keras.models.load_model('classifier_weights/Reservoir_classifier_v2.h5')
        # FRD = Model(inputs=RS.input,
        #                      outputs=layers.GlobalAveragePooling2D()(RS.layers[-5].output))  # only feature block

        # get activation values
        act1 = FRD.predict(x1, verbose=0)
        act2 = FRD.predict(x2, verbose=0)

        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        frd = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return frd

    @staticmethod
    def frechet_inception_distance(x1, x2, shape=(128, 128), batch_size=None):
        # Frechet Inception distance (Heusel, 2018) https://arxiv.org/abs/1706.08500
        if batch_size is None:
            batch_size = x1.shape[0]
        if x1.shape[3] == 1:  # expand to three channels
            x1 = np.repeat(x1, 3, axis=3)
        if x2.shape[3] == 1:
            x2 = np.repeat(x2, 3, axis=3)  # expand to three channels

        iv3 = InceptionV3(include_top=False, pooling='avg', input_shape=(shape[0], shape[1], 3))
        images1 = resize(x1, [shape[0], shape[1]])
        images2 = resize(x2, [shape[0], shape[1]])

        def get_inception_probs(model, inps, b_size):
            # from original fid code https://github.com/tsc2017/Frechet-Inception-Distance
            nbatches = int(np.ceil(float(x1.shape[0]) / b_size))
            act = np.zeros([inps.shape[0], 2048], dtype=np.float32)
            for i in range(nbatches):
                # print('batch: {} in {}'.format(i, nbatches))
                inp = inps[i * b_size:(i + 1) * b_size]
                act[i * b_size: i * b_size + min(b_size, inp.shape[0])] = model.predict(inp, verbose=0)
            return act

        print('computing inception probs')
        act1 = get_inception_probs(iv3, images1, batch_size)
        act2 = get_inception_probs(iv3, images2, batch_size)

        # calculate mean and covariance statistics
        print('computing statistics')
        mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    @staticmethod
    def discriminator_outputs(model, batch):
        size = batch.shape[0]
        noise = np.random.normal(0, 1, (size, model.latent_dim))
        real2 = model.ema_discriminator.predict(batch, verbose=0)
        generated = model.ema_generator.predict(noise, verbose=0)
        fake2 = model.ema_discriminator.predict(generated, verbose=0)
        return real2, fake2



