#projector
import tensorflow as tf
import numpy as np


class GradientProjector(object):
    def __init__(self, net, img_shape, latent_dim):
        self.generator = net
        self.img_shape = img_shape
        self.latent_dim = latent_dim

        self.dec = self.generator

        self.opt = tf.keras.optimizers.Adam(learning_rate=0.1, beta_1=0.5, beta_2=0.9)  # optimizer

    def project(self, sample, steps):

        @tf.function
        def step(z2, real):
            with tf.GradientTape() as tape:
                generated = self.generator(z2, training=False)
                real = tf.image.resize(real, [48, 48])
                generated = tf.image.resize(generated, [48, 48])

                loss = tf.reduce_sum(tf.square(generated - real))

                grad = tape.gradient(loss, [z])
                self.opt.apply_gradients(zip(grad, [z]))

                ## stochastic clipping
                # rnd = tf.random.normal([1, 512], stddev=1)
                # z.assign(tf.where(z>5.5, rnd, z))
                # z.assign(tf.where(z<-5.5, rnd, z))

            return z.value(), loss

        obj = []

        z_t = tf.random.normal([1, 512], stddev=1)
        # z = tf.zeros([1, 512])
        z = tf.Variable(z_t, trainable=True)
        sample = tf.image.resize(sample, [48, 48])

        # feature_real = self.FRD(sample)

        for i in range(steps):

            z_, obj_ = step(z, np.nan_to_num(sample))

            # z_t = 0.9 * z_t + 0.1*z_

            obj.append(obj_)

            if np.argmin(np.array(obj)) == i:
              z_t = z_

        return z_t, obj
