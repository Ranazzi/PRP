# file with ADA pipeline
import tensorflow as tf
import keras
from tensorflow.keras import layers, Model


class Gate_Random(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(Gate_Random, self).__init__(*args, **kwargs)

    def call(self, inputs, augmented, probability):
        batch_size = tf.shape(inputs)[0]
        cond = tf.random.uniform(shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0)
        bools = tf.math.less(cond, probability)
        return tf.where(bools, augmented, inputs)


class Flip(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(Flip, self).__init__(*args, **kwargs)

    def call(self, images):
        # out = tf.identity(images)
        return tf.image.flip_left_right(images)


class Rotate_Int(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(Rotate_Int, self).__init__(*args, **kwargs)

    def call(self, images):
        batch_size = tf.shape(images)[0]
        angles = tf.random.uniform(shape=(batch_size,), minval=0, maxval=4, dtype='int32')
        elems = (images, angles)
        # return tf.map_fn(lambda x: tf.image.rot90(x[0], x[1]), elems, fn_output_signature=tf.float32, parallel_iterations=batch_size)
        return tf.vectorized_map(lambda x: tf.image.rot90(x[0], x[1]), elems)


class Translate_Int(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(Translate_Int, self).__init__(*args, **kwargs)

    def call(self, images):
        batch_size = tf.shape(images)[0]
        tx = tf.random.uniform(shape=(batch_size,), minval=-0.125, maxval=0.125)
        ty = tf.random.uniform(shape=(batch_size,), minval=-0.125, maxval=0.125)
        elems = (images, tx, ty)

        def translate_img(images, tx, ty):
            m_off = round(0.125 * 48)
            padded = tf.pad(images, tf.constant([[m_off, m_off], [m_off, m_off], [0, 0]]), mode='reflect')
            rolled = tf.roll(padded, shift=[tx, ty], axis=[0, 1])
            cropped = tf.image.crop_to_bounding_box(rolled, m_off, m_off, 48, 48)
            return cropped

        return tf.vectorized_map(lambda x: translate_img(x[0],
                                                         tf.cast(tf.math.rint(x[1] * 48), tf.int32),
                                                         tf.cast(tf.math.rint(x[2] * 48), tf.int32)),
                                 elems)


class Compute_pRot(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(Compute_pRot, self).__init__(*args, **kwargs)

    # @tf.function
    def call(self, input):
        prot = tf.math.subtract(tf.ones_like(input),
                                tf.math.pow(tf.math.subtract(tf.ones_like(input), input), 0.5))
        return prot


class Random_Brightness(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(Random_Brightness, self).__init__(*args, **kwargs)

    def call(self, images):
        batch_size = tf.shape(images)[0]

        value = tf.random.uniform(shape=(batch_size,), minval=0, maxval=0.2)

        elems = (images, value)
        return tf.vectorized_map(lambda x: tf.image.adjust_brightness(x[0], x[1]),
                                 elems)


class Magnitude_Rnd(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(Magnitude_Rnd, self).__init__(*args, **kwargs)

    def call(self, images):
        batch_size = tf.shape(images)[0]
        strength = tf.random.uniform(shape=(batch_size,), minval=0.75, maxval=1)
        strength = tf.expand_dims(strength, axis=-1)
        strength = tf.expand_dims(strength, axis=-1)

        matrix = tf.tile(strength, [1, tf.shape(images)[1], tf.shape(images)[2]])
        matrix = tf.expand_dims(matrix, axis=-1)
        return tf.math.multiply(images, matrix)


class Facies_Inv(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(Facies_Inv, self).__init__(*args, **kwargs)

    def call(self, images):
        return images * -1


class Additive_noise(keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(Additive_noise, self).__init__(*args, **kwargs)

    def call(self, images):
        batch_size = tf.shape(images)[0]
        # n_std = tf.random.normal(shape=(batch_size,), minval=0.75, maxval=1)  # define std
        n_std = tf.abs(tf.random.normal([batch_size, ], 0, 0.1))
        n_std = tf.reshape(n_std, [-1, 1, 1, 1])
        images += tf.random.normal([batch_size, 48, 48, 1]) * n_std
        return images


# Adaptive Discriminator Augmentation (ADA)
class AdaptiveDiscriminatorAugmentation(object):
    def __init__(self, img_shape, method='gradient', target=0.0, max_translation=0.125, max_rotation=0.125,
                 max_zoom=0.25):
        if method == 'fixed':
            self.probability = tf.Variable(target)
        else:
            self.probability = tf.Variable(0.0, trainable=True)
        self.probability_tracker = []
        self.accuracy_tracker = []
        self.method = method
        self.target = target
        self.max_translation = max_translation
        self.max_rotation = max_rotation
        self.max_zoom = max_zoom

        self.augment = self.build_augmenter(img_shape)

    def build_augmenter(self, img_shape, blitting=True, geometric=True, color=False, noise=False):
        input_layer = layers.Input(shape=img_shape, name='image_input')
        input_prob = layers.Input(shape=(1,), name='probability')

        output = layers.Lambda((lambda x: x))(input_layer)

        if blitting:
            pipeline = Flip()(input_layer, training=True)
            output = Gate_Random()(input_layer, pipeline, input_prob)

            pipeline = Rotate_Int()(output, training=True)
            output = Gate_Random()(output, pipeline, input_prob)

            pipeline = Translate_Int()(output, training=True)
            output = Gate_Random()(output, pipeline, input_prob)

        if geometric:
            pRot = Compute_pRot()(input_prob)

            pipeline = layers.RandomZoom((-0.2, 0.2), (-0.2, 0.2), interpolation='nearest')(output, training=True)
            output = Gate_Random()(output, pipeline, input_prob)

            pipeline = layers.RandomRotation(0.25, interpolation='nearest')(output, training=True)
            output = Gate_Random()(output, pipeline, pRot)

            pipeline = layers.RandomZoom((-0.2 / 5, 0.2 / 5), (-0.2, 0.2), interpolation='nearest')(output,
                                                                                                    training=True)
            output = Gate_Random()(output, pipeline, input_prob)

            pipeline = layers.RandomRotation(0.25, interpolation='nearest')(output, training=True)
            output = Gate_Random()(output, pipeline, pRot)

            pipeline = layers.RandomTranslation(0.125, 0.125, interpolation='nearest', fill_mode='reflect')(output,
                                                                                                            training=True)
            output = Gate_Random()(output, pipeline, input_prob)

        if color:
            pipeline = Random_Brightness()(output, training=True)
            output = Gate_Random()(output, pipeline, input_prob)

            pipeline = Magnitude_Rnd()(output, training=True)
            # pipeline = tf.clip_by_value(pipeline, -1.0, 1.0)
            output = Gate_Random()(output, pipeline, input_prob)

            pipeline = Facies_Inv()(output, training=True)
            output = Gate_Random()(output, pipeline, input_prob)

        if noise:
            pipeline = Additive_noise()(output, training=True)
            # pipeline = tf.clip_by_value(pipeline, -1.0, 1.0)
            output = Gate_Random()(output, pipeline, input_prob)

        aug = Model(inputs=[input_layer, input_prob], outputs=[output])
        return aug

    def update(self, real_logits, integration_steps):
        # current_accuracy = tf.reduce_mean(0.5 * (1.0 + tf.sign(real_logits)))
        rt = tf.reduce_mean(tf.sign(real_logits))
        if self.method == 'gradient':
            pass
        if self.method == 'accuracy':
            proportional_error = (rt - self.target)
            derivative_error = (rt - self.accuracy_tracker[-1]) / 1

            accuracy_error = proportional_error / 2 + (derivative_error / 2)

            ratio = 1 / integration_steps
            self.probability.assign(
                tf.clip_by_value(self.probability + ratio * accuracy_error, 0.0, 1.0))
        if self.method == 'accuracy2':
            accuracy_error = tf.sign(rt - self.target)
            ratio = 1 / integration_steps
            self.probability.assign(
                tf.clip_by_value(self.probability + ratio * accuracy_error, 0.0, 1.0))
        if self.method == 'fixed':
            pass
        return rt
