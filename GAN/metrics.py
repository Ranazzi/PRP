"""Metrics to validate GANs:
    FID: Frechet Inception Distance
    FRD: Frechet Reservoir Distance
"""
import numpy as np
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

from tensorflow.keras import layers, Model
from tensorflow.keras.initializers import RandomNormal, he_normal, GlorotNormal
from tensorflow.image import resize
from scipy.linalg import sqrtm


class GanMetrics(object):
    @staticmethod
    def frechet_reservoir_distance(x1, x2, batch_size=None):
        if batch_size is None:
            batch_size = x1.shape[0]
        print('loading network')
        Cls = ChannelClassifier((48, 48, 1), n_classes=6, n_filters=16, lr=0.001)  # validation each half epoch
        Cls.load_checkpoint()
        FRD = Model(inputs=Cls.classifier.input,
                    outputs=layers.GlobalAveragePooling2D()(Cls.classifier.layers[-5].output))  # first dense
        def get_probs(model, inps):
            act = model.predict(inps, verbose=0)
            return act

        print('geting activation values')
        act1 = get_probs(FRD, x1)
        act2 = get_probs(FRD, x2)
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

def build_classifier(img_shape, nfilters, nclasses):
    input_layer = layers.Input(shape=img_shape)  # input layer from the img shape

    c_1 = conv2d_bn(input_layer, 1 * nfilters, k_size=3)
    c_1 = conv2d_bn(c_1, 1 * nfilters, k_size=3, padding='valid')
    c_1 = conv2d_bn(c_1, 2 * nfilters, k_size=3)
    c_1 = layers.MaxPooling2D((3, 3), strides=(2, 2))(c_1)

    c_1 = conv2d_bn(c_1, 2.5 * nfilters, k_size=1, padding='valid')
    c_1 = conv2d_bn(c_1, 6 * nfilters, k_size=3, padding='valid')

    # mix 1: 28 x 28 x 8*nfilters
    b_1_1 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(c_1)
    b_1_1 = conv2d_bn(b_1_1, 1 * nfilters / 4, k_size=1)

    b_1_2 = conv2d_bn(c_1, 2 * nfilters / 4, k_size=1)

    b_1_3 = conv2d_bn(c_1, 1.5 * nfilters / 4, k_size=1)
    b_1_3 = conv2d_bn(b_1_3, 2 * nfilters / 4, k_size=5)

    b_1_4 = conv2d_bn(c_1, 2 * nfilters / 4, k_size=1)
    b_1_4 = conv2d_bn(b_1_4, 3 * nfilters / 4, k_size=3)
    b_1_4 = conv2d_bn(b_1_4, 3 * nfilters / 4, k_size=3)

    c_1 = layers.concatenate([b_1_1, b_1_2, b_1_3, b_1_4], axis=-1)

    # mix 3: 14 x 14 x 24*nfilters
    b_3_1 = layers.MaxPooling2D((3, 3), strides=(2, 2))(c_1)

    b_3_2 = conv2d_bn(c_1, 2 * nfilters / 4, k_size=1)
    b_3_2 = conv2d_bn(b_3_2, 3 * nfilters / 4, k_size=3)
    b_3_2 = conv2d_bn(b_3_2, 3 * nfilters / 4, k_size=3, strides=(2, 2), padding='valid')

    b_3_3 = conv2d_bn(c_1, 12 * nfilters / 4, 3, strides=(2, 2), padding='valid')

    c_1 = layers.concatenate([b_3_1, b_3_2, b_3_3], axis=-1)
    #
    # mix 4: 14 x 14 x 24*nfilters
    b_4_1 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(c_1)
    b_4_1 = conv2d_bn(b_4_1, 6 * nfilters / 4, k_size=1)

    b_4_2 = conv2d_bn(c_1, 6 * nfilters / 4, k_size=1)

    b_4_3 = conv2d_bn(c_1, 4 * nfilters / 4, k_size=1)
    b_4_3 = conv2d_bn(b_4_3, 6 * nfilters / 4, k_size=4)

    b_4_4 = conv2d_bn(c_1, 4 * nfilters / 4, k_size=1)
    b_4_4 = conv2d_bn(b_4_4, 6 * nfilters / 4, k_size=(1, 4))
    b_4_4 = conv2d_bn(b_4_4, 6 * nfilters / 4, k_size=(4, 1))

    c_1 = layers.concatenate([b_4_1, b_4_2, b_4_3, b_4_4], axis=-1)

    cf = layers.GlobalAveragePooling2D()(c_1)
    c = layers.Dropout(0.3)(cf)
    c = layers.Dense(8 * nfilters, activation='relu', kernel_initializer=GlorotNormal())(c)
    predictions = layers.Dense(nclasses, activation='softmax', kernel_initializer=GlorotNormal())(c)
    classifier = Model(inputs=[input_layer], outputs=[predictions], name='Classifier')
    return classifier


def sparse_categorical_accuracy(y_true, y_pred):
    m = tf.keras.metrics.SparseCategoricalAccuracy()
    m.update_state(y_true, y_pred)
    return m.result().numpy()


class ChannelClassifier(object):
    def __init__(self, img_shape, n_filters, n_classes, lr=0.001, save_interval=500, val_interval=50):
        self.img_shape = img_shape
        self.n_filters = n_filters
        self.n_classes = n_classes
        self.n_channels = img_shape[2]
        self.lr = lr
        self.save_interval = save_interval
        self.val_interval = val_interval
        if val_interval is not None:
            self.accuracy = {'train': [], 'val': []}
        self.cls_opt = tf.keras.optimizers.Adam(learning_rate=self.lr)  # cls optimizer
        self.classifier = build_classifier(img_shape=self.img_shape, nfilters=self.n_filters, nclasses=self.n_classes)
        self.losses = []

    def save_weights(self):  # function to save weights for a given iteration
        self.classifier.save_weights('./classifier_weights/weights')

    def load_checkpoint(self):  # function to load weights for a given iteration
        self.classifier.load_weights('gdrive/My Drive/Colab Notebooks/began/classifier_weights/weights')

    def compute_accuracy(self, X_val, L_val, type='val'):
        logits = self.classifier(X_val)
        if type == 'val':
            self.accuracy['val'].append(sparse_categorical_accuracy(L_val, logits))
        else:
            self.accuracy['train'].append(sparse_categorical_accuracy(L_val, logits))

    def display_metrics(self):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, figsize=(5, 5))
        axs[0].plot(np.array(self.losses), 'k', label='total')
        axs[0].title.set_text("Loss")
        axs[1].plot(np.arange(0, self.accuracy['train'].__len__() * self.val_interval, self.val_interval),
                    (np.array(self.accuracy['train']) + np.array(self.accuracy['train'])) / 2, 'r', label='train')
        axs[1].title.set_text("Accuracy")
        axs[1].plot(np.arange(0, self.accuracy['val'].__len__() * self.val_interval, self.val_interval),
                    (np.array(self.accuracy['val']) + np.array(self.accuracy['val'])) / 2, 'b', label='validation')
        axs[1].set_ylim([0.2, 1])
        axs[1].legend()
        plt.tight_layout()

    def train(self, X_real, L_real, epochs=10, batch_size=64, X_val=None, L_val=None):
        @tf.function
        def train_step(b_real, l_real):
            with tf.GradientTape() as tape:
                b_real = layers.GaussianNoise(0.01)(b_real, training=True)
                outputs = self.classifier(b_real, training=True)
                sce = tf.keras.losses.SparseCategoricalCrossentropy()
                loss = sce(l_real, outputs)
                clsgrad = tape.gradient(target=loss, sources=self.classifier.trainable_variables)
                self.cls_opt.apply_gradients(zip(clsgrad, self.classifier.trainable_variables))

            return loss

        for epoch in range(epochs):
            idx = np.random.choice(X_real.shape[0], batch_size, replace=False)
            r_batch = X_real[idx]  # pick a batch of real
            l_batch = L_real[idx]  # pick a batch of real
            # r_batch = ada.augment([r_batch, 0.4])
            loss = train_step(r_batch, l_batch)  # train with batch
            self.losses.append(loss)
            if (epoch % self.val_interval == 0) and X_val is not None:
                idx = np.random.choice(X_val.shape[0], 1500, replace=False)
                r_batch = X_real[idx]  # pick a batch of real
                l_batch = L_real[idx]  # pick a batch of fake
                self.compute_accuracy(r_batch, l_batch, 'train')
                # pick a validation batch
                idx = np.random.choice(X_val.shape[0], 1500, replace=False)
                r_batch = X_val[idx]  # pick a batch of real
                l_batch = L_val[idx]  # pick a batch of fake
                self.compute_accuracy(r_batch, l_batch, 'val')
            if epoch % self.save_interval == 0:
                # save checkpoint
                print('Saving weights')
                self.save_weights()
                # summary
                output = "[it: %d] [Loss: %f] [Acc: %f]" % (epoch + 1, self.losses[-1], self.accuracy['train'][-1])
                print(output)


def conv2d_bn(x,
              filters,
              k_size,
              padding='same',
              strides=(1, 1)):
    if type(k_size) is int:
        k1 = k2 = k_size
    else:
        k1, k2 = k_size[0], k_size[1]
    x = layers.Conv2D(
        filters, (k1, k2),
        strides=strides,
        padding=padding,
        use_bias=False)(
        x)
    x = layers.BatchNormalization(scale=False)(x)
    x = layers.Activation('relu')(x)
    return x
