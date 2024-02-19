import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import time

import pickle
from GAN.netblocks import NetBlocks
from GAN.metrics import GanMetrics
from GAN.augmentation import AdaptiveDiscriminatorAugmentation

class GanNetworks(object):
    def __init__(self, img_shape, latent_dim, g_lr, d_lr, metric, metric_interval, d_reg, ada, folder):
        self.folder = folder

        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.n_channels = img_shape[-1]

        self.g_lr = g_lr  # generator learning rate
        self.d_lr = d_lr  # discriminator learning rate

        self.d_reg = d_reg

        # Prior assembly attributes to None following:
        # https://stackoverflow.com/questions/19284857/instance-attribute-attribute-name-defined-outside-init
        self.generator = None  # assembly generator
        self.ema_generator = None

        self.discriminator = None  # assembly discriminator
        self.ema_discriminator = None

        if ada is not None:
            self.ada = ada  # assembly adaptive augmentation

        self.gen_opt = None  # assembly generator optimizer
        self.dis_opt = None  # assembly discriminator optimizer

        self.losses = {'D': [], 'Dreal': [], 'Dfake': [], 'G': []}

        if metric is not None:
            self.metrics = dict()
            self.metric_interval = metric_interval
            if 'FID' in metric:
                self.metrics['FID'] = []
            if 'FRD' in metric:
                self.metrics['FRD'] = []
            if 'D_out' in metric:
                self.metrics['D_out'] = dict()
                self.metrics['D_out']['real'] = []
                self.metrics['D_out']['generated'] = []

    def save_checkpoint(self, model):  # function to save weights for a given iteration
        if model == 'D':
            self.discriminator.save_weights('{}/{}'.format(self.folder, model))
        elif model == 'D_ema':
            self.ema_discriminator.save_weights('{}/{}'.format(self.folder, model))
        elif model == 'G':
            self.generator.save_weights('{}/{}'.format(self.folder, model))
        elif model == 'G_ema':
            self.ema_generator.save_weights('{}/{}'.format(self.folder, model))
        elif model == 'best':
            self.ema_generator.save_weights('{}/{}_G'.format(self.folder, model))
            self.ema_discriminator.save_weights('{}/{}_D'.format(self.folder, model))

    def load_checkpoint(self, model):
        if model == 'D':
            self.discriminator.load_weights('{}/{}'.format(self.folder, model))
        elif model == 'D_ema':
            self.ema_discriminator.load_weights('{}/{}'.format(self.folder, model))
        elif model == 'G':
            self.generator.load_weights('{}/{}'.format(self.folder, model))
        elif model == 'G_ema':
            self.ema_generator.load_weights('{}/{}'.format(self.folder, model))

    def display_batch(self, r=5, c=5, noise=None, save=None):
        plt.close('all')
        plt.ion()
        if noise is None:
            noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.ema_generator.predict(noise)
        if gen_imgs.shape[-1] == 3:
            gen_imgs = (gen_imgs * 127.5 + 127.5).astype('int')
            limits = (0, 255)
        elif gen_imgs.shape[-1] == 2:  # case with two channels output
            gen_imgs = np.argmin(gen_imgs, axis=-1)
            limits = (-1, 1)
        else:
            limits = (-1, 1)
        fig, axs = plt.subplots(r, c, figsize=(5, 5))
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], aspect='equal', cmap='viridis', interpolation='none',
                                 clim=limits)
                axs[i, j].axis('off')
                cnt += 1
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.1)
        if save is not None:
            plt.savefig('{}iteration_{}.png'.format(self.folder, save))
        return gen_imgs

    def display_losses(self, moving=100):
        fig, axs = plt.subplots(2, 2, figsize=(7, 7))
        axs[0, 0].plot(self.losses['D'], 'b', label='D', alpha=0.5)
        axs[0, 0].plot(self.losses['G'], 'r', label='G', alpha=0.5)
        axs[0, 0].legend(), axs[0, 0].set_xlabel('iterations')
        axs[0, 1].plot(np.convolve(self.losses['D'], np.ones(moving), mode='valid') / moving, 'b', label='D')
        axs[0, 1].plot(np.convolve(self.losses['G'], np.ones(moving), mode='valid') / moving, 'r', label='G')
        axs[0, 1].legend(), axs[0, 1].set_xlabel('iterations')
        axs[1, 0].plot(self.losses['Dreal'], 'g', label='Real', alpha=0.5)
        axs[1, 0].plot(self.losses['Dfake'], 'c', label='Fake', alpha=0.5)
        axs[1, 0].legend(), axs[1, 0].set_xlabel('iterations')
        axs[1, 1].plot(np.convolve(self.losses['Dreal'], np.ones(moving), mode='valid') / moving, 'g', label='Real')
        axs[1, 1].plot(np.convolve(self.losses['Dfake'], np.ones(moving), mode='valid') / moving, 'c', label='Fake')
        axs[1, 1].legend(), axs[1, 1].set_xlabel('iterations')
        plt.tight_layout()

    def interpolate(self, n=5, z1=None, z2=None, cmap='jet'):
        # function to interpolate imagens between two samples
        if z1 is None:
            z1 = np.random.normal(0, 1, (1, self.latent_dim))
        im1 = self.generator.predict(z1)
        if z2 is None:
            z2 = np.random.normal(0, 1, (1, self.latent_dim))
        im2 = self.generator.predict(z2)

        var = (z1 - z2) / (n + 1) * -1  # compute grads

        fig, axs = plt.subplots(1, n + 2)
        axs[0].imshow(im1[0], aspect='equal', cmap=cmap, interpolation='none', clim=(-1, 1))
        axs[0].axis('off')
        axs[-1].imshow(im2[0], aspect='equal', cmap=cmap, interpolation='none', clim=(-1, 1))
        axs[-1].axis('off')

        for i in range(1, n + 1):
            interp = z1 + var * i
            interp_image = self.generator.predict(interp)
            axs[i].imshow(interp_image[0], aspect='equal', cmap=cmap, interpolation='none', clim=(-1, 1))
            axs[i].axis('off')

    def compute_metrics(self, X_test, save=None):
        print('computing validation metrics')
        size = X_test.shape[0]
        print('generating validation set')
        X_pred = self.ema_generator.predict(np.random.normal(0, 1, (size, self.latent_dim)), verbose=1)
        if 'FID' in self.metrics:
            imsizex = np.minimum(2 * self.img_shape[0], 75)
            imsizey = np.minimum(2 * self.img_shape[1], 75)
            imsize = (imsizex, imsizey)
            act = GanMetrics.frechet_inception_distance(X_test, X_pred, shape=imsize, batch_size=50)
            print("FID: %f" % act)
            self.metrics['FID'].append(act)
            if save:
                np.save('{}/FID.npy'.format(self.folder), self.metrics['FID'])
        if 'FRD' in self.metrics:
            act = GanMetrics.frechet_reservoir_distance(X_test, X_pred)
            print("FRD: %f" % act)
            self.metrics['FRD'].append(act)
            if save:
                np.save('{}/FRD.npy'.format(self.folder), self.metrics['FRD'])

        if 'D_out' in self.metrics:
            real, fake = GanMetrics.discriminator_outputs(self, X_test[0:1000])
            self.metrics['D_out']['real'].append(real)
            self.metrics['D_out']['generated'].append(fake)
            if save:
                with open('{}/d_out'.format(save), 'wb') as fp:
                    pickle.dump(self.metrics['D_out'], fp)


class DcganR1Ada(GanNetworks):
    def __init__(self, img_shape=(51, 51, 2), latent_dim=500,
                 g_lr=0.001, d_lr=0.001,
                 metric=None, metric_interval=150, d_reg=0, ada=None, folder='./tfcheckpoints/'):
        super().__init__(img_shape, latent_dim, g_lr, d_lr, metric, metric_interval, d_reg, ada, folder)

        self.build_model(ada=ada)  # build network

    def build_model(self, ada):
        self.generator = NetBlocks.build_baseline_generator(latent_dim=self.latent_dim,
                                                            img_shape=self.img_shape, batch_norm=0.9,
                                                            nfilters=32)  # gen object
        self.ema_generator = tf.keras.models.clone_model(self.generator)
        self.discriminator = NetBlocks.build_baseline_discriminator(img_shape=self.img_shape, batch_norm=None,
                                                                    layer_norm=False,
                                                                    nfilters=32)  # build dis object
        self.ema_discriminator = tf.keras.models.clone_model(self.discriminator)
        self.gen_opt = tf.keras.optimizers.Adam(learning_rate=self.g_lr, beta_1=0.5, beta_2=0.9)  # gen optimizer
        self.dis_opt = tf.keras.optimizers.Adam(learning_rate=self.d_lr, beta_1=0.5, beta_2=0.9)  # dis optimizer

        if ada is not None:
            self.ada = AdaptiveDiscriminatorAugmentation(img_shape=self.img_shape, method='accuracy', target=ada)

    def train_gan(self, X_train, epochs=5000, method='random', batch_size=32, plot_interval=None, noise=None,
                  start=0, save=False):
        @tf.function
        def train_step(batch, b_size, reg=None, ada=None):
            noise = tf.random.normal([b_size, self.latent_dim])
            with tf.GradientTape() as dis_tape:  # discriminator step
                generated = self.generator(noise, training=True)  # gen batch of fake samples

                if ada is True:
                    batch = self.ada.augment([batch, self.ada.probability], training=True)
                    generated = self.ada.augment([generated, self.ada.probability], training=True)

                r_logits = self.discriminator(batch, training=True)
                f_logits = self.discriminator(generated, training=True)  # dis output for fake

                dis_loss_f = tf.reduce_mean(tf.nn.softplus(f_logits))
                dis_loss_r = tf.reduce_mean(tf.nn.softplus(-r_logits))
                dis_loss_total = (dis_loss_r + dis_loss_f)  # compute discriminator loss

                if reg is not None:
                    r1_grads = tf.gradients(tf.reduce_sum(r_logits), [batch])[0]
                    r1_penalty = tf.reduce_sum(tf.square(r1_grads), axis=[1, 2, 3])
                    gp = tf.reduce_mean(r1_penalty) * (reg * 0.5)
                    dis_loss_total += tf.cast(gp, dtype='float32')

            dis_grad = dis_tape.gradient(target=dis_loss_total, sources=self.discriminator.trainable_variables)
            self.dis_opt.apply_gradients(zip(dis_grad, self.discriminator.trainable_variables))

            with tf.GradientTape() as gen_tape:  # generator step
                generated = self.generator(noise, training=True)  # gen batch of fake samples

                if ada is True:
                    generated = self.ada.augment([generated, self.ada.probability], training=True)

                f_logits = self.discriminator(generated, training=True)  # dis output for fake
                gen_loss = tf.reduce_mean(tf.nn.softplus(-f_logits))
            gen_grad = gen_tape.gradient(target=gen_loss, sources=self.generator.trainable_variables)
            self.gen_opt.apply_gradients(zip(gen_grad, self.generator.trainable_variables))

            # accuracy = self.ada.update(r_logits, integration_steps=1000)

            # return gen_loss, dis_loss_total, dis_loss_r, dis_loss_f, accuracy
            return gen_loss, dis_loss_total, dis_loss_r, dis_loss_f

        if method == 'all':
            pass

        elif method == 'random':
            # this method picks a random batch every iteration
            best = 99999
            augment = True if hasattr(self, 'ada') else False

            for epoch in range(start, epochs):
                if epoch == 0 and hasattr(self, 'ada'):
                    self.ada.accuracy_tracker.append(0.0)
                # if X_train == 'anime':
                #     from matplotlib.image import imread
                #     idx = np.random.choice(36739, batch_size, replace=False)
                #     _d = list()
                #     for __ in range(idx.__len__()):
                #         _d.append(imread('./datasets/anime/images/{}.jpg'.format(idx[__])))
                #     train_batch = np.array(_d)
                #     train_batch = (train_batch - 127.5) / 127.5
                # else:
                idx = np.random.choice(X_train.shape[0], batch_size, replace=False)
                train_batch = X_train[idx]
                gen_loss, dis_loss, d_loss_r, d_loss_f = train_step(train_batch, batch_size,
                                                                    reg=self.d_reg, ada=augment)  # train with batch

                if augment:
                    if epoch % 5 == 0:
                        idx = np.random.choice(X_train.shape[0], 200, replace=False)
                        train_batch = X_train[idx]
                        r_logits = self.discriminator(train_batch, training=False)
                        accuracy = self.ada.update(r_logits, integration_steps=500)

                        self.ada.probability_tracker.append(self.ada.probability.numpy())
                        self.ada.accuracy_tracker.append(accuracy)

                self.losses['D'].append(dis_loss)
                self.losses['Dreal'].append(d_loss_r)
                self.losses['Dfake'].append(d_loss_f)
                self.losses['G'].append(gen_loss)

                for weight, ema_weight in zip(self.generator.weights, self.ema_generator.weights):
                    ema_weight.assign(0.99 * ema_weight + (1 - 0.99) * weight)

                if epoch < 25000:
                    rampup_length = 0.99 * (epoch + 1) / 25000
                else:
                    rampup_length = 0.99

                for weight, ema_weight in zip(
                        self.discriminator.weights, self.ema_discriminator.weights
                ):
                    ema_weight.assign(rampup_length * ema_weight + (1 - rampup_length) * weight)

                # summary
                output = "it: %d, [D loss: %f] [G loss: %f]" % (epoch + 1, dis_loss, gen_loss)
                print(output)
                if hasattr(self, 'metrics'):
                    if (epoch % self.metric_interval == 0) and (epoch != 0):
                        self.compute_metrics(X_test=X_train[:10000], save=True)
                        if self.metrics['FRD'][-1] < best:
                            if save:
                                self.save_checkpoint('best')
                            best = self.metrics['FRD'][-1]
                        # if save:
                        #     np.save('{}/prob_tracker.npy'.format(self.folder),
                        #             np.array(self.ada.probability_tracker))
                        #     np.save('{}/acc_tracker.npy'.format(self.folder),
                        #             np.array(self.ada.accuracy_tracker))
                if plot_interval is not None:
                    if epoch % plot_interval == 0:
                        if save:  # save checkpoint
                            print('Saving discriminator and generator weights')
                            self.save_checkpoint('D')
                            self.save_checkpoint('G')
                            self.save_checkpoint('G_ema')
                            self.save_checkpoint('D_ema')
                            _ = self.ema_generator.predict(noise)
                            np.save('{}/realization_{}.npy'.format(self.folder, epoch), _)  # save a batch of realizations
                        self.display_batch(save=epoch, noise=noise)
