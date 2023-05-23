import numpy as np
from data_loader import ResLoader
import matplotlib.pyplot as plt

from GAN.networks import DcganR1Ada
from gradient_projector import GradientProjector
from ensemble_projector import EnsembleProjector

import tensorflow as tf

name = 'mnist'
# load training dataset
if name == 'res':
    X_train = ResLoader.load_3_facies(folder='./datasets/3facies')
    shape = (48, 48, 1)
elif name == 'mnist':  # MNIST handwritten
    X_train, _, X_val, _ = ResLoader.load_mnist_data()
    shape = (28, 28, 1)

gan = DcganR1Ada(img_shape=shape, latent_dim=64, d_lr=0.0001, g_lr=0.0001,
                 metric='FRD', metric_interval=500,
                 d_reg=None, ada=None, folder='./tfcheckpoints/')  # assembly network with folder
gan.ema_generator.load_weights('./tfcheckpoints/best_G')

# latents = np.sort(np.random.normal(0, 1, (1, 64))) * -1
# X_true = gan.ema_generator.predict(latents)
# np.save('./projection_results/z_true.npy', latents)
# np.save('./projection_results/X_true.npy', X_true)
#
#
# lr_values = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
# for exp in range(1):
#     for i in range(lr_values.__len__()):
#         projector = GradientProjector(gan.ema_generator, shape, 64, learning_rate=lr_values[i])
#         projected, obj = projector.project(X_true, 20000)
#         np.save('./projection_results/gradient/z_grad_lr{}_exp{}.npy'.format(lr_values[i], exp), projected)
#         np.save('./projection_results/gradient/obj_grad_{}_exp{}.npy'.format(lr_values[i], exp), np.array(obj))


# def minimum_grad():
#     lr_values = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
#     min_obj = np.zeros((3, lr_values.__len__()))
#     for exp in range(3):
#         for i in range(lr_values.__len__()):
#             min_obj[exp, i] = np.load('./projection_results/gradient/obj_grad_{}_exp{}.npy'.format(lr_values[i], exp)).min()
#     plt.figure()
#     plt.fill_between(lr_values, min_obj.min(0), min_obj.max(0), color='xkcd:green', alpha=0.2)
#     plt.plot(lr_values, np.median(min_obj, 0), color='xkcd:green')
#     plt.yscale('log')
#     plt.xscale('log')
#
#
# fig, axs = plt.subplots(1, 3)
# axs[0].imshow(X_true[0]), axs[0].axis('off')
# axs[1].imshow(gan.ema_generator.predict(projected)[0]), axs[1].axis('off')
# axs[2].plot(obj), plt.yscale('log')
# plt.tight_layout()

it_values = np.array([1, 2, 4, 8, 16, 32])
X_true = np.load('./projection_results/X_true.npy')
for exp in range(3):
    for i in range(it_values.__len__()):
        projector2 = EnsembleProjector(gan.ema_generator, shape, 64, 10000, upscaling_rate=1, loss_net_folder=None)
        en_projected, en_obj = projector2.project(X_true, it_values[i], 0.00005)
        # last forward here
        X_final = gan.ema_generator(en_projected.T) .numpy()
        loss_f = tf.reduce_sum(tf.square(X_final - X_true), axis=(1, 2, 3))
        en_obj.append(loss_f)
        np.save('./projection_results/ensemble_large/obj_ens_{}_exp{}.npy'.format(it_values[i], exp), np.array(en_obj))
        np.save('./projection_results/ensemble_large/z_ens_{}_exp{}.npy'.format(it_values[i], exp), np.array(en_projected))

# fig, axs = plt.subplots(1, 3)
# axs[0].imshow(X_true[0]), axs[0].axis('off')
# axs[1].imshow(gan.ema_generator.predict(en_projected.T[0:1])[0]), axs[1].axis('off')
# axs[2].plot(en_obj, 'b'), plt.yscale('log')
# plt.tight_layout()
#
# plt.figure()
# plt.plot(en_projected.mean(-1), 'xkcd:blue', label='ensemble')
# plt.plot(projected.flatten(), 'xkcd:green', label='gradient')
# plt.plot(latents.flatten(), 'r', label='true')
# plt.legend()
# plt.tight_layout()
#
# #stats
# X_pred = gan.ema_generator.predict(en_projected.T)
#
# fig, axs = plt.subplots(1, 5)
# axs[0].imshow(X_true[0]), axs[0].axis('off'), axs[0].set_title('true')
# axs[1].imshow(gan.ema_generator.predict(en_projected.T[0:1])[0]), axs[1].axis('off'), axs[1].set_title('single')
# axs[2].imshow(gan.ema_generator.predict(np.expand_dims(en_projected.mean(-1), axis=0))[0]), axs[2].axis('off'), axs[2].set_title('z_mean')
# axs[3].imshow(X_pred.mean(0)), axs[3].axis('off'), axs[3].set_title('X_mean')
# axs[4].imshow(X_pred.std(0)), axs[4].axis('off'), axs[4].set_title('X_std')
#
# fig, axs = plt.subplots(1, 2)
# axs[0].plot(obj, 'r')
# obj_p = np.percentile(np.array(en_obj), np.arange(0, 100.1, 10), axis=1)
# # axs[1].fill_between(np.arange(0, 10, 1), obj_p[0], obj_p[-1], color='#1b9e77', alpha=0.1)
# axs[1].fill_between(np.arange(0, 20, 1), obj_p[1], obj_p[-2], color='#1b9e77', alpha=0.1)
# axs[1].fill_between(np.arange(0, 20, 1), obj_p[2], obj_p[-3], color='#1b9e77', alpha=0.1)
# axs[1].fill_between(np.arange(0, 20, 1), obj_p[3], obj_p[-4], color='#1b9e77', alpha=0.1)
# axs[1].fill_between(np.arange(0, 20, 1), obj_p[4], obj_p[-5], color='#1b9e77', alpha=0.1)
# axs[1].plot(np.arange(0, 20, 1), np.median(np.array(en_obj),-1), color='#1b9e77')
# for i in range(2):
#     axs[i].set_yscale('log')
#     axs[i].set_ylim(0.002, 1000)



X_true = X_val[1:2]

import time
st = time.time()
projector = GradientProjector(gan.ema_generator, shape, 64, learning_rate=0.02)
projected, obj = projector.project(X_true, 20000)
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

import time
st = time.time()
projector2 = EnsembleProjector(gan.ema_generator, shape, 64, 2000, upscaling_rate=1, loss_net_folder=None)
en_projected, en_obj = projector2.project(X_true, 10, 0.005)
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')


X_pred = gan.ema_generator.predict(en_projected.T)
fig, axs = plt.subplots(1, 5)
axs[0].imshow(X_true[0], cmap='gray'), axs[0].axis('off'), axs[0].set_title('true')
axs[1].imshow(gan.ema_generator.predict(en_projected.T[0:1])[0], cmap='gray'), axs[1].axis('off'), axs[1].set_title('single')
axs[2].imshow(gan.ema_generator.predict(np.expand_dims(en_projected.mean(-1), axis=0))[0], cmap='gray'), axs[2].axis('off'), axs[2].set_title('z_mean')
axs[3].imshow(X_pred.mean(0), cmap='gray'), axs[3].axis('off'), axs[3].set_title('X_mean')
axs[4].imshow(X_pred.std(0), cmap='gray'), axs[4].axis('off'), axs[4].set_title('X_std')

en_p = np.percentile(en_projected, np.arange(0, 100.1, 5), axis=1)
xvalues = np.arange(0, 64, 1)
for _ in range(10):
    plt.fill_between(xvalues, en_p[0 + _], en_p[-1 - _], color='b', alpha=0.1)
plt.plot(xvalues, np.median(en_p, 0), color='b')

en_p2 = np.percentile(np.array(en_obj), np.arange(0, 100.1, 5), axis=1)
plt.figure()
xvalues = np.arange(0, 32, 1)
for _ in range(10):
    plt.fill_between(xvalues, en_p2[0 + _], en_p2[-1 - _], color='b', alpha=0.1)
plt.plot(xvalues, np.median(en_p2, 0), color='b')
plt.axhline(np.array(obj).min())
plt.yscale('log')