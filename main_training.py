import numpy as np
from data_loader import ResLoader
import matplotlib.pyplot as plt

from GAN.networks import DcganR1Ada

name = 'mnist'
# load training dataset
if name == 'res':
    X_train = ResLoader.load_3_facies(folder='./datasets/3facies')
    shape = (48, 48, 1)
elif name == 'mnist':  # MNIST handwritten
    X_train, _, X_val, _ = ResLoader.load_mnist_data()
    shape = (28, 28, 1)


zdim = 64
noise = np.random.normal(0, 1, (5 * 5, zdim))

gan = DcganR1Ada(img_shape=shape, latent_dim=zdim, d_lr=0.0001, g_lr=0.0001,
                 metric='FRD', metric_interval=500,
                 d_reg=None, ada=None, folder='./tfcheckpoints/')  # assembly network with folder
gan.train_gan(X_train, epochs=100000, plot_interval=500, method='random', noise=noise, batch_size=32, start=0, save=True)  # train gan