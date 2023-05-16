import numpy as np
from data_loader import ResLoader
import matplotlib.pyplot as plt

from GAN.networks import DcganR1Ada


# load training dataset
X_train = ResLoader.load_3_facies(folder='./datasets/3facies')
plt.imshow(X_train[0]), plt.axis('off')

noise = np.random.normal(0, 1, (5 * 5, 128))

shape = (48,48,1)
zdim = 128

gan = DcganR1Ada(img_shape=shape, latent_dim=zdim, d_lr=0.001, g_lr=0.001,
                 metric=None, metric_interval=5,
                 d_reg=5, ada=None, folder='./tfcheckpoints/')  # assembly network with folder
gan.train_gan(X_train, epochs=15000, plot_interval=5, method='random', noise=noise, batch_size=32, start=0)  # train gan