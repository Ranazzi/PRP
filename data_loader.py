import numpy as np


def createarray(strinput: str, dim=None):
    """ read .inc file with V*t pattern (number V repeated t times), then convert to array
    example:
    :param strinput: array with pattern
    :param dim: dimension to avoid .append
    :return: array with values
    """
    text_file = strinput
    str_array = re.split('; |\s|\n', text_file)
    if dim is None:
        array = []
        for idc in str_array:
            if "*" in idc:
                a1 = idc.split("*")
                a2 = list(map(float, a1))
                a3 = np.ones(int(a2[0])) * a2[1]
                array = np.append(array, a3)  # .append is not efficient
            else:
                try:
                    array = np.append(array, float(idc))
                except ValueError:
                    continue
    else:
        array = np.zeros(dim)
        counter = 0
        for idc in str_array:
            if "*" in idc:
                a1 = idc.split("*")
                a2 = list(map(float, a1))
                a3 = np.ones(int(a2[0])) * a2[1]
                array[counter:(counter + a3.__len__())] = a3
                counter += a3.__len__()
            else:
                try:
                    array[counter] = float(idc)
                    counter += 1
                except ValueError:
                    continue
    return array


class ResLoader(object):
    @staticmethod
    def load_channel2d_large(preprocess=True):
        if preprocess is True:
            rows = np.load('./cases/channel2d_large/dataset.npy')
            rows = rows.astype('float')
        else:
            import csv
            rows = []
            for i in range(8):
                with open('cases/channel2d_large/channel_10000_rotation_48_{}.txt'.format(i + 1)) as file:
                    csvreader = csv.reader(file)
                    header = []
                    header = next(csvreader)
                    for row in csvreader:
                        rows.append(row)
            rows = np.array(rows)
            rows[rows == 'category_0'] = -1
            rows[rows == 'category_1'] = 1
            rows = rows.astype('int')
            rows = rows.reshape((48, 48, 1, 80000), order='F')
            rows = np.einsum('xyzn->nxyz', rows)
        return rows

    @staticmethod
    def load_3_facies(folder):
        rows = np.load('{}/dataset.npy'.format(folder))
        rows = rows.astype('float')
        return rows

    @staticmethod
    def load_unisim_ii(Ne, preprocess=True, conditional=False, null=False, parameter='kx', scale=None, crop=False):
        if preprocess:
            En_kx = np.load(("cases/UNISIM-II/preprocessed/{}.npy".format(parameter)))
        else:
            if conditional:
                En_kx = np.zeros((Ne * 30, 46, 69, 1))
            else:
                En_kx = np.zeros((Ne, 46, 69, 30))
            # En_rtype = np.zeros((Ne, 46, 69, 30))
            for n in range(Ne):
                raw = open("cases/UNISIM-II/{}{}.inc".format(parameter, n + 1), 'r')
                a = raw.read()
                c = createarray(a)
                if conditional:
                    En_kx[(n * 30):((n + 1) * (30)), :, :, 0] = np.einsum('xyn->nxy', c.reshape((46, 69, 30), order='F'))
                else:
                    En_kx[n, :, :, :] = c.reshape((46, 69, 30), order='F')

        if null:
            raw = open("cases/UNISIM-II/null_mtrx.inc", 'r')
            a = raw.read()
            c = createarray(a)
            d = c.reshape((46, 69, 30), order='F')
            # d = d[:, :, :]

            En_kx[:, d[:, :, 0] == 0] = np.nan

        if scale == 'log':
            En_kx[En_kx < 1] = 1
            En_kx = np.log(En_kx)

        if crop:
            import tensorflow as tf
            En_kx = tf.image.crop_to_bounding_box(En_kx, 5, 25, 32, 32)
            En_kx = tf.image.resize(En_kx, [48, 48]).numpy()

        b_down = np.nanmin(En_kx)
        b_up = np.nanmax(En_kx)
        bounds = (b_down, b_up)
        # En_kx = 2 * (En_kx - np.nanmin(En_kx)) / (np.nanmax(En_kx) - np.nanmin(En_kx)) - 1
        return En_kx


    @staticmethod
    def load_gaussian(size):
        from FyeldGenerator import generate_field
        # Draw samples from a normal distribution

        def Pkgen(n):
            def Pk(k):
                return np.power(k, -n)

            return Pk

        def distrib(shape):
            a = np.random.normal(loc=0, scale=1, size=shape)
            b = np.random.normal(loc=0, scale=1, size=shape)
            return a + 1j * b

        shape = (48, 48)

        X_train = np.zeros((size, 48, 48, 1))
        for i in range(size):
            X_train[i, :, :, 0] = generate_field(distrib, Pkgen(3), shape)
        X_train = 2 * (X_train - np.nanmin(X_train)) / (np.nanmax(X_train) - np.nanmin(X_train)) - 1
        return X_train

    @staticmethod
    def load_channel3d(preprocess=True):
        if preprocess is True:
            rows = np.load('./cases/channel3d/dataset.npy').astype('float')
        return rows


    @staticmethod
    def load_classifier_data(N_classes, N_perclass, train_frac=0.9, shuffle=True):
        X_train_raw = np.zeros((N_classes * N_perclass, 48, 48, 1))
        L_train_raw = np.zeros((N_classes * N_perclass, 1))
        for i in range(N_classes):
        # for i in range(30,  60):
            X_train_raw[i * N_perclass: (i + 1) * N_perclass, :, :, 0] = np.load(
                './cases/classifier/X_{:0>2}.npy'.format(i))[0:N_perclass]
            L_train_raw[i * N_perclass: (i + 1) * N_perclass, 0] = i
        X_train_raw[np.argwhere(np.isnan(X_train_raw))] = 0
        if shuffle:
            p = np.random.permutation(len(X_train_raw))
            X_train_raw, L_train_raw = X_train_raw[p], L_train_raw[p]

        limit_f = np.round(X_train_raw.shape[0] * train_frac).astype('int')
        X_train, X_val = X_train_raw[:limit_f, ...], X_train_raw[limit_f:, ...]
        L_train, L_val = L_train_raw[:limit_f, ...], L_train_raw[limit_f:, ...]

        return X_train, L_train, X_val, L_val

    @staticmethod
    def load_mnist_data():
        import tensorflow as tf
        (train_images_tmp, train_classes), (val_images_tmp, val_classes) = tf.keras.datasets.mnist.load_data()
        train_images_tmp = train_images_tmp.reshape(train_images_tmp.shape[0], 28, 28, 1).astype('float32')
        X_train = (train_images_tmp - 127.5) / 127.5  # image normalization to [-1, 1]

        val_images_tmp = val_images_tmp.reshape(val_images_tmp.shape[0], 28, 28, 1).astype('float32')
        X_val = (val_images_tmp - 127.5) / 127.5  # image normalization to [-1, 1]

        return X_train, train_classes, X_val, val_classes
