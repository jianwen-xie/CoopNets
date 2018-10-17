import time
import numpy as np
import tensorflow as tf
from model.utils.mnist_util import maybe_download_minst

def get_valid(ds, limit_size=-1):
    if ds == 'mnist':
        train_images, train_labels = maybe_download_minst('./MNIST-data', train=True, one_hot=False)
        validation_images = train_images[50000:60000]
        validation_images = np.multiply(validation_images, 1.0 / 255)
        validation_images = np.reshape(validation_images, [-1, 784])
        return validation_images[:limit_size]
    else:
        raise ValueError("Unknow dataset: {}".format(ds))


def get_test(ds):

    if ds == 'mnist':
        test_data, test_labels = maybe_download_minst('./MNIST-data', train=False, one_hot=False)
        test_data = np.reshape(test_data, [-1, 784])
        return np.asarray(test_data, dtype=np.float32)
    else:
         raise ValueError("Unknow dataset: {}".format(ds))


def tf_log_mean_exp(x):
    max_ = tf.reduce_max(x, axis=1)
    return max_ + tf.log(tf.reduce_mean(tf.exp(x - tf.expand_dims(max_, axis=1)), axis=1))


def normalize_data(data, low=0.0, high=255.0, shape=(28, 28)):
    res = np.zeros(shape=(len(data), shape[0], shape[1]), dtype=np.float32)

    for i in range(len(data)):
        temp = np.maximum(low, np.minimum(high, data[i]))
        cmin = temp.min()
        cmax = temp.max()
        temp = (temp - cmin) / (cmax - cmin)
        temp = np.resize(temp, shape)
        res[i] = temp
    return res

def tf_parzen(x, mu, sigma):
    a = (tf.expand_dims(x, 1) - tf.expand_dims(mu, 0)) / sigma
    E = tf_log_mean_exp(-0.5 * tf.reduce_sum(tf.multiply(a, a), axis=2))
    Z = tf.cast(784, tf.float32) * tf.log(sigma * np.sqrt(np.pi * 2.0))
    return E - Z


class ParsenDensityEsimator():

    def __init__(self, sess, batch_size=100, sigma=None, limit_size=1000, dataset='mnist', sigma_start=-1, sigma_end=0, cross_val=10):

        self.batch_size = batch_size
        self.sigma = sigma
        self.limit_size = limit_size
        self.dataset = dataset
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.cross_val = cross_val

        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.samples = tf.placeholder(tf.float32, shape=[None, 784])
        self.sigma_placeholder = tf.placeholder(tf.float32)
        self.parzen = tf_parzen(self.x, self.samples, self.sigma_placeholder)

        self.sess = sess

    def get_nll(self, x, samples, sigma, batch_size=10):
        """
        Credit: Yann N. Dauphin
        """

        inds = range(x.shape[0])
        n_batches = int(np.ceil(float(len(inds)) / batch_size))

        times = []
        nlls = []
        for i in range(n_batches):
            begin = time.time()
            nll = self.sess.run(self.parzen, feed_dict={self.x: x[inds[i::n_batches]],
                                                        self.samples: samples, self.sigma_placeholder: sigma})
            # nll = parzen(x[inds[i::n_batches]])
            end = time.time()
            times.append(end-begin)
            nlls.extend(nll)

            #if i % 10 == 0:
            #    print(i, np.mean(times), np.mean(nlls))

        return np.array(nlls)

    def cross_validate_sigma(self, samples, data, sigmas, batch_size):

        lls = []
        for sigma in sigmas:
            #print(sigma)
            tmp = self.get_nll(data, samples, sigma, batch_size=batch_size)
            lls.append(np.asarray(tmp).mean())

        ind = np.argmax(lls)
        return sigmas[ind]

    def fit(self, samples, test):
        # cross validate sigma
        if self.sigma is None:
            valid = get_valid(self.dataset, limit_size=self.limit_size)
            sigma_range = np.logspace(self.sigma_start, self.sigma_end, num=self.cross_val)
            sigma = self.cross_validate_sigma(samples, valid, sigma_range, self.batch_size)
        else:
            sigma = float(self.sigma)

        print("Using Sigma: {}".format(sigma))

        # fit and evaulate
        ll = self.get_nll(test, samples, sigma, batch_size=self.batch_size)
        se = ll.std() / np.sqrt(test.shape[0])

        print("Log-Likelihood of test set = {}, se: {}".format(ll.mean(), se))

        return ll.mean(), se

    def eval_parzen(self, samples_des, samples_gen):

        samples_des = np.maximum(-1, np.minimum(1, samples_des))
        samples_des = (samples_des - samples_des.min()) / (samples_des.max() - samples_des.min())
        samples_des = np.asarray(np.reshape(samples_des, [-1, samples_des.shape[1] * samples_des.shape[2]]),
                                 dtype=np.float32)

        samples_gen = np.maximum(-1, np.minimum(1, samples_gen))
        samples_gen = (samples_gen - samples_gen.min()) / (samples_gen.max() - samples_gen.min())
        samples_gen = np.asarray(np.reshape(samples_gen, [-1, samples_gen.shape[1] * samples_gen.shape[2]]),
                                 dtype=np.float32)
        test = get_test(self.dataset)
        # test = get_test(parzen_args['dataset'])
        test = np.multiply(test, 1.0 / 255)

        parzon_des_mean, parzon_des_se = self.fit(samples_des, test)
        parzon_gen_mean, parzon_gen_se = self.fit(samples_gen, test)

        return parzon_des_mean, parzon_des_se, parzon_gen_mean, parzon_gen_se


