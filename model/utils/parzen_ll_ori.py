import argparse
import time
import numpy as np
import tensorflow as tf
from scipy import io
from mnist_util import maybe_download_minst

def get_valid(ds, limit_size = -1):
    if ds == 'mnist':
        train_images, train_labels = maybe_download_minst('../data/mnist', train=True, one_hot=False)
        validation_images = train_images[50000:60000]
        validation_images = np.multiply(validation_images, 1.0 / 255)
        validation_images = np.reshape(validation_images, [-1, 784])
        return validation_images[:limit_size]
    else:
         raise ValueError("Unknow dataset: {}".format(ds))

def get_test(ds):
    if ds == 'mnist':
        test_data, test_labels = maybe_download_minst('../data/mnist', train=False, one_hot=False)
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
    Z = tf.cast(tf.shape(mu)[1], tf.float32) * tf.log(tf.constant(sigma * np.sqrt(np.pi * 2.0), dtype=tf.float32))
    return E - Z


class ParsenDensityEsimator():
    def __init__(self, sess, args):
        self.batch_size = args.batch_size

        self.sigma = args.sigma
        self.limit_size = args.limit_size
        self.dataset = args.dataset
        self.sigma_start = args.sigma_start
        self.sigma_end = args.sigma_end
        self.cross_val = args.cross_val
        self.x = tf.placeholder(tf.float32)

        self.sess = sess

    def get_nll(self, x, parzen, batch_size=10):
        """
        Credit: Yann N. Dauphin
        """

        inds = range(x.shape[0])
        n_batches = int(np.ceil(float(len(inds)) / batch_size))

        times = []
        nlls = []
        for i in range(n_batches):
            begin = time.time()
            nll = self.sess.run(parzen, feed_dict={self.x: x[inds[i::n_batches]]})
            # nll = parzen(x[inds[i::n_batches]])
            end = time.time()
            times.append(end-begin)
            nlls.extend(nll)

            if i % 10 == 0:
                print( i, np.mean(times), np.mean(nlls))

        return np.array(nlls)

    def cross_validate_sigma(self, samples, data, sigmas, batch_size):

        lls = []
        for sigma in sigmas:
            print(sigma)
            parzen = tf_parzen(self.x, samples, sigma)
            tmp = self.get_nll(data, parzen, batch_size=batch_size)
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
        parzen = tf_parzen(self.x, samples, sigma)
        ll = self.get_nll(test, parzen, batch_size=self.batch_size)
        se = ll.std() / np.sqrt(test.shape[0])

        print("Log-Likelihood of test set = {}, se: {}".format(ll.mean(), se))

        return ll.mean(), se

def main():
    parser = argparse.ArgumentParser(description = 'Parzen window, log-likelihood estimator')
    parser.add_argument('-p', '--path', help='model path')
    parser.add_argument('-s', '--sigma', default = None)
    parser.add_argument('-d', '--dataset', choices=['mnist', 'tfd'], default='mnist')
    parser.add_argument('-f', '--fold', default = 0, type=int)
    parser.add_argument('-v', '--valid', default = False, action='store_true')
    parser.add_argument('-n', '--num_samples', default=10000, type=int)
    parser.add_argument('-l', '--limit_size', default=1000, type=int)
    parser.add_argument('-b', '--batch_size', default=100, type=int)
    parser.add_argument('-c', '--cross_val', default=10, type=int,
                            help="Number of cross valiation folds")
    parser.add_argument('--sigma_start', default=-1, type=float)
    parser.add_argument('--sigma_end', default=0., type=float)
    args = parser.parse_args()

    # load model
    # model = serial.load(args.path)
    # src = model.dataset_yaml_src
    # model.set_batch_size(batch_size)

    # load test set
    # test = yaml_parse.load(src)
    # test = get_test(args.dataset, test, args.fold)
    test = get_test(args.dataset)
    test = np.multiply(test, 1.0 / 255)
    print(test.shape, np.max(test), np.min(test))
    # generate samples
    # samples = model.generator.sample(args.num_samples).eval()
    # output_space = model.generator.mlp.get_output_space()
    # if 'Conv2D' in str(output_space):
    #     samples = output_space.convert(samples, output_space.axes, ('b', 0, 1, 'c'))
    #     samples = samples.reshape((samples.shape[0], np.prod(samples.shape[1:])))
    # del model
    # gc.collect()
    # samples = np.asarray(io.loadmat('output/MNIST/test/data10000.mat')['data'], dtype=np.float32).reshape([-1, 784])
    samples = np.load('samples.npy').astype(np.float32)
    samples = samples[:10000]
    samples = normalize_data(samples, low=0, high=255, shape=(28, 28))
    samples = np.reshape(samples, [-1, 784])
    print(samples.shape, np.max(samples), np.min(samples))
    # samples = np.random.uniform(0, 1, size=[10000, 784]).astype(np.float32)
    # train_data, train_labels = maybe_download_minst('../data/mnist', train=True, one_hot=False)
    # indices = np.arange(len(train_data))
    # np.random.shuffle(indices)
    # indices = indices[:10000]
    # train_data = train_data[indices]
    # samples = np.reshape(train_data, [-1, 784]).astype(np.float32)
    # samples = np.multiply(samples, 1.0 /255)
    # print(samples.shape)

    with tf.Session() as sess:
        kde = ParsenDensityEsimator(sess, args)
        kde.fit(samples, test)
    # gc.collect()


    # valid
    # if args.valid:
    #     valid = get_valid(args.dataset)
    #     ll = get_nll(valid, parzen, batch_size = batch_size)
    #     se = ll.std() / np.sqrt(val.shape[0])
    #     print("Log-Likelihood of valid set = {}, se: {}".format(ll.mean(), se))


if __name__ == "__main__":
    main()