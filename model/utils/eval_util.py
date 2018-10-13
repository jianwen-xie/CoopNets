from __future__ import division
import scipy.io as sio
from model.utils.parzen_ll import get_test
import numpy as np
from model.utils.parzen_args import args

def eval_parzen(kde, samples_des, samples_gen):

    samples_des = np.maximum(-1, np.minimum(1, samples_des))
    samples_des = (samples_des - samples_des.min()) / (samples_des.max() - samples_des.min())
    samples_des = np.asarray(np.reshape(samples_des, [-1, samples_des.shape[1]*samples_des.shape[2]]), dtype=np.float32)

    samples_gen = np.maximum(-1, np.minimum(1, samples_gen))
    samples_gen = (samples_gen - samples_gen.min()) / (samples_gen.max() - samples_gen.min())
    samples_gen = np.asarray(np.reshape(samples_gen, [-1, samples_gen.shape[1]*samples_gen.shape[2]]), dtype=np.float32)
    test = get_test(args.dataset)
    test = np.multiply(test, 1.0 / 255)

    parzon_des_mean, parzon_des_se = kde.fit(samples_des, test)
    parzon_gen_mean, parzon_gen_se = kde.fit(samples_gen, test)

    return parzon_des_mean, parzon_des_se, parzon_gen_mean, parzon_gen_se

