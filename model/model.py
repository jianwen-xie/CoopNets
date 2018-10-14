from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from six.moves import xrange

from model.utils.interpolate import *
from model.utils.custom_ops import *
from model.utils.data_io import DataSet, saveSampleResults
from model.utils.parzen_ll import ParsenDensityEsimator
from model.utils.inception_model import *
import scipy.io as sio
from model.utils.eval_util import eval_parzen


class CoopNets(object):
    def __init__(self, num_epochs=200, image_size=64, batch_size=100, nTileRow=12, nTileCol=12, net_type='scene',
                 d_lr=0.001, g_lr=0.0001, beta1_gen=0.5, beta1_des=0.5,
                 des_step_size=0.002, des_sample_steps=10, des_refsig=0.016,
                 gen_step_size=0.1, gen_sample_steps=0, gen_refsig=0.3, gen_latent_size=100,
                 data_path='./data/', log_step=10, category='volcano',
                 sample_dir='./synthesis', model_dir='./checkpoints', log_dir='./log', test_dir='./test',
                 prefetch=True, read_len=500, output_dir='', calculate_inception=False, calculate_parzen=False):
        self.type = net_type
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.nTileRow = nTileRow
        self.nTileCol = nTileCol
        self.num_chain = nTileRow * nTileCol
        self.beta1_des = beta1_des
        self.beta1_gen = beta1_gen
        self.prefetch = prefetch
        self.read_len = read_len
        self.category = category
        self.num_channel = 1 if net_type == 'mnist' else 3

        self.calculate_inception = True if (net_type == "scene" and calculate_inception) else False
        self.calculate_parzen = True if (net_type == "mnist" and calculate_parzen) else False
        self.output_dir = output_dir

        self.d_lr = d_lr
        self.g_lr = g_lr
        self.delta1 = des_step_size
        self.sigma1 = des_refsig
        self.delta2 = gen_step_size
        self.sigma2 = gen_refsig
        self.t1 = des_sample_steps
        self.t2 = gen_sample_steps

        self.data_path = os.path.join(data_path, category)
        self.log_step = log_step

        self.log_dir = log_dir
        self.sample_dir = sample_dir
        self.model_dir = model_dir
        self.test_dir = test_dir
        self.z_size = gen_latent_size

        self.syn = tf.placeholder(shape=[None, self.image_size, self.image_size, self.num_channel], dtype=tf.float32, name='syn')
        self.obs = tf.placeholder(shape=[None, self.image_size, self.image_size, self.num_channel], dtype=tf.float32, name='obs')
        self.z = tf.placeholder(shape=[None, self.z_size], dtype=tf.float32, name='z')

        self.debug = False

    def build_model(self):

        self.gen_res = self.generator(self.z, reuse=False)
        obs_res = self.descriptor(self.obs, reuse=False)
        syn_res = self.descriptor(self.syn, reuse=True)

        self.recon_err = tf.reduce_mean(
            tf.pow(tf.subtract(tf.reduce_mean(self.syn, axis=0), tf.reduce_mean(self.obs, axis=0)), 2))
        self.recon_err_mean, self.recon_err_update = tf.contrib.metrics.streaming_mean(self.recon_err)

        # descriptor variables
        des_vars = [var for var in tf.trainable_variables() if var.name.startswith('des')]

        self.des_loss = tf.subtract(tf.reduce_mean(syn_res, axis=0), tf.reduce_mean(obs_res, axis=0))
        self.des_loss_mean, self.des_loss_update = tf.contrib.metrics.streaming_mean(self.des_loss)

        des_optim = tf.train.AdamOptimizer(self.d_lr, beta1=self.beta1_des)
        des_grads_vars = des_optim.compute_gradients(self.des_loss, var_list=des_vars)
        des_grads = [tf.reduce_mean(tf.abs(grad)) for (grad, var) in des_grads_vars if '/w' in var.name]
        # update by mean of gradients
        self.apply_d_grads = des_optim.apply_gradients(des_grads_vars)

        # generator variables
        gen_vars = [var for var in tf.trainable_variables() if var.name.startswith('gen')]

        self.gen_loss = tf.reduce_mean(1.0 / (2 * self.sigma2 * self.sigma2) * tf.square(self.obs - self.gen_res),
                                       axis=0)
        # print(self.gen_loss)
        self.gen_loss_mean, self.gen_loss_update = tf.contrib.metrics.streaming_mean(self.gen_loss)

        gen_optim = tf.train.AdamOptimizer(self.g_lr, beta1=self.beta1_gen)
        gen_grads_vars = gen_optim.compute_gradients(self.gen_loss, var_list=gen_vars)
        gen_grads = [tf.reduce_mean(tf.abs(grad)) for (grad, var) in gen_grads_vars if '/w' in var.name]
        self.apply_g_grads = gen_optim.apply_gradients(gen_grads_vars)

        # symbolic langevins
        self.langevin_descriptor = self.langevin_dynamics_descriptor(self.syn)
        self.langevin_generator = self.langevin_dynamics_generator(self.z)

        tf.summary.scalar('des_loss', self.des_loss_mean)
        tf.summary.scalar('gen_loss', self.gen_loss_mean)
        tf.summary.scalar('recon_err', self.recon_err_mean)

        self.summary_op = tf.summary.merge_all()

    def langevin_dynamics_descriptor(self, syn_arg):
        def cond(i, syn):
            return tf.less(i, self.t1)

        def body(i, syn):
            noise = tf.random_normal(shape=[self.num_chain, self.image_size, self.image_size, self.num_channel], name='noise')
            syn_res = self.descriptor(syn, reuse=True)
            grad = tf.gradients(syn_res, syn, name='grad_des')[0]
            syn = syn - 0.5 * self.delta1 * self.delta1 * (syn / self.sigma1 / self.sigma1 - grad) + self.delta1 * noise
            return tf.add(i, 1), syn

        with tf.name_scope("langevin_dynamics_descriptor"):
            i = tf.constant(0)
            i, syn = tf.while_loop(cond, body, [i, syn_arg])
            return syn

    def langevin_dynamics_generator(self, z_arg):
        def cond(i, z):
            return tf.less(i, self.t2)

        def body(i, z):
            noise = tf.random_normal(shape=[self.num_chain, self.z_size], name='noise')

            gen_res = self.generator(z, reuse=True)
            gen_loss = tf.reduce_mean(1.0 / (2 * self.sigma2 * self.sigma2) * tf.square(self.obs - gen_res),
                                       axis=0)
            grad = tf.gradients(gen_loss, z, name='grad_gen')[0]
            z = z - 0.5 * self.delta2 * self.delta2 * (z + grad) + self.delta2 * noise
            return tf.add(i, 1), z

        with tf.name_scope("langevin_dynamics_generator"):
            i = tf.constant(0)
            i, z = tf.while_loop(cond, body, [i, z_arg])
            return z

    def train(self, sess):

        self.build_model()

        # Prepare training data
        is_mnist = True if self.type == "mnist" else False
        dataset = DataSet(self.data_path, image_size=self.image_size, batch_sz=self.batch_size,
                          prefetch=self.prefetch, read_len=self.read_len, is_mnist=is_mnist)

        # initialize training
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        sample_results_des = np.random.randn(self.num_chain * dataset.num_batch, self.image_size, self.image_size, self.num_channel)
        sample_results_gen = np.random.randn(self.num_chain * dataset.num_batch, self.image_size, self.image_size, self.num_channel)

        saver = tf.train.Saver(max_to_keep=50)

        writer = tf.summary.FileWriter(self.log_dir, sess.graph)

        # measure 1: Parzon-window based likelihood
        if self.calculate_parzen:
            from model.utils.parzen_args import args
            kde = ParsenDensityEsimator(sess, args)
            parzon_des_mean_list, parzon_des_se_list = [], []
            parzon_gen_mean_list, parzon_gen_se_list = [], []
            parzon_log_file = os.path.join(self.output_dir, 'parzon.txt')
            parzon_write_file = os.path.join(self.output_dir, 'parzon.mat')
            parzon_syn_data_file = os.path.join(self.output_dir, 'parzon_syn_dat.mat')
            parzon_max = -10000

        # measure 2: inception score
        if self.calculate_inception:
            inception_log_file = os.path.join(self.output_dir, 'inception.txt')
            inception_write_file = os.path.join(self.output_dir, 'inception.mat')

        # make graph immutable
        tf.get_default_graph().finalize()

        # store graph in protobuf
        with open(self.model_dir + '/graph.proto', 'w') as f:
            f.write(str(tf.get_default_graph().as_graph_def()))

        inception_mean, inception_sd = [], []

        # train
        minibatch = -1

        for epoch in xrange(self.num_epochs):
            start_time = time.time()
            for i in xrange(dataset.num_batch):
                minibatch = minibatch + 1
                obs_data = dataset.get_batch()

                # Step G0: generate X ~ N(0, 1)
                z_vec = np.random.randn(self.num_chain, self.z_size)
                g_res = sess.run(self.gen_res, feed_dict={self.z: z_vec})
                # Step D1: obtain synthesized images Y
                if self.t1 > 0:
                    syn = sess.run(self.langevin_descriptor, feed_dict={self.syn: g_res})
                # Step G1: update X using Y as training image
                if self.t2 > 0:
                    z_vec = sess.run(self.langevin_generator, feed_dict={self.z: z_vec, self.obs: syn})
                # Step D2: update D net
                d_loss = sess.run([self.des_loss, self.des_loss_update, self.apply_d_grads],
                                  feed_dict={self.obs: obs_data, self.syn: syn})[0]
                # Step G2: update G net
                g_loss = sess.run([self.gen_loss, self.gen_loss_update, self.apply_g_grads],
                                  feed_dict={self.obs: syn, self.z: z_vec})[0]

                # Compute MSE
                mse = sess.run([self.recon_err, self.recon_err_update],
                               feed_dict={self.obs: obs_data, self.syn: syn})[0]

                sample_results_gen[i * self.num_chain:(i + 1) * self.num_chain] = g_res
                sample_results_des[i * self.num_chain:(i + 1) * self.num_chain] = syn

                if minibatch % self.log_step == 0:
                    end_time = time.time()
                    [des_loss_avg, gen_loss_avg, mse_avg, summary] = sess.run([self.des_loss_mean, self.gen_loss_mean,
                                                                               self.recon_err_mean, self.summary_op])
                    writer.add_summary(summary, minibatch)
                    writer.flush()
                    print('Epoch #{:d}, minibatch #{:d}, avg.des loss: {:.4f}, avg.gen loss: {:.4f}, '
                          'avg.L2 dist: {:4.4f}, time: {:.2f}s'.format(epoch, minibatch, des_loss_avg, gen_loss_avg,
                                                                       mse_avg, end_time - start_time))
                    start_time = time.time()

                    # save synthesis images
                    if not os.path.exists(self.sample_dir):
                        os.makedirs(self.sample_dir)
                    saveSampleResults(syn, "%s/des_%06d_%06d.png" % (self.sample_dir, epoch, minibatch), col_num=self.nTileCol)
                    saveSampleResults(g_res, "%s/gen_%06d_%06d.png" % (self.sample_dir, epoch, minibatch), col_num=self.nTileCol)

                if minibatch % (self.log_step * 20) == 0:
                    # save check points
                    if not os.path.exists(self.model_dir):
                        os.makedirs(self.model_dir)
                    saver.save(sess, "%s/%s" % (self.model_dir, 'model.ckpt'), global_step=minibatch)

            if self.calculate_inception and epoch % 20 == 0:

                sample_results_partial = sample_results_des[:len(dataset)]
                sample_results_partial = np.minimum(1, np.maximum(-1, sample_results_partial))
                sample_results_partial = (sample_results_partial + 1) / 2 * 255

                m, s = get_inception_score(sample_results_partial)
                print("Inception score: mean {}, sd {}".format(m, s))
                fo = open(inception_log_file, 'a')
                fo.write("Epoch {}: mean {}, sd {} \n".format(epoch, m, s))
                fo.close()
                inception_mean.append(m)
                inception_sd.append(s)
                sio.savemat(inception_write_file, {'mean': np.asarray(inception_mean), 'sd': np.asarray(inception_sd)})

            if self.calculate_parzen:

                samples_des = sample_results_des[:10000]
                samples_gen = sample_results_gen[:10000]

                parzon_des_mean, parzon_des_se, parzon_gen_mean, parzon_gen_se = eval_parzen(kde, samples_des, samples_gen)

                parzon_des_mean_list.append(parzon_des_mean)
                parzon_des_se_list.append(parzon_des_se)
                parzon_gen_mean_list.append(parzon_gen_mean)
                parzon_gen_se_list.append(parzon_gen_se)

                if parzon_des_mean > parzon_max:
                    parzon_max = parzon_des_mean
                    sio.savemat(parzon_syn_data_file, {'samples_des': samples_des, 'samples_gen': samples_gen})

                fo = open(parzon_log_file, 'a')
                fo.write("Epoch {}: des mean {}, sd {}; gen mean {}, sd {}, max score {}. \n".
                         format(epoch, parzon_des_mean, parzon_des_se, parzon_gen_mean, parzon_gen_se, parzon_max))
                fo.close()

                sio.savemat(parzon_write_file, {'parzon_des_mean': np.asarray(parzon_des_mean_list),
                                                'parzon_des_se': np.asarray(parzon_des_se_list),
                                                'parzon_gen_mean': np.asarray(parzon_gen_mean_list),
                                                'parzon_gen_se': np.asarray(parzon_gen_se_list)})

    def interpolation(self, sess, ckpt, sample_size):
        assert ckpt is not None, 'no checkpoint provided.'

        gen_res = self.generator(self.z, reuse=False)
        num_batches = int(math.ceil(sample_size / self.num_chain))

        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt)
        print('Loading checkpoint {}.'.format(ckpt))

        for i in xrange(num_batches):
            z_vec = np.random.randn(self.num_chain, self.z_size)
            # g_res = sess.run(gen_res, feed_dict={self.z: z_vec})
            # saveSampleResults(g_res, "%s/gen%03d.png" % (self.test_dir, i), col_num=self.nTileCol)

            # output interpolation results
            interp_z = linear_interpolator(z_vec, npairs=self.nTileRow, ninterp=self.nTileCol)
            interp = sess.run(gen_res, feed_dict={self.z: interp_z})
            saveSampleResults(interp, "%s/interp%03d.png" % (self.test_dir, i), col_num=self.nTileCol)

        print("The results are saved in a folder: {}".format(self.test_dir))

    def sampling(self, sess, ckpt, sample_size, sample_step, calculate_inception=False):
        assert ckpt is not None, 'no checkpoint provided.'

        self.t1 = sample_step

        gen_res = self.generator(self.z, reuse=False)
        obs_res = self.descriptor(self.obs, reuse=False)

        self.langevin_descriptor = self.langevin_dynamics_descriptor(gen_res)
        num_batches = int(math.ceil(sample_size / self.num_chain))

        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, ckpt)
        print('Loading checkpoint {}.'.format(ckpt))

        sample_results_des = np.random.randn(self.num_chain * num_batches, self.image_size, self.image_size, self.num_channel)
        for i in xrange(num_batches):
            z_vec = np.random.randn(self.num_chain, self.z_size)

            # synthesis by generator
            g_res = sess.run(gen_res, feed_dict={self.z: z_vec})
            saveSampleResults(g_res, "%s/gen%03d_test.png" % (self.test_dir, i), col_num=self.nTileCol)

            # synthesis by descriptor and generator
            syn = sess.run(self.langevin_descriptor, feed_dict={self.z: z_vec})
            saveSampleResults(syn, "%s/des%03d_test.png" % (self.test_dir, i), col_num=self.nTileCol)

            sample_results_des[i * self.num_chain:(i + 1) * self.num_chain] = syn

            if i % 10 == 0:
                print("Sampling batches: {}, from {} to {}".format(i, i * self.num_chain,
                                                                   min((i+1) * self.num_chain, sample_size)))
        sample_results_des = sample_results_des[:sample_size]
        sample_results_des = np.minimum(1, np.maximum(-1, sample_results_des))
        sample_results_des = (sample_results_des + 1) / 2 * 255

        if calculate_inception:
            m, s = get_inception_score(sample_results_des)
            print("Inception score: mean {}, sd {}".format(m, s))

        sampling_output_file = os.path.join(self.output_dir, 'samples_des.npy')
        np.save(sampling_output_file, sample_results_des)
        print("The results are saved in folder: {}".format(self.output_dir))

    def descriptor(self, inputs, reuse=False):
        with tf.variable_scope('des', reuse=reuse):

            if self.type == 'scene':
                conv1 = conv2d(inputs, 64, kernal=(5, 5), strides=(2, 2), padding="SAME", activate_fn=leaky_relu,
                                name="conv1")

                conv2 = conv2d(conv1, 128, kernal=(3, 3), strides=(2, 2), padding="SAME", activate_fn=leaky_relu,
                                name="conv2")

                conv3 = conv2d(conv2, 256, kernal=(3, 3), strides=(1, 1), padding="SAME", activate_fn=leaky_relu,
                                name="conv3")

                fc = fully_connected(conv3, 100, name="fc")

                return fc

            elif self.type == 'mnist':
                conv1 = conv2d(inputs, 64, kernal=(4, 4), strides=(2, 2), padding="SAME", activate_fn=leaky_relu,
                               name="conv1")

                conv2 = conv2d(conv1, 128, kernal=(4, 4), strides=(2, 2), padding="SAME", activate_fn=leaky_relu,
                               name="conv2")

                conv3 = conv2d(conv2, 256, kernal=(4, 4), strides=(2, 2), padding="SAME", activate_fn=leaky_relu,
                               name="conv3")

                fc = fully_connected(conv3, 100, name="fc")

                return fc

            else:
                return NotImplementedError

    def generator(self, inputs, reuse=False, is_training=True):
        with tf.variable_scope('gen', reuse=reuse):
            if self.type == 'scene':

                inputs = tf.reshape(inputs, [-1, 1, 1, self.z_size])
                convt1 = convt2d(inputs, (None, self.image_size // 16, self.image_size // 16, 512), kernal=(4, 4)
                                 , strides=(1, 1), padding="VALID", name="convt1")
                convt1 = tf.contrib.layers.batch_norm(convt1, is_training=is_training)
                convt1 = leaky_relu(convt1)

                convt2 = convt2d(convt1, (None, self.image_size // 8, self.image_size // 8, 256), kernal=(5, 5)
                                 , strides=(2, 2), padding="SAME", name="convt2")
                convt2 = tf.contrib.layers.batch_norm(convt2, is_training=is_training)
                convt2 = leaky_relu(convt2)

                convt3 = convt2d(convt2, (None, self.image_size // 4, self.image_size // 4, 128), kernal=(5, 5)
                                 , strides=(2, 2), padding="SAME", name="convt3")
                convt3 = tf.contrib.layers.batch_norm(convt3, is_training=is_training)
                convt3 = leaky_relu(convt3)

                convt4 = convt2d(convt3, (None, self.image_size // 2, self.image_size // 2, 64), kernal=(5, 5)
                                 , strides=(2, 2), padding="SAME", name="convt4")
                convt4 = tf.contrib.layers.batch_norm(convt4, is_training=is_training)
                convt4 = leaky_relu(convt4)

                convt5 = convt2d(convt4, (None, self.image_size, self.image_size, self.num_channel), kernal=(5, 5)
                                 , strides=(2, 2), padding="SAME", name="convt5")
                convt5 = tf.nn.tanh(convt5)

                return convt5

            elif self.type == 'mnist':

                inputs = tf.reshape(inputs, [-1, 1, 1, self.z_size])
                convt1 = convt2d(inputs, (None, 4, 4, 512), kernal=(4, 4)
                                 , strides=(1, 1), padding="VALID", name="convt1")
                convt1 = tf.contrib.layers.batch_norm(convt1, is_training=is_training)
                convt1 = leaky_relu(convt1)

                convt2 = convt2d(convt1, (None, 7, 7, 256), kernal=(4, 4)
                                 , strides=(2, 2), padding="SAME", name="convt2")
                convt2 = tf.contrib.layers.batch_norm(convt2, is_training=is_training)
                convt2 = leaky_relu(convt2)

                convt3 = convt2d(convt2, (None, 14, 14, 128), kernal=(4, 4)
                                 , strides=(2, 2), padding="SAME", name="convt3")
                convt3 = tf.contrib.layers.batch_norm(convt3, is_training=is_training)
                convt3 = leaky_relu(convt3)

                convt4 = convt2d(convt3, (None, self.image_size, self.image_size, self.num_channel), kernal=(4, 4)
                                 , strides=(2, 2), padding="SAME", name="convt5")
                convt4 = tf.nn.tanh(convt4)

                return convt4

            else:
                return NotImplementedError