from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import tensorflow as tf
from model.model import CoopNets

FLAGS = tf.app.flags.FLAGS

# learning parameters
tf.flags.DEFINE_string('net_type', 'scene', 'network type: [scene/mnist]')
tf.flags.DEFINE_integer('image_size', 64, 'Image size to rescale images') # 64 for scene, and 28 for mnist
tf.flags.DEFINE_integer('batch_size', 100, 'Batch size of training images')
tf.flags.DEFINE_integer('num_epochs', 1000, 'Number of epochs to train')
tf.flags.DEFINE_integer('nTileRow', 12, 'Row number of synthesized images')
tf.flags.DEFINE_integer('nTileCol', 12, 'Column number of synthesized images')
tf.flags.DEFINE_float('beta1_des', 0.5, 'Momentum term of adam')
tf.flags.DEFINE_float('beta1_gen', 0.5, 'Momentum term of adam')

# parameters for descriptorNet
tf.flags.DEFINE_float('d_lr', 0.007, 'Initial learning rate for descriptor')
tf.flags.DEFINE_float('des_refsig', 0.016, 'Standard deviation for reference distribution of descriptor')
tf.flags.DEFINE_integer('des_sample_steps', 15, 'Sample steps for Langevin dynamics of descriptor')
tf.flags.DEFINE_float('des_step_size', 0.002, 'Step size for descriptor Langevin dynamics')

# parameters for generatorNet
tf.flags.DEFINE_float('g_lr', 0.0001, 'Initial learning rate for generator')  # 0.0001
tf.flags.DEFINE_float('gen_refsig', 0.3, 'Standard deviation for reference distribution of generator')
tf.flags.DEFINE_integer('gen_sample_steps', 0, 'Sample steps for Langevin dynamics of generator')
tf.flags.DEFINE_float('gen_step_size', 0.1, 'Step size for generator Langevin dynamics')
tf.flags.DEFINE_integer('gen_latent_size', 100, 'Number of dimensions of latent variables')

# utils
tf.flags.DEFINE_string('data_dir', './data', 'The data directory')
tf.flags.DEFINE_string('category', 'rock', 'The name of dataset')
tf.flags.DEFINE_boolean('prefetch', True, 'True if reading all images at once')
tf.flags.DEFINE_boolean('calculate_inception', False, 'True if inception score is calculated (only for scene dataset)')
tf.flags.DEFINE_boolean('calculate_parzen', False, 'True if parzen score is calculated (only for MNIST dataset)')
tf.flags.DEFINE_integer('read_len', 500, 'Number of batches per reading')
tf.flags.DEFINE_string('output_dir', './output', 'The output directory for saving results')
tf.flags.DEFINE_integer('log_step', 50, 'Number of minibatches to save output results')
tf.flags.DEFINE_boolean('test', False, 'True if in testing mode')
tf.flags.DEFINE_string('test_type', 'inter', 'testing type: [inter/syn]: inter: interpolation | syn: synthesis')
tf.flags.DEFINE_string('ckpt', None, 'Checkpoint path to load: e.g., output/rock/checkpoints/model.ckpt-2000')
tf.flags.DEFINE_integer('sample_size', 100, 'Number of images to generate during test.')


def main(_):

    output_dir = os.path.join(FLAGS.output_dir, FLAGS.category)
    sample_dir = os.path.join(output_dir, 'synthesis')
    log_dir = os.path.join(output_dir, 'log')
    model_dir = os.path.join(output_dir, 'checkpoints')
    if FLAGS.test_type == 'inter':
        test_dir = os.path.join(output_dir, 'test/interpolation')
    elif FLAGS.test_type == 'syn':
        test_dir = os.path.join(output_dir, 'test/synthesis')
    else:
        return NotImplementedError

    model = CoopNets(

        net_type=FLAGS.net_type,
        num_epochs=FLAGS.num_epochs,
        image_size=FLAGS.image_size,
        batch_size=FLAGS.batch_size,
        beta1_des=FLAGS.beta1_des,
        beta1_gen=FLAGS.beta1_gen,
        nTileRow=FLAGS.nTileRow, nTileCol=FLAGS.nTileCol,
        d_lr=FLAGS.d_lr, g_lr=FLAGS.g_lr,
        des_refsig=FLAGS.des_refsig, gen_refsig=FLAGS.gen_refsig,
        des_step_size=FLAGS.des_step_size, gen_step_size=FLAGS.gen_step_size,
        des_sample_steps=FLAGS.des_sample_steps, gen_sample_steps=FLAGS.gen_sample_steps,
        gen_latent_size=FLAGS.gen_latent_size,
        log_step=FLAGS.log_step, data_path=FLAGS.data_dir, category=FLAGS.category,
        sample_dir=sample_dir, log_dir=log_dir, model_dir=model_dir, test_dir=test_dir,
        prefetch=FLAGS.prefetch, read_len=FLAGS.read_len, output_dir=output_dir,
        calculate_inception=FLAGS.calculate_inception, calculate_parzen=FLAGS.calculate_parzen
    )

    with tf.Session() as sess:
        if FLAGS.test:

            if tf.gfile.Exists(test_dir):
                tf.gfile.DeleteRecursively(test_dir)
            tf.gfile.MakeDirs(test_dir)

            if FLAGS.test_type == 'syn':
                model.sampling(sess, FLAGS.ckpt, 55000, 10)
            elif FLAGS.test_type == 'inter':
                model.interpolation(sess, FLAGS.ckpt, 55000)
            else:
                return NotImplementedError

        else:
            if tf.gfile.Exists(log_dir):
                tf.gfile.DeleteRecursively(log_dir)
            tf.gfile.MakeDirs(log_dir)

            if tf.gfile.Exists(sample_dir):
                tf.gfile.DeleteRecursively(sample_dir)
            tf.gfile.MakeDirs(sample_dir)

            if tf.gfile.Exists(model_dir):
                tf.gfile.DeleteRecursively(model_dir)
            tf.gfile.MakeDirs(model_dir)

            model.train(sess)

if __name__ == '__main__':
    tf.app.run()
