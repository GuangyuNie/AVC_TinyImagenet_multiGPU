'''
Multi GPU training for Xception on TinyImagenet with Madry's method
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import re
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from utils_multiGPU import *
import model
from pgd_attack import *
DIST = 'Linf'

def log_output(hist, sess, accuracy, loss, feed_dict, feed_dict_adv, feed_dict_test, feed_dict_test_adv):
    ## training
    # natural
    train_acc_i, train_loss_i = sess.run([accuracy, loss], feed_dict)
    hist['train_acc'] += [train_acc_i]
    hist['train_loss'] += [train_loss_i]
    # adversarial
    train_adv_acc_i, train_adv_loss_i = sess.run([accuracy, loss], feed_dict_adv)
    hist['train_adv_acc'] += [train_adv_acc_i]
    hist['train_adv_loss'] += [train_adv_loss_i]

    ## test
    test_acc_i, test_loss_i = sess.run([accuracy, loss], feed_dict_test)
    hist['test_acc'] += [test_acc_i]
    hist['test_loss'] += [test_loss_i]
    # adversarial
    test_adv_acc_i, test_adv_loss_i = sess.run([accuracy, loss], feed_dict_test_adv)
    hist['test_adv_acc'] += [test_adv_acc_i]
    hist['test_adv_loss'] += [test_adv_loss_i]

    print('train_acc:{:.4f}       train_loss:{:.4f}'.format(train_acc_i, train_loss_i))
    print('train_adv_acc:{:.4f}      train_adv_loss:{:.4f}'.format(train_adv_acc_i, train_adv_loss_i))
    print('test_acc:{:.4f}       test_loss:{:.4f}'.format(test_acc_i, test_loss_i))
    print('test_adv_acc:{:.4f}      test_adv_loss:{:.4f}'.format(test_adv_acc_i, test_adv_loss_i))

    return hist


def load_data():
    ## load training data ##
    from random import shuffle
    train_data = np.load('./data/train_data.npy', encoding=('latin1')).item()
    train_images = train_data['image']
    train_labels = train_data['label']
    idx = list(range(len(train_images)))
    shuffle(idx)
    train_images = train_images[idx]
    train_labels = train_labels[idx]
    # testing
    test_data = np.load('./data/val_data.npy', encoding=('latin1')).item()
    test_images = test_data['image']
    test_labels = test_data['label']
    idx_v = list(range(len(test_images)))
    shuffle(idx_v)
    test_images = test_images[idx_v]
    test_labels = test_labels[idx_v]
    return train_images, train_labels, test_images, test_labels

def train():
    """Train CIFAR-10 for a number of steps."""
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        # Create a variable to count the number of train() calls. This equals the
        # number of batches processed * FLAGS.num_gpus.
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        image_batch_pl = tf.placeholder(tf.float32,  shape = (batch_size, 64, 64, 3), name = 'input_images')
        label_batch_pl = tf.placeholder(tf.int64, shape=(batch_size), name='labels')
        is_training_pl = tf.placeholder(tf.bool, shape=(), name='labels')
        opt = tf.train.AdamOptimizer(lr)


        # BUILD MODEL
        tower_grads = []
        adv_grads = []
        accuracy =[]
        batch_size_i = batch_size // FLAGS.num_gpus
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (model.TOWER_NAME, i)) as scope:
                        # Calculate the loss for one tower of the CIFAR model. This function
                        # constructs the entire CIFAR model but shares the variables across
                        # all towers.
                        image_batch_pl_i = image_batch_pl[i*batch_size_i:(i+1)*batch_size_i]
                        label_batch_pl_i = label_batch_pl[i*batch_size_i:(i+1)*batch_size_i]
                        loss, cw_loss, acc_i = tower_loss(scope, image_batch_pl_i, label_batch_pl_i, is_training_pl)
                        adv_grad_i = tf.gradients(loss, image_batch_pl_i)[0]


                        batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grad_i = opt.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grad_i)
                        # track all adversarial gradients, by Hope
                        adv_grads.append(adv_grad_i)
                        accuracy.append(acc_i)

        accuracy = tf.reduce_mean(accuracy)
        adv_grads = tf.concat(adv_grads, 0)
        # mean of each gradient. synchronization point across all towers.
        grads = average_gradients(tower_grads)
        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

        # ## SUMMARY
        # # Add a summary to track the learning rate.
        # summaries.append(tf.summary.scalar('learning_rate', lr))
        # # Add histograms for gradients.
        # for grad, var in grads:
        #     if grad is not None:
        #         summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        # # Add histograms for trainable variables.
        # for var in tf.trainable_variables():
        #     summaries.append(tf.summary.histogram(var.op.name, var))
        # # Build the summary operation from the last tower summaries.
        # summary_op = tf.summary.merge(summaries)


        ## INIT
        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        # Build an initialization operation to run below.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        init = tf.global_variables_initializer()
        sess.run(init)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        ## RESTORE
        # Create a saver. by Guangyu
        g_list = [op.name for op in tf.get_default_graph().get_operations() if op.op_def and op.op_def.name=='VariableV2']
        not_restore = [str(g)+':0' for g in g_list if 'xxx' in g]
        not_resotre = not_restore.append('global_step:0')
        restore_list = [v for v in tf.global_variables() if v.name not in not_restore]
        saver = tf.train.Saver(var_list = restore_list)
        saver.restore(sess, "/home/hope-yao/Documents/models/tutorials/image/AVC_Madry_multiGPU_pretrain/model_save_base_final/center_loss.ckpt")

        ## LOAD DATA
        train_images, train_labels, test_images, test_labels = load_data()
        itr_per_epoch = train_images.shape[0] // batch_size
        itr_per_epoch_test = test_images.shape[0] // batch_size
        hist = {'train_loss': [],
                'train_acc': [],
                'train_adv_loss': [],
                'train_adv_acc': [],
                'test_loss': [],
                'test_acc': [],
                'test_adv_loss': [],
                'test_adv_acc': []}

        ## START TRAINING
        for ep_i in xrange(FLAGS.max_epoch):
            for itr_i in range(itr_per_epoch):
                start_time = time.time()

                x_batch_nat = train_images[itr_i * batch_size:(1 + itr_i) * batch_size]
                y_batch = train_labels[itr_i * batch_size:(1 + itr_i) * batch_size]
                feed_dict_pgd = {image_batch_pl: x_batch_nat/255.,
                                 label_batch_pl: y_batch,
                                 is_training_pl: False}
                x_batch_adv = get_PGD(sess, adv_grads, feed_dict_pgd, image_batch_pl, dist=DIST)
                feed_dict_adv = {image_batch_pl: x_batch_adv*255.,
                                 label_batch_pl: y_batch,
                                 is_training_pl: True}
                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict_adv)

                if itr_i%10==0:
                    # output training
                    duration = time.time() - start_time
                    num_examples_per_step = batch_size * 10
                    examples_per_sec = num_examples_per_step / duration
                    print('%s: ep %d, itr %d, loss = %.2f,  %.1f examples/sec' %(datetime.now(), ep_i, itr_i, loss_value, examples_per_sec))

                # output testing
                if itr_i%100==0:
                    feed_dict_pgd = {image_batch_pl: x_batch_nat/255.,
                                     label_batch_pl: y_batch,
                                     is_training_pl: False}
                    x_batch_adv = get_PGD(sess, adv_grads, feed_dict_pgd, image_batch_pl, dist=DIST) # DICTIONARY IS PASSED VIA REFERENCE
                    feed_dict_train_adv = {image_batch_pl: x_batch_adv*255.,
                                           label_batch_pl: y_batch,
                                           is_training_pl: False}
                    feed_dict_train = {image_batch_pl: x_batch_nat,
                                       label_batch_pl: y_batch,
                                       is_training_pl: False}

                    testing_batch_i = np.random.choice(itr_per_epoch_test, 1)[0]  # randomly pick a batch for testing
                    x_batch_nat_test = test_images[testing_batch_i * batch_size:(1 + testing_batch_i) * batch_size]
                    y_batch_test = test_labels[testing_batch_i * batch_size:(1 + testing_batch_i) * batch_size]
                    feed_dict_pgd = {image_batch_pl: x_batch_nat_test/255.,
                                      label_batch_pl: y_batch_test,
                                      is_training_pl: False}
                    x_batch_adv_test = get_PGD(sess, adv_grads, feed_dict_pgd, image_batch_pl, dist=DIST)
                    feed_dict_test_adv = {image_batch_pl: x_batch_adv_test*255.,
                                          label_batch_pl: y_batch_test,
                                          is_training_pl: False}
                    feed_dict_test = {image_batch_pl: x_batch_nat_test,
                                      label_batch_pl: y_batch_test,
                                      is_training_pl: False}

                    hist = log_output(hist, sess, accuracy, loss, feed_dict_train, feed_dict_train_adv, feed_dict_test, feed_dict_test_adv)
                    np.save(os.path.join(log_dir, 'hist'), hist)

            if ep_i%5==0:
                saver.save(sess, os.path.join(log_dir, 'AVC_Madry_multiGPU_ep{}.ckpt'.format(ep_i)))


# if step % 100 == 0:
#  summary_str = sess.run(summary_op)
#  summary_writer.add_summary(summary_str, step)
#
# # Save the model checkpoint periodically.
# if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
#   checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
#   saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':

    FLAGS = tf.app.flags.FLAGS

    tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                               """Directory where to write event logs """
                               """and checkpoint.""")
    tf.app.flags.DEFINE_integer('max_epoch', 2000,
                                """Number of batches to run.""")
    tf.app.flags.DEFINE_integer('num_gpus', 2,
                                """How many GPUs to use.""")
    tf.app.flags.DEFINE_boolean('log_device_placement', False,
                                """Whether to log device placement.""")
    batch_size = 24  # split on 4 or 8 GPU, each GPU has 32 or 16
    lr = 1e-4

    log_dir = './model_save_base_madry'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    train()