# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using multiple GPUs with synchronous updates.

Accuracy:
multi_gpu_train.py achieves ~86% accuracy after 100K steps (256
epochs of data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
--------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
2 Tesla K20m  | 0.13-0.20              | ~84% at 30K steps  (2.5 hours)
3 Tesla K20m  | 0.13-0.18              | ~84% at 30K steps
4 Tesla K20m  | ~0.10                  | ~84% at 30K steps

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
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
import model
from pgd_attack import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
batch_size = 4 # split on 4 or 8 GPU, each GPU has 32 or 16

def tower_loss(scope, images, labels):
  """Calculate the total loss on a single tower running the CIFAR model.

  Args:
    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
    images: Images. 4D tensor of shape [batch_size, height, width, 3].
    labels: Labels. 1D tensor of shape [batch_size].

  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """

  # Build inference Graph.
  logits, acc = model.inference(images)

  # Build the portion of the Graph calculating the losses. Note that we will
  # assemble the total_loss using a custom function below.
  _ = model.loss(logits, labels)

  # Assemble all of the losses for the current tower only.
  losses = tf.get_collection('losses', scope)

  # Calculate the total loss for the current tower.
  total_loss = tf.add_n(losses, name='total_loss')

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s_[0-9]*/' % model.TOWER_NAME, '', l.op.name)
    tf.summary.scalar(loss_name, l)

  return total_loss, acc


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


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
    lr = 1e-4
    opt = tf.train.AdamOptimizer(lr)


    # Calculate the gradients for each model tower.
    tower_grads = []
    adv_grads = []
    accuracy =[]
    # batch_size = image_batch.get_shape().as_list()[0]
    batch_size_i = batch_size // FLAGS.num_gpus
    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (model.TOWER_NAME, i)) as scope:
            # Calculate the loss for one tower of the CIFAR model. This function
            # constructs the entire CIFAR model but shares the variables across
            # all towers.
            image_batch_pl_i = image_batch_pl[i*batch_size_i:(i+1)*batch_size_i]
            label_batch_pl_i = label_batch_pl[i*batch_size_i:(i+1)*batch_size_i]
            loss, acc_i = tower_loss(scope, image_batch_pl_i, label_batch_pl_i)
            adv_grad_i = tf.gradients(loss, image_batch_pl_i)[0]

            # Reuse variables for the next tower.
            tf.get_variable_scope().reuse_variables()

            # Retain the summaries from the final tower.
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            # Calculate the gradients for the batch of data on this CIFAR tower.
            grads = opt.compute_gradients(loss)

            # Keep track of the gradients across all towers.
            tower_grads.append(grads)
            # track all adversarial gradients, by Hope
            adv_grads.append(adv_grad_i)
            accuracy.append(acc_i)
    accuracy = tf.concat(accuracy, 0)
    adv_grads = tf.concat(adv_grads, 0)
    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Add a summary to track the learning rate.
    summaries.append(tf.summary.scalar('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        model.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    ## load training data ##
    from random import shuffle
    train_images = np.load('./data/train_data.npy', encoding=('latin1')).item()['image']
    train_labels = np.load('./data/train_data.npy', encoding=('latin1')).item()['label']
    idx = list(range(len(train_images)))
    shuffle(idx)
    train_images = train_images[idx]
    train_labels = train_labels[idx]
    # testing
    test_images = np.load('./data/val_data.npy', encoding=('latin1')).item()['image']
    test_labels = np.load('./data/val_data.npy', encoding=('latin1')).item()['label']
    idx_v = list(range(len(test_images)))
    shuffle(idx_v)
    test_images = test_images[idx_v]
    test_labels = test_labels[idx_v]

    itr_per_epoch = train_images.shape[0] // batch_size

    # with tf.variable_scope(tf.get_variable_scope()):
    #   for gpu_i in xrange(FLAGS.num_gpus):
    #     with tf.device('/gpu:%d' % i):
    #       with tf.name_scope('%s_%d' % (model.TOWER_NAME, i)) as scope:
    #         image_batch_pl_i = image_batch_pl[gpu_i * batch_size_i:(gpu_i + 1) * batch_size_i]
    #         label_batch_pl_i = label_batch_pl[gpu_i * batch_size_i:(gpu_i + 1) * batch_size_i]
    #         # x_batch_nat_i = x_batch_nat[gpu_i * batch_size_i:(gpu_i + 1) * batch_size_i]
    #         # y_batch_i = y_batch[gpu_i * batch_size_i:(gpu_i + 1) * batch_size_i]
    #
    #         grad = init_PGD(loss, image_batch_pl_i)
    #
    #         grad_i = grad[gpu_i * batch_size_i:(gpu_i + 1) * batch_size_i]
    #         tf.get_variable_scope().reuse_variables()
    #         x_batch_adv += [x_batch_adv_i]

    hist = {'train_loss': [],
            'train_acc': [],
            'train_adv_loss': [],
            'train_adv_acc': [],
            'test_loss': [],
            'test_acc': [],
            'test_adv_loss': [],
            'test_adv_acc': []}

    for step in xrange(FLAGS.max_steps):
      start_time = time.time()

      jj = step%itr_per_epoch
      x_batch_nat = train_images[jj * batch_size:(1 + jj) * batch_size]
      y_batch = train_labels[jj * batch_size:(1 + jj) * batch_size]
      x_batch_adv = get_PGD(sess, adv_grads, image_batch_pl, label_batch_pl, x_batch_nat, y_batch, epsilon=0.03, a=0.01, k=10, rand=True)
      feed_dict_adv = {image_batch_pl: x_batch_adv,
                       label_batch_pl: y_batch}
      _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict_adv)
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / FLAGS.num_gpus

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

      #if step % 100 == 0:
      #  summary_str = sess.run(summary_op)
       # summary_writer.add_summary(summary_str, step)

      # # Save the model checkpoint periodically.
      # if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
      #   checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
      #   saver.save(sess, checkpoint_path, global_step=step)


      # if step%50==0:
      #     # training
      #
      #     feed_dict = {image_batch_pl: x_batch_nat,
      #                  label_batch_pl: y_batch}
      #     train_acc_i, train_loss_i = sess.run([accuracy, loss], feed_dict)
      #     hist['train_acc'] +=[train_acc_i]
      #     hist['train_loss'] +=[train_loss_i]
      #
      #     feed_dict_adv = {image_batch_pl: x_batch_adv,
      #                      label_batch_pl: y_batch}
      #     train_adv_acc_i, train_adv_loss_i = sess.run([accuracy, loss], feed_dict_adv)
      #     hist['train_adv_acc'] +=[train_adv_acc_i]
      #     hist['train_adv_loss'] +=[train_adv_loss_i]
      #
      #     # test
      #     jjj = jj/10
      #     x_batch_nat_test = test_images[jjj*batch_size:(1+jjj)*batch_size]
      #     y_batch_test = test_labels[jjj*batch_size:(1+jjj)*batch_size]
      #     x_batch_adv_test = attack.perturb(x_batch_nat_test, y_batch_test, sess)
      #
      #     feed_dict={image_batch_pl: x_batch_nat_test,
      #                label_batch_pl: y_batch_test}
      #     test_acc_i, test_loss_i = sess.run([accuracy, loss], feed_dict)
      #     hist['test_acc'] +=[test_acc_i]
      #     hist['test_loss'] +=[test_loss_i]
      #
      #     feed_dict_adv={image_batch_pl: x_batch_adv_test,
      #                    label_batch_pl: y_batch_test}
      #     test_adv_acc_i, test_adv_loss_i = sess.run([accuracy, loss], feed_dict_adv)
      #     hist['test_adv_acc'] +=[test_adv_acc_i]
      #     hist['test_adv_loss'] +=[test_adv_loss_i]
      #
      #     print('train_acc:{:.4f}       train_loss:{:.4f}'.format(train_acc_i,train_loss_i))
      #     print('train_adv_acc:{:.4f}      train_adv_loss:{:.4f}'.format(train_adv_acc_i, train_adv_loss_i))
      #     print('test_acc:{:.4f}       test_loss:{:.4f}'.format(test_acc_i,test_loss_i))
      #     print('test_adv_acc:{:.4f}      test_adv_loss:{:.4f}'.format(test_adv_acc_i, test_adv_loss_i))
      #
      #     np.save('hist', hist)
      #
      #
      #     saver.save(sess,'./model_save_base_madry/center_loss.ckpt')


def main(argv=None):  # pylint: disable=unused-argument

  train()


if __name__ == '__main__':
  tf.app.run()
