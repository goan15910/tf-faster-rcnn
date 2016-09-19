
"""Build Faster-RCNN model
"""

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# Basic model parameters
tf.app.flags.DEFINE_integer('batch_size', 256, """Number of images to process in a batch""")

# Global constants
CONV_STRIDE = 1
WEIGHT_DECAY = 5e-4
DROP_PROB = 0.5 

# Constants describing the training process
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 1e-1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 1e-2       # Initial learning rate.
MOMENTUM = 0.9


def _add_weight_decay(var, wd):
    """Add weight decay constrain to variable"""
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)


def _set_conv_layer(name, tensor, k_shape, stride, stddev, pad_policy='SAME', wd=None, trainable=True):
    """Set the convolution layer with RELU activation

    Args:
        name: layer name
        tensor: input tensor
        k_shape: shape of kernel tensor [in_height, in_width, in_channels, out_channels]
        stride: convolution stride
        stddev: standard deviation of weights intialization
        pad_policy: SAME/VALID, default SAME
        wd: weight decay factor, won't apply if wd=None
        trainable: Boolean, parameters are trainable if True

    Returns:
        Conv output

    """
    with tf.variable_scope(name) as scope:
        kernel = tf.Variable(tf.truncated_normal(k_shape, dtype=tf.float32, stddev=stddev), trainable=trainable, name=scope.name + '_kernels')
        if wd is not None:
            _add_weight_decay(kernel, wd)
        conv = tf.nn.conv2d(tensor, kernel, [1, stride, stride, 1], padding=pad_policy)
        biases = tf.Variable(tf.constant(0.0, shape=k_shape[-1:], dtype=tf.float32), trainable=trainable, name=scope.name + '_biases')
        bias = tf.nn.bias_add(conv, biases)
        return tf.nn.relu(bias, name=scope.name)


def _set_fc_layer(name, tensor, w_shape, stddev, wd=None, relu=True, drop=None, trainable=True, phase):
    """Set the fc layer with RELU activation

    Args:
        name: layer name
        tensor: input tensor
        w_shape: shape of fc weight tensor [in_dim, out_dim]
        wd: weight decay factor, won't apply if wd=None
        relu: Boolean, apply RELU to fc output if True
        drop: dropout rate, won't apply if drop=None
        phase: String. TRAIN or TEST.

    Returns:
        fc output

    """
    with tf.variable_scope(name) as scope:
        weights = tf.Variable(tf.truncated_normal(w_shape, dtype=tf.float32, stddev=stddev), trainable=trainble, name=scope.name + '_weights')
        if wd is not None:
            _add_weight_decay(weights, wd)
        biases = tf.Variable(tf.constant(0.0, shape=w_shape[-1:], dtype=tf.float32), trainable=trainable, name=scope.name + '_biases')
        fc = tf.add(tf.matmul(tensor, weights), biases, name=scope.name)
        if relu == True:
            fc = tf.nn.relu(tf.matmul(tensor, weights) + biases, name='relu')
        if (drop is not None) and (phase == 'TRAIN'):
            fc = tf.nn.drop(fc, drop, name='dropout')
        return fc


def inference(images, n_classes, phase):
    """
    """
