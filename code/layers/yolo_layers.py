from __future__ import absolute_import
import functools

from keras import backend as K
from keras import activations, initializers, regularizers, constraints
from keras.engine import Layer, InputSpec
from keras.utils import conv_utils

import tensorflow as tf

class YOLOConvolution2D(Layer):
    """ This class implements Convolution operator with Batch Normalization
    in the same way as in Darknet framework.
    In Darknet batch normalization is done before adding the biases in a conv layer.
    In Keras batch normalization is done in a separate layer.
    Thus, if we want to use YOLO pre-trained weights it is not possible to use a
    combination of Convolution2D+BatchNormalization standard keras layers.
    This code replicates the code of keras Convolution2D and BatchNormalization
    but in a single layer. We simplify the BatchNormalization part and only
    implement the mode used in Darknet (feature-wise normalization).

    Convolution operator for filtering windows of two-dimensional inputs.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures.

    # Arguments
        nb_filter: Number of convolution filters to use.
        nb_row: Number of rows in the convolution kernel.
        nb_col: Number of columns in the convolution kernel.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)), or alternatively,
            Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass
            a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of numpy arrays to set as initial weights.
        border_mode: 'valid', 'same' or 'full'. ('full' requires the Theano backend.)
        subsample: tuple of length 2. Factor by which to subsample output.
            Also called strides elsewhere.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode is it at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "tf".
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.

    # Output shape
        4D tensor with shape:
        `(samples, nb_filter, new_rows, new_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, nb_filter)` if dim_ordering='tf'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self, nb_filter, nb_row, nb_col,
                 init='glorot_uniform', activation=None, weights=None,
                 border_mode='valid', subsample=(1, 1), dim_ordering='default',
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, epsilon=1e-3, momentum=0.99,
                 beta_init='zero', gamma_init='one',
                 gamma_regularizer=None, beta_regularizer=None, **kwargs):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        if border_mode not in {'valid', 'same', 'full'}:
            raise ValueError('Invalid border mode for Convolution2D:', border_mode)
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.init = initializers.get(init)
        self.activation = activations.get(activation)
        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        if dim_ordering not in {'tf', 'th'}:
            raise ValueError('dim_ordering must be in {tf, th}.')
        self.dim_ordering = dim_ordering

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.center = True
        self.scale  = True
        self.input_spec = [InputSpec(ndim=4)]
        self.initial_weights = weights
        # for BatchNormalization
        self.supports_masking = True
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.initial_weights = weights
        self.uses_learning_phase = True
        self.data_format = 'channels_first'
        super(YOLOConvolution2D, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.dim_ordering == 'th':
            stack_size = input_shape[1]
            self.W_shape = (self.nb_filter, stack_size, self.nb_row, self.nb_col)
            self.data_format = 'channels_first'
        elif self.dim_ordering == 'tf':
            stack_size = input_shape[3]
            self.W_shape = (self.nb_row, self.nb_col, stack_size, self.nb_filter)
            self.data_format = 'channels_last'
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        input_dim = input_shape[channel_axis]
        self.kernel_size = (self.nb_row, self.nb_col)
        self.W_shape = self.kernel_size + (input_dim, self.nb_filter)
        self.W = self.add_weight(shape=self.W_shape,
                                 initializer=initializers.get('glorot_uniform'),
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((self.nb_filter,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        # for BatchNormalization
        self.axis = 1
        shape = (self.nb_filter,)
        self.gamma = self.add_weight(shape=shape,
                                     initializer=self.gamma_init,
                                     regularizer=self.gamma_regularizer,
                                     name='{}_gamma'.format(self.name))
        self.beta = self.add_weight(shape=shape,
                                    initializer=self.beta_init,
                                    regularizer=self.beta_regularizer,
                                    name='{}_beta'.format(self.name))
        self.moving_mean = self.add_weight(shape=shape, initializer='zero',
                                            name='{}_moving_mean'.format(self.name),
                                            trainable=False)
        self.moving_variance = self.add_weight(shape=shape, initializer='one',
                                           name='{}_moving_variance'.format(self.name),
                                           trainable=False)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True

    def compute_output_shape(self, input_shape):
        if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)

        rows = conv_utils.conv_output_length(rows, self.nb_row,
                                  self.border_mode, self.subsample[0])
        cols = conv_utils.conv_output_length(cols, self.nb_col,
                                  self.border_mode, self.subsample[1])

        if self.dim_ordering == 'th':
            return (input_shape[0], self.nb_filter, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, self.nb_filter)

    def call(self, x, training=True):
        output = K.conv2d(x, self.W, 
                          strides=self.subsample,
                          padding=self.border_mode,
                          data_format=self.data_format)
        

        # batch normalization

        input_shape = K.int_shape(output)
        # Prepare broadcasting shape.
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        # Determines whether broadcasting is needed.
        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

        def normalize_inference():
            if needs_broadcasting:
                # In this case we must explicitly broadcast all parameters.
                broadcast_moving_mean = K.reshape(self.moving_mean,
                                                  broadcast_shape)
                broadcast_moving_variance = K.reshape(self.moving_variance,
                                                      broadcast_shape)
                if self.center:
                    broadcast_beta = K.reshape(self.beta, broadcast_shape)
                else:
                    broadcast_beta = None
                if self.scale:
                    broadcast_gamma = K.reshape(self.gamma,
                                                broadcast_shape)
                else:
                    broadcast_gamma = None
                return K.batch_normalization(
                    output,
                    broadcast_moving_mean,
                    broadcast_moving_variance,
                    broadcast_beta,
                    broadcast_gamma,
                    epsilon=self.epsilon)
            else:
                return K.batch_normalization(
                    output,
                    self.moving_mean,
                    self.moving_variance,
                    self.beta,
                    self.gamma,
                    epsilon=self.epsilon)

        # If the learning phase is *static* and set to inference:
        if training in {0, False}:
            output_normed = normalize_inference()

        else:

            # If the learning is either dynamic, or set to training:
            normed_training, mean, variance = K.normalize_batch_in_training(
                output, self.gamma, self.beta, reduction_axes,
                epsilon=self.epsilon)
    
            self.add_update([K.moving_average_update(self.moving_mean,
                                                     mean,
                                                     self.momentum),
                             K.moving_average_update(self.moving_variance,
                                                     variance,
                                                     self.momentum)],
                            output)
    
            # Pick the normalized form corresponding to the training phase.
            output_normed =  K.in_train_phase(normed_training,
                                    normalize_inference,
                                    training=training)


        if self.bias:
            if self.dim_ordering == 'th':
                output_normed += K.reshape(self.b, (1, self.nb_filter, 1, 1))
            elif self.dim_ordering == 'tf':
                output_normed += K.reshape(self.b, (1, 1, 1, self.nb_filter))
            else:
                raise ValueError('Invalid dim_ordering:', self.dim_ordering)
        output = self.activation(output_normed)
        return output

    def get_config(self):
        config = {'nb_filter': self.nb_filter,
                  'nb_row': self.nb_row,
                  'nb_col': self.nb_col,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample': self.subsample,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'epsilon': self.epsilon,
                  'gamma_regularizer': self.gamma_regularizer.get_config() if self.gamma_regularizer else None,
                  'beta_regularizer': self.beta_regularizer.get_config() if self.beta_regularizer else None,
                  'momentum': self.momentum}
        base_config = super(YOLOConvolution2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Works only in TF
class Reorg(Layer):
    """ This class implements REORG layer as in Darknet framework.
    When we bring finer grained features in from earlier in the network, the reorg layer
    makes these features match the feature map size at the later layer.
    E.g. if the end feature map is 13x13 and the feature map from earlier is 26x26x512, 
    the reorg layer maps the 26x26x512 feature map onto a 13x13x2048 feature map so that
    it can be concatenated with the feature maps at 13x13 resolution.
    The Darknet reorg layer does not only perform a simple reshape, but instead slices
    the data in its own way.
    """
    def __init__(self, **kwargs):
        super(Reorg, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Reorg, self).build(input_shape)

    def call(self, data, mask=None):
        tmp1 = tf.strided_slice(data,[0,0,0,0],[1024,tf.to_int32(data.get_shape()[1]),tf.to_int32(data.get_shape()[2]),tf.to_int32(data.get_shape()[3])],[1,1,2,2])

        tmp2 = tf.strided_slice(data,[0,0,0,1],[1024,tf.to_int32(data.get_shape()[1]),tf.to_int32(data.get_shape()[2]),tf.to_int32(data.get_shape()[3])],[1,1,2,2])

        tmp3 = tf.strided_slice(data,[0,0,1,0],[1024,tf.to_int32(data.get_shape()[1]),tf.to_int32(data.get_shape()[2]),tf.to_int32(data.get_shape()[3])],[1,1,2,2])

        tmp4 = tf.strided_slice(data,[0,0,1,1],[1024,tf.to_int32(data.get_shape()[1]),tf.to_int32(data.get_shape()[2]),tf.to_int32(data.get_shape()[3])],[1,1,2,2])

        if int(tf.__version__[0]) < 1:
            return tf.concat(1,[tmp1, tmp2, tmp3, tmp4])
        else:
            return tf.concat([tmp1, tmp2, tmp3, tmp4],1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]*4, input_shape[2]/2, input_shape[3]/2)

