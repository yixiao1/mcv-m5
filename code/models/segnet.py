# Keras imports
from keras.models import *
from keras.layers import *
from keras import backend as K

IMAGE_ORDERING = 'channels_last' 

# Paper: https://arxiv.org/pdf/1511.00561.pdf
# Work inspired on: https://github.com/imlab-uiip/keras-segnet/blob/master/build_model.py

# Convolution + Batch normalization + ReLU (most SegNet encoders-decoders use this structure)
def conv_batchnorm_relu_block(input_x, n_filters, kernel_size, block_id, conv_id, relu=True):

    x = Convolution2D(n_filters, kernel_size, padding='same', data_format=IMAGE_ORDERING,
                      name='block{}_conv{}'.format(block_id, conv_id))(input_x)
    x = BatchNormalization(name='block{}_batchnorm{}'.format(block_id, conv_id))(x)
    if relu:
    	x = Activation('relu', name='block{}_relu{}'.format(block_id, conv_id))(x)

    return x

# TODO feature maps after unpooling should be sparse
#      idea: multiply unpooling output by indexMask
def MaxPooling2D_IndexMask(pool, stride, block_id, conv_id):

    pooled = MaxPooling2D(pool_size=pool, strides=stride, data_format=IMAGE_ORDERING,
                          name='block{}_conv{}'.format(block_id, conv_id))(input_x)
    upsampled = UpSampling2D(size=pool, data_format=IMAGE_ORDERING )(pooled)
    indexMask = K.tf.equal(inputs, upsampled)

    return pooled, indexMask


def build_segnet(img_shape=(416, 608, 3), nclasses=8, l2_reg=0.,
                 init='glorot_uniform', path_weights=None,
                 load_pretrained=False, freeze_layers_from=None, basic=False):

	img_input = Input(shape=img_shape)
	x = img_input

	# SegNet-basic --> 4 encoders and 4 decoders
	# SegNet-VGG ----> 13 encoders (from VGGNet) and 13 decoders

	if basic:

	    kernel = 7

	    #### ENCODER
	    x = conv_batchnorm_relu_block(x, 64, kernel, 1, 1)
	    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool1', data_format=IMAGE_ORDERING)(x)
	    x = conv_batchnorm_relu_block(x, 128, kernel, 2, 1)
	    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool2', data_format=IMAGE_ORDERING)(x)
	    x = conv_batchnorm_relu_block(x, 256, kernel, 3, 1)
	    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool3', data_format=IMAGE_ORDERING)(x)
	    x = conv_batchnorm_relu_block(x, 512, kernel, 4, 1)
	    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool4', data_format=IMAGE_ORDERING)(x)


	    #### DECODER
            x = UpSampling2D(size=(2,2), data_format=IMAGE_ORDERING)(x)
    	    x = ZeroPadding2D((1,1), data_format=IMAGE_ORDERING)(x)
            x = conv_batchnorm_relu_block(x, 512, kernel, 5, 1)
	
            x = UpSampling2D(size=(2,2), data_format=IMAGE_ORDERING)(x)
    	    x = ZeroPadding2D((1,1), data_format=IMAGE_ORDERING)(x)
            x = conv_batchnorm_relu_block(x, 256, kernel, 6, 1)

            x = UpSampling2D(size=(2,2), data_format=IMAGE_ORDERING)(x)
    	    x = ZeroPadding2D((1,1), data_format=IMAGE_ORDERING)(x)
            x = conv_batchnorm_relu_block(x, 128, kernel, 7, 1)

            x = UpSampling2D(size=(2,2), data_format=IMAGE_ORDERING)(x)
    	    x = ZeroPadding2D((1,1), data_format=IMAGE_ORDERING)(x)
            x = conv_batchnorm_relu_block(x, 64, kernel, 8, 1)

	    o = Conv2D(filters=nclasses, kernel_size=1, padding='same', data_format=IMAGE_ORDERING)(x)

        else:

	    kernel = 3

	    #### ENCODER

	    # Block 1
	    x = conv_batchnorm_relu_block(x, 64, kernel, 1, 1)
	    x = conv_batchnorm_relu_block(x, 64, kernel, 1, 2)
	    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool1', data_format=IMAGE_ORDERING)(x)

	    # Block2
	    x = conv_batchnorm_relu_block(x, 128, kernel, 2, 1)
	    x = conv_batchnorm_relu_block(x, 128, kernel, 2, 2)
	    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool2', data_format=IMAGE_ORDERING)(x)

	    # Block 3
	    x = conv_batchnorm_relu_block(x, 256, kernel, 3, 1)
	    x = conv_batchnorm_relu_block(x, 256, kernel, 3, 2)
	    x = conv_batchnorm_relu_block(x, 256, kernel, 3, 3)
	    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool3', data_format=IMAGE_ORDERING)(x)

	    # Block 4
	    x = conv_batchnorm_relu_block(x, 512, kernel, 4, 1)
	    x = conv_batchnorm_relu_block(x, 512, kernel, 4, 2)
	    x = conv_batchnorm_relu_block(x, 512, kernel, 4, 3)
	    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool4', data_format=IMAGE_ORDERING)(x)

	    # Block 5
	    x = conv_batchnorm_relu_block(x, 512, kernel, 5, 1)
	    x = conv_batchnorm_relu_block(x, 512, kernel, 5, 2)
	    x = conv_batchnorm_relu_block(x, 512, kernel, 5, 3)
	    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING)(x)

	
	    #### DECODER

	    # Block 6
	    x = UpSampling2D(size=(2,2), data_format=IMAGE_ORDERING)(x)
    	    x = ZeroPadding2D((1,1), data_format=IMAGE_ORDERING)(x)
            x = conv_batchnorm_relu_block(x, 512, kernel, 6, 1)
	    x = conv_batchnorm_relu_block(x, 512, kernel, 6, 2)
	    x = conv_batchnorm_relu_block(x, 512, kernel, 6, 3)

	    # Block 7
	    x = UpSampling2D(size=(2,2), data_format=IMAGE_ORDERING)(x)
    	    x = ZeroPadding2D((1,1), data_format=IMAGE_ORDERING)(x)
            x = conv_batchnorm_relu_block(x, 512, kernel, 7, 1)
	    x = conv_batchnorm_relu_block(x, 512, kernel, 7, 2)
	    x = conv_batchnorm_relu_block(x, 256, kernel, 7, 3)

	    # Block 8
	    x = UpSampling2D(size=(2,2), data_format=IMAGE_ORDERING)(x)
    	    x = ZeroPadding2D((1,1), data_format=IMAGE_ORDERING)(x)
            x = conv_batchnorm_relu_block(x, 256, kernel, 8, 1)
	    x = conv_batchnorm_relu_block(x, 256, kernel, 8, 2)
	    x = conv_batchnorm_relu_block(x, 128, kernel, 8, 3)

	    # Block 9
	    x = UpSampling2D(size=(2,2), data_format=IMAGE_ORDERING)(x)
    	    x = ZeroPadding2D((1,1), data_format=IMAGE_ORDERING)(x)
            x = conv_batchnorm_relu_block(x, 128, kernel, 9, 1)
	    x = conv_batchnorm_relu_block(x, 64, kernel, 9, 2)

	    # Block 10
	    x = UpSampling2D(size=(2,2), data_format=IMAGE_ORDERING)(x)
    	    x = ZeroPadding2D((1,1), data_format=IMAGE_ORDERING)(x)
            x = conv_batchnorm_relu_block(x, 64, kernel, 10, 1)
	    o = Conv2D(filters=nclasses, kernel_size=3, padding='same', data_format=IMAGE_ORDERING)(x)
           
	# Ensure that output has the same size HxW as input and apply softmax to compute pixel prob.
        o_shape = Model(img_input, o).output_shape
        o = Cropping2D(((o_shape[1] - img_shape[0]) / 2, (o_shape[2] - img_shape[1]) / 2), name='crop')(o)

        """
        o_shape = Model(img_input, o).output_shape
        outputHeight = o_shape[1]
        outputWidth = o_shape[2]
        o = (Reshape((-1, outputHeight * outputWidth)))(o)
        o = (Permute((2, 1)))(o)
        """

        # Reshape to vector
        curlayer_output_shape = Model(inputs=img_input, outputs=o).output_shape
        if K.image_dim_ordering() == 'tf':
            outputHeight = curlayer_output_shape[1]
            outputWidth = curlayer_output_shape[2]
        else:
            outputHeight = curlayer_output_shape[2]
            outputWidth = curlayer_output_shape[3]
        o = Reshape(target_shape=(outputHeight * outputWidth, nclasses))(o)

        o = Activation('softmax')(o)
        model = Model(img_input, o)
        #model.outputWidth = outputWidth
        #model.outputHeight = outputHeight

        # Freeze some layers
        if freeze_layers_from is not None:
            freeze_layers(model, freeze_layers_from)

        return model


# Freeze layers for finetunning
def freeze_layers(model, freeze_layers_from):
    # Freeze the VGG part only
    if freeze_layers_from == 'base_model':
        print ('   Freezing base model layers')
        freeze_layers_from = 23

    # Show layers (Debug pruposes)
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
    print ('   Freezing from layer 0 to ' + str(freeze_layers_from))

    # Freeze layers
    for layer in model.layers[:freeze_layers_from]:
        layer.trainable = False
    for layer in model.layers[freeze_layers_from:]:
        layer.trainable = True


if __name__ == '__main__':
    input_shape = [224, 224, 3]
    print (' > Building')
    model = build_fcn8(input_shape, 11, 0.)
    print (' > Compiling')
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
model.summary()# Keras imports
from keras.models import *
from keras.layers import *
from keras import backend as K

IMAGE_ORDERING = 'channels_last' 

# Paper: https://arxiv.org/pdf/1511.00561.pdf
# Work inspired on: https://github.com/imlab-uiip/keras-segnet/blob/master/build_model.py

# Convolution + Batch normalization + ReLU (most SegNet encoders-decoders use this structure)
def conv_batchnorm_relu_block(input_x, n_filters, kernel_size, block_id, conv_id, relu=True):

    x = Convolution2D(n_filters, kernel_size, padding='same', data_format=IMAGE_ORDERING,
                      name='block{}_conv{}'.format(block_id, conv_id))(input_x)
    x = BatchNormalization(name='block{}_batchnorm{}'.format(block_id, conv_id))(x)
    if relu:
    	x = Activation('relu', name='block{}_relu{}'.format(block_id, conv_id))(x)

    return x

# TODO feature maps after unpooling should be sparse
#      idea: multiply unpooling output by indexMask
def MaxPooling2D_IndexMask(pool, stride, block_id, conv_id):

    pooled = MaxPooling2D(pool_size=pool, strides=stride, data_format=IMAGE_ORDERING,
                          name='block{}_conv{}'.format(block_id, conv_id))(input_x)
    upsampled = UpSampling2D(size=pool, data_format=IMAGE_ORDERING )(pooled)
    indexMask = K.tf.equal(inputs, upsampled)

    return pooled, indexMask


def build_segnet(img_shape=(416, 608, 3), nclasses=8, l2_reg=0.,
                 init='glorot_uniform', path_weights=None,
                 load_pretrained=False, freeze_layers_from=None, basic=False):

	img_input = Input(shape=img_shape)
	x = img_input

	# SegNet-basic --> 4 encoders and 4 decoders
	# SegNet-VGG ----> 13 encoders (from VGGNet) and 13 decoders

	if basic:

	    kernel = 7

	    #### ENCODER
	    x = conv_batchnorm_relu_block(x, 64, kernel, 1, 1)
	    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool1', data_format=IMAGE_ORDERING)(x)
	    x = conv_batchnorm_relu_block(x, 128, kernel, 2, 1)
	    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool2', data_format=IMAGE_ORDERING)(x)
	    x = conv_batchnorm_relu_block(x, 256, kernel, 3, 1)
	    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool3', data_format=IMAGE_ORDERING)(x)
	    x = conv_batchnorm_relu_block(x, 512, kernel, 4, 1)
	    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool4', data_format=IMAGE_ORDERING)(x)


	    #### DECODER
            x = UpSampling2D(size=(2,2), data_format=IMAGE_ORDERING)(x)
    	    x = ZeroPadding2D((1,1), data_format=IMAGE_ORDERING)(x)
            x = conv_batchnorm_relu_block(x, 512, kernel, 5, 1)
	
            x = UpSampling2D(size=(2,2), data_format=IMAGE_ORDERING)(x)
    	    x = ZeroPadding2D((1,1), data_format=IMAGE_ORDERING)(x)
            x = conv_batchnorm_relu_block(x, 256, kernel, 6, 1)

            x = UpSampling2D(size=(2,2), data_format=IMAGE_ORDERING)(x)
    	    x = ZeroPadding2D((1,1), data_format=IMAGE_ORDERING)(x)
            x = conv_batchnorm_relu_block(x, 128, kernel, 7, 1)

            x = UpSampling2D(size=(2,2), data_format=IMAGE_ORDERING)(x)
    	    x = ZeroPadding2D((1,1), data_format=IMAGE_ORDERING)(x)
            x = conv_batchnorm_relu_block(x, 64, kernel, 8, 1)

	    o = Conv2D(filters=nclasses, kernel_size=1, padding='same', data_format=IMAGE_ORDERING)(x)

        else:

	    kernel = 3

	    #### ENCODER

	    # Block 1
	    x = conv_batchnorm_relu_block(x, 64, kernel, 1, 1)
	    x = conv_batchnorm_relu_block(x, 64, kernel, 1, 2)
	    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool1', data_format=IMAGE_ORDERING)(x)

	    # Block2
	    x = conv_batchnorm_relu_block(x, 128, kernel, 2, 1)
	    x = conv_batchnorm_relu_block(x, 128, kernel, 2, 2)
	    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool2', data_format=IMAGE_ORDERING)(x)

	    # Block 3
	    x = conv_batchnorm_relu_block(x, 256, kernel, 3, 1)
	    x = conv_batchnorm_relu_block(x, 256, kernel, 3, 2)
	    x = conv_batchnorm_relu_block(x, 256, kernel, 3, 3)
	    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool3', data_format=IMAGE_ORDERING)(x)

	    # Block 4
	    x = conv_batchnorm_relu_block(x, 512, kernel, 4, 1)
	    x = conv_batchnorm_relu_block(x, 512, kernel, 4, 2)
	    x = conv_batchnorm_relu_block(x, 512, kernel, 4, 3)
	    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool4', data_format=IMAGE_ORDERING)(x)

	    # Block 5
	    x = conv_batchnorm_relu_block(x, 512, kernel, 5, 1)
	    x = conv_batchnorm_relu_block(x, 512, kernel, 5, 2)
	    x = conv_batchnorm_relu_block(x, 512, kernel, 5, 3)
	    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING)(x)

	
	    #### DECODER

	    # Block 6
	    x = UpSampling2D(size=(2,2), data_format=IMAGE_ORDERING)(x)
    	    x = ZeroPadding2D((1,1), data_format=IMAGE_ORDERING)(x)
            x = conv_batchnorm_relu_block(x, 512, kernel, 6, 1)
	    x = conv_batchnorm_relu_block(x, 512, kernel, 6, 2)
	    x = conv_batchnorm_relu_block(x, 512, kernel, 6, 3)

	    # Block 7
	    x = UpSampling2D(size=(2,2), data_format=IMAGE_ORDERING)(x)
    	    x = ZeroPadding2D((1,1), data_format=IMAGE_ORDERING)(x)
            x = conv_batchnorm_relu_block(x, 512, kernel, 7, 1)
	    x = conv_batchnorm_relu_block(x, 512, kernel, 7, 2)
	    x = conv_batchnorm_relu_block(x, 256, kernel, 7, 3)

	    # Block 8
	    x = UpSampling2D(size=(2,2), data_format=IMAGE_ORDERING)(x)
    	    x = ZeroPadding2D((1,1), data_format=IMAGE_ORDERING)(x)
            x = conv_batchnorm_relu_block(x, 256, kernel, 8, 1)
	    x = conv_batchnorm_relu_block(x, 256, kernel, 8, 2)
	    x = conv_batchnorm_relu_block(x, 128, kernel, 8, 3)

	    # Block 9
	    x = UpSampling2D(size=(2,2), data_format=IMAGE_ORDERING)(x)
    	    x = ZeroPadding2D((1,1), data_format=IMAGE_ORDERING)(x)
            x = conv_batchnorm_relu_block(x, 128, kernel, 9, 1)
	    x = conv_batchnorm_relu_block(x, 64, kernel, 9, 2)

	    # Block 10
	    x = UpSampling2D(size=(2,2), data_format=IMAGE_ORDERING)(x)
    	    x = ZeroPadding2D((1,1), data_format=IMAGE_ORDERING)(x)
            x = conv_batchnorm_relu_block(x, 64, kernel, 10, 1)
	    o = Conv2D(filters=nclasses, kernel_size=3, padding='same', data_format=IMAGE_ORDERING)(x)
           
	# Ensure that output has the same size HxW as input and apply softmax to compute pixel prob.
        o_shape = Model(img_input, o).output_shape
        o = Cropping2D(((o_shape[1] - img_shape[0]) / 2, (o_shape[2] - img_shape[1]) / 2), name='crop')(o)

        """
        o_shape = Model(img_input, o).output_shape
        outputHeight = o_shape[1]
        outputWidth = o_shape[2]
        o = (Reshape((-1, outputHeight * outputWidth)))(o)
        o = (Permute((2, 1)))(o)
        """

        # Reshape to vector
        curlayer_output_shape = Model(inputs=img_input, outputs=o).output_shape
        if K.image_dim_ordering() == 'tf':
            outputHeight = curlayer_output_shape[1]
            outputWidth = curlayer_output_shape[2]
        else:
            outputHeight = curlayer_output_shape[2]
            outputWidth = curlayer_output_shape[3]
        o = Reshape(target_shape=(outputHeight * outputWidth, nclasses))(o)

        o = Activation('softmax')(o)
        model = Model(img_input, o)
        #model.outputWidth = outputWidth
        #model.outputHeight = outputHeight

        # Freeze some layers
        if freeze_layers_from is not None:
            freeze_layers(model, freeze_layers_from)

        return model


# Freeze layers for finetunning
def freeze_layers(model, freeze_layers_from):
    # Freeze the VGG part only
    if freeze_layers_from == 'base_model':
        print ('   Freezing base model layers')
        freeze_layers_from = 23

    # Show layers (Debug pruposes)
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
    print ('   Freezing from layer 0 to ' + str(freeze_layers_from))

    # Freeze layers
    for layer in model.layers[:freeze_layers_from]:
        layer.trainable = False
    for layer in model.layers[freeze_layers_from:]:
        layer.trainable = True


if __name__ == '__main__':
    input_shape = [224, 224, 3]
    print (' > Building')
    model = build_fcn8(input_shape, 11, 0.)
    print (' > Compiling')
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    model.summary()
