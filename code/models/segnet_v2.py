# Keras imports
from keras.models import *
from keras.layers import *
from keras import backend as K

IMAGE_ORDERING = 'channels_last' 

# Paper: https://arxiv.org/pdf/1511.00561.pdf

# Convolution + Batch normalization + ReLU (most SegNet encoders-decoders use this structure)
def conv_batchnorm_relu_block(input_x, n_filters, kernel_size, block_id, conv_id, relu=True):

    x = Convolution2D(n_filters, kernel_size, padding='same', data_format=IMAGE_ORDERING,
                      name='block{}_conv{}'.format(block_id, conv_id))(input_x)
    x = BatchNormalization(name='block{}_batchnorm{}'.format(block_id, conv_id))(x)
    if relu:
    	x = Activation('relu', name='block{}_relu{}'.format(block_id, conv_id))(x)

    return x


# Unpooling + Convolution (used in multiple decoders of SegNet)
def unpool_conv(input_x, n_filters, kernel_size, block_id, conv_id, relu=True):

    x = UpSampling2D(size=(2,2), data_format=IMAGE_ORDERING)(input_x)
    x = ZeroPadding2D((1,1), data_format=IMAGE_ORDERING)(x)
    x = conv_batchnorm_relu_block(x, n_filters, kernel_size, block_id, conv_id, relu)

    return x


def build_segnet(img_shape=(416, 608, 3), nclasses=8, l2_reg=0.,
               init='glorot_uniform', path_weights=None,
               load_pretrained=False, freeze_layers_from=None, basic=False):


	# https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5
	img_input = Input(shape=img_shape)

	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING )(img_input)
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING )(x)
	f1 = x
	# Block 2
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING )(x)
	f2 = x

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=IMAGE_ORDERING )(x)
	f3 = x

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING )(x)
	f4 = x

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING )(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING )(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING )(x)
	f5 = x

        if load_pretrained:
          print('   Loading VGG pre-trained weights...')
	  x = Flatten(name='flatten_x')(x)
	  x = Dense(4096, activation='relu', name='fc1_x')(x)
	  x = Dense(4096, activation='relu', name='fc2_x')(x)
	  x = Dense( 1000 , activation='softmax', name='predictions_x')(x)

	  vgg  = Model(  img_input , x  )
	  vgg.load_weights('weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5', by_name=True)

	x = f5
	kernel = 3	

        #### DECODER
	
        # Block 6
        x = unpool_conv(x, 512, kernel, 6, 1)
        x = conv_batchnorm_relu_block(x, 512, kernel, 6, 2)
        x = conv_batchnorm_relu_block(x, 512, kernel, 6, 3)

        # Block 7
        x = unpool_conv(x, 512, kernel, 7, 1)
        x = conv_batchnorm_relu_block(x, 512, kernel, 7, 2)
        x = conv_batchnorm_relu_block(x, 256, kernel, 7, 3)

        # Block 8
        x = unpool_conv(x, 256, kernel, 8, 1)
        x = conv_batchnorm_relu_block(x, 256, kernel, 8, 2)
        x = conv_batchnorm_relu_block(x, 128, kernel, 8, 3)

        # Block 9
        x = unpool_conv(x, 128, kernel, 9, 1)
        x = conv_batchnorm_relu_block(x, 64, kernel, 9, 2)

        # Block 10
        x = unpool_conv(x, 64, kernel, 10, 1)
        o = Conv2D(filters=nclasses, kernel_size=3, padding='same', name='block10_conv2', data_format=IMAGE_ORDERING)(x)

	# Ensure that output has the same size HxW as input and apply softmax to compute pixel prob.
	o_shape = Model(img_input , o ).output_shape
        o = Cropping2D(((o_shape[1]-img_shape[0])/2,(o_shape[2]-img_shape[1])/2), name='crop')(o)
	o_shape = Model(img_input , o ).output_shape
	outputHeight = o_shape[1]
	outputWidth = o_shape[2]
	o = (Reshape((  -1  , outputHeight*outputWidth   )))(o)
	o = (Permute((2, 1)))(o)
	o = (Activation('softmax'))(o)
	model = Model( img_input , o )
	model.outputWidth = outputWidth
	model.outputHeight = outputHeight

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
