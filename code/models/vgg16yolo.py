import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import LeakyReLU
from keras.layers import Concatenate
from keras.layers import Reshape
from layers.yolo_layers import YOLOConvolution2D,Reorg
from keras.layers import Convolution2D

def build_vgg16yolo(img_shape=(3, 416, 416), n_classes=80, n_priors=5,
               load_pretrained=False,freeze_layers_from='base_model'):

    # YOLO model is only implemented for TF backend
    assert(K.backend() == 'tensorflow')

    model = []

    # Get base model
    model = vgg16YOLO(input_shape=img_shape, num_classes=n_classes, num_priors=n_priors)
    base_model_layers = [layer.name for layer in model.layers[0:42]]

    if load_pretrained:
      # Rename last layer to not load pretrained weights
      model.layers[-1].name += '_new'
      model.load_weights('weights/yolo.hdf5',by_name=True)

    # Freeze some layers
    if freeze_layers_from is not None:
        if freeze_layers_from == 'base_model':
            for layer in model.layers:
                if layer.name in base_model_layers:
                    layer.trainable = False
        else:
            print ('   Freezing from layer 0 to ' + str(freeze_layers_from))
            for layer in model.layers[:freeze_layers_from]:
               layer.trainable = False
            for layer in model.layers[freeze_layers_from:]:
               layer.trainable = True

    return model

def vgg16YOLO(input_shape=(3,416,416),num_classes=80,num_priors=5):
    """YOLO (v2) architecture

    # Arguments
        input_shape: Shape of the input image
        num_classes: Number of classes (excluding background)

    # References
        https://arxiv.org/abs/1612.08242
        https://arxiv.org/abs/1506.02640
    """
    K.set_image_dim_ordering('th')

    net = {}
    # Block 1
    input_tensor = input_tensor = Input(shape=input_shape)
    img_size = (input_shape[1], input_shape[0])
    net['input'] = input_tensor
    net['conv1_1'] = Convolution2D(64, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv1_1')(net['input'])
    net['conv1_2'] = Convolution2D(64, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv1_2')(net['conv1_1'])
    net['pool1'] = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same',
                                name='pool1')(net['conv1_2'])
    # Block 2
    net['conv2_1'] = Convolution2D(128, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv2_1')(net['pool1'])
    net['conv2_2'] = Convolution2D(128, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv2_2')(net['conv2_1'])
    net['pool2'] = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same',
                                name='pool2')(net['conv2_2'])
    # Block 3
    net['conv3_1'] = Convolution2D(256, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv3_1')(net['pool2'])
    net['conv3_2'] = Convolution2D(256, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv3_2')(net['conv3_1'])
    net['conv3_3'] = Convolution2D(256, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv3_3')(net['conv3_2'])
    net['pool3'] = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same',
                                name='pool3')(net['conv3_3'])
    # Block 4
    net['conv4_1'] = Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv4_1')(net['pool3'])
    net['conv4_2'] = Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv4_2')(net['conv4_1'])
    net['conv4_3'] = Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv4_3')(net['conv4_2'])
    net['pool4'] = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same',
                                name='pool4')(net['conv4_3'])
    # Block 5
    net['conv5_1'] = Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv5_1')(net['pool4'])
    net['conv5_2'] = Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv5_2')(net['conv5_1'])
    net['conv5_3'] = Convolution2D(512, 3, 3,
                                   activation='relu',
                                   border_mode='same',
                                   name='conv5_3')(net['conv5_2'])
    # before are from VGG, below are from YOLO
    net['relu5_5'] = (LeakyReLU(alpha=0.1))(net['conv5_3'])
    net['pool5'] = (MaxPooling2D(pool_size=(2, 2),padding='valid'))(net['relu5_5'])

    net['conv6_1'] = (YOLOConvolution2D(1024, 3, 3, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001))(net['pool5'])
    net['relu6_1'] = (LeakyReLU(alpha=0.1))(net['conv6_1'])
    net['conv6_2'] = (YOLOConvolution2D(512, 1, 1, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001))(net['relu6_1'])
    net['relu6_2'] = (LeakyReLU(alpha=0.1))(net['conv6_2'])
    net['conv6_3'] = (YOLOConvolution2D(1024, 3, 3, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001))(net['relu6_2'])
    net['relu6_3'] = (LeakyReLU(alpha=0.1))(net['conv6_3'])
    net['conv6_4'] = (YOLOConvolution2D(512, 1, 1, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001))(net['relu6_3'])
    net['relu6_4'] = (LeakyReLU(alpha=0.1))(net['conv6_4'])
    net['conv6_5'] = (YOLOConvolution2D(1024, 3, 3, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001))(net['relu6_4'])
    net['relu6_5'] = (LeakyReLU(alpha=0.1))(net['conv6_5'])
    net['conv6_6'] = (YOLOConvolution2D(1024, 3, 3, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001))(net['relu6_5'])
    net['relu6_6'] = (LeakyReLU(alpha=0.1))(net['conv6_6'])
    net['conv6_7'] = (YOLOConvolution2D(1024, 3, 3, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001))(net['relu6_6'])
    net['relu6_7'] = (LeakyReLU(alpha=0.1))(net['conv6_7'])
    net['reorg7'] = (Reorg())(net['relu5_5'])
    net['merge7'] = (Concatenate(axis=1)([net['reorg7'], net['relu6_7']]))
    net['conv8'] = (YOLOConvolution2D(1024, 3, 3, border_mode='same',subsample=(1,1),
                                      epsilon=0.000001))(net['merge7'])
    net['relu8'] = (LeakyReLU(alpha=0.1))(net['conv8'])
    net['conv9'] = (Conv2D(num_priors*(4+num_classes+1), (1, 1), padding='same',
                                              strides=(1,1)))(net['relu8'])

    model = Model(net['input'], net['conv9'])
    return model


