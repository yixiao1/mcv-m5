# Keras imports
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import (Convolution2D, MaxPooling2D, AveragePooling2D,
                                        ZeroPadding2D)

from keras.applications.resnet50 import ResNet50


# Paper: https://arxiv.org/abs/1512.03385

def build_resnet50(img_shape=(3, 224, 224), n_classes=1000, l2_reg=0.,
                load_imageNet=False, freeze_layers_from='base_model'):

    # Decide if load pretrained weights from imagenet
    if load_imageNet:
        weights = 'imagenet'
    else:
        weights = None

    # Get base model

    base_model = ResNet50(include_top=False, weights=weights, input_tensor=None, input_shape=img_shape)

    # Add final layers
    x = base_model.output
    x = Flatten(name="flatten")(x)
    x = Dense(n_classes, name='fc1000'.format(n_classes))(x)
    predictions = Activation("softmax", name="softmax")(x)

    # This is the model we will train
    model = Model(input=base_model.input, output=predictions)

    # Freeze some layers
    if freeze_layers_from is not None:
        if freeze_layers_from == 'base_model':
            print ('   Freezing base model layers')
            for layer in base_model.layers:
                layer.trainable = False
        else:
            for i, layer in enumerate(model.layers):
                print(i, layer.name)
            print ('   Freezing from layer 0 to ' + str(freeze_layers_from))
            for layer in model.layers[:freeze_layers_from]:
               layer.trainable = False
            for layer in model.layers[freeze_layers_from:]:
               layer.trainable = True
    return model
