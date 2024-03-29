from __future__ import print_function
from keras.layers import Input, Dense, Dropout, Activation, Concatenate, BatchNormalization, Reshape, Lambda, Flatten
from keras.models import Model, Sequential
from keras.layers import Conv2D, GlobalAveragePooling2D, AveragePooling2D, LSTM, TimeDistributed
from keras.regularizers import l2
#from utils import *
from keras.optimizers import *



def Densenet_LSTM():



    sequential_image= Input(shape=(17,120,200,1))
    #sequential_coordinate= Input (shape= (17,120,200,1))


    base_network = DenseNet(dense_blocks=5, dense_layers=-1, growth_rate=8, dropout_rate=0.2,
                           bottleneck=True, compression=1.0, weight_decay=1e-4, depth=40,batch_norm=False)


    encoded_patches= TimeDistributed(base_network)(sequential_image)

    LSTM_layer= LSTM(256, return_sequences=True)(encoded_patches)
    Dense1= Dense(128, activation ='linear')(LSTM_layer)
    outputs= Dense (8, activation='linear')(Dense1)
    model= Model(inputs= sequential_image,outputs= outputs)

    return model



def DenseNet(dense_blocks=3, dense_layers=-1, growth_rate=12, dropout_rate=None,
             bottleneck=False, compression=1.0, weight_decay=1e-4, depth=40, batch_norm=True):
    """
    Creating a DenseNet

    Arguments:
        input_shape  : shape of the input images. E.g. (28,28,1) for MNIST
        dense_blocks : amount of dense blocks that will be created (default: 3)
        dense_layers : number of layers in each dense block. You can also use a list for numbers of layers [2,4,3]
                       or define only 2 to add 2 layers at all dense blocks. -1 means that dense_layers will be calculated
                       by the given depth (default: -1)
        growth_rate  : number of filters to add per dense block (default: 12)
        dropout_rate : defines the dropout rate that is accomplished after each conv layer (except the first one).
                       In the paper the authors recommend a dropout of 0.2 (default: None)
        bottleneck   : (True / False) if true it will be added in convolution block (default: False)
        compression  : reduce the number of feature-maps at transition layer. In the paper the authors recomment a compression
                       of 0.5 (default: 1.0 - will have no compression effect)
        weight_decay : weight decay of L2 regularization on weights (default: 1e-4)
        depth        : number or layers (default: 40)

    Returns:
        Model        : A Keras model instance
    """

    if compression <= 0.0 or compression > 1.0:
        raise Exception(
            'Compression have to be a value between 0.0 and 1.0. If you set compression to 1.0 it will be turn off.')

    if type(dense_layers) is list:
        if len(dense_layers) != dense_blocks:
            raise AssertionError('Number of dense blocks have to be same length to specified layers')
    elif dense_layers == -1:
        if bottleneck:
            dense_layers = (depth - (dense_blocks + 1)) / dense_blocks // 2
        else:
            dense_layers = (depth - (dense_blocks + 1)) // dense_blocks
        dense_layers = [int(dense_layers) for _ in range(dense_blocks)]
    else:
        dense_layers = [int(dense_layers) for _ in range(dense_blocks)]

    Input_image = Input(shape=(120, 200, 1))
    #Input_coordinate = Input(shape=(4,))
    nb_channels = growth_rate * 2

    print('Creating DenseNet')
    print('#############################################')
    print('Dense blocks: %s' % dense_blocks)
    print('Layers per dense block: %s' % dense_layers)
    print('#############################################')

    # Initial convolution layer
    x = Conv2D(nb_channels, (3, 3), padding='same', strides=(1, 1),
               use_bias=False, kernel_regularizer=l2(weight_decay))(Input_image)

    # Building dense blocks
    for block in range(dense_blocks):

        # Add dense block
        x, nb_channels = dense_block(x, dense_layers[block], nb_channels, growth_rate, dropout_rate, bottleneck,
                                     weight_decay,batch_norm)

        if block < dense_blocks - 1:  # if it's not the last dense block
            # Add transition_block
            x = transition_layer(x, nb_channels, dropout_rate, compression, weight_decay,batch_norm)
            nb_channels = int(nb_channels * compression)
    if batch_norm:
        x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    flattenlayer = GlobalAveragePooling2D()(x)

    #concatinated_layer= Concatenate()([flattenlayer,Input_coordinate])

    model = Model(inputs=Input_image, outputs=flattenlayer)
    

    return model


def dense_block(x, nb_layers, nb_channels, growth_rate, dropout_rate=None, bottleneck=False, weight_decay=1e-4, batch_norm= True):
    """
    Creates a dense block and concatenates inputs
    """

    x_list = [x]
    for i in range(nb_layers):
        cb = convolution_block(x, growth_rate, dropout_rate, bottleneck, weight_decay,batch_norm)
        x_list.append(cb)
        x = Concatenate(axis=-1)(x_list)
        nb_channels += growth_rate
    return x, nb_channels


def convolution_block(x, nb_channels, dropout_rate=None, bottleneck=False, weight_decay=1e-4,batch_norm=True):
    """
    Creates a convolution block consisting of BN-ReLU-Conv.
    Optional: bottleneck, dropout
    """

    # Bottleneck
    if bottleneck:
        bottleneckWidth = 4
        if batch_norm:
            x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)
        x = Conv2D(nb_channels * bottleneckWidth, (1, 1), use_bias=False, kernel_regularizer=l2(weight_decay))(x)
        # Dropout
        if dropout_rate:
            x = Dropout(dropout_rate)(x)

    # Standard (BN-ReLU-Conv)
    if batch_norm:
        x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_channels, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay))(x)

    # Dropout
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_layer(x, nb_channels, dropout_rate=None, compression=1.0, weight_decay=1e-4,batch_norm=True):
    """
    Creates a transition layer between dense blocks as transition, which do convolution and pooling.
    Works as downsampling.

    """
    if batch_norm:
        x = BatchNormalization(gamma_regularizer=l2(weight_decay), beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(int(nb_channels * compression), (1, 1), padding='same',
               use_bias=False, kernel_regularizer=l2(weight_decay))(x)

    # Adding dropout
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    return x
