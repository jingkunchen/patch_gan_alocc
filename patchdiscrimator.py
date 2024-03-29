'''
@Description: patch discriminator
@Author: Jingkun Chen
@Date: 2019-08-12 18:32:54
@LastEditTime: 2019-08-12 19:25:32
@LastEditors: Please set LastEditors
'''
from keras.layers import Flatten, Dense, Input, Reshape, merge, Lambda, concatenate
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
import keras.backend as K
import numpy as np

def PatchGanDiscriminator(output_img_dim, patch_dim, nb_patches):
    """
    Creates the generator according to the specs in the paper below.
    [https://arxiv.org/pdf/1611.07004v1.pdf][5. Appendix]

    PatchGAN only penalizes structure at the scale of patches. This
    discriminator tries to classify if each N x N patch in an
    image is real or fake. We run this discriminator convolutationally
    across the image, averaging all responses to provide
    the ultimate output of D.

    The discriminator has two parts. First part is the actual discriminator
    seconds part we make it a PatchGAN by running each image patch through the model
    and then we average the responses

    Discriminator does the following:
    1. Runs many pieces of the image through the network
    2. Calculates the cost for each patch
    3. Returns the avg of the costs as the output of the network

    :param patch_dim: ( width, height, channels) T
    :param nb_patches:
    :return:
    """
    # -------------------------------
    # DISCRIMINATOR
    # C64-C128-C256-C512-C512-C512 (for 256x256)
    # otherwise, it scales from 64
    # 1 layer block = Conv - BN - LeakyRelu
    # -------------------------------
    stride = 2
    bn_mode = 2
    axis = 1
    input_layer = Input(shape=patch_dim)

    # We have to build the discriminator dinamically because
    # the size of the disc patches is dynamic
    num_filters_start = 64
    nb_conv = int(np.floor(np.log(output_img_dim[1]) / np.log(2)))
    filters_list = [num_filters_start * min(8, (2 ** i)) for i in range(nb_conv)]

    # CONV 1
    # Do first conv bc it is different from the rest
    # paper skips batch norm for first layer
    
    disc_out = Conv2D(kernel_size=(4, 4), filters=64, strides=(2, 2), padding="same")(input_layer)
    disc_out = LeakyReLU(alpha=0.2)(disc_out)

    # CONV 2 - CONV N
    # do the rest of the convs based on the sizes from the filters
    for i, _ in enumerate(filters_list[1:]):
        name = 'disc_conv_{}'.format(i+2)

        disc_out = Conv2D( kernel_size=(4, 4), filters=128, strides=(2, 2), padding="same")(disc_out)
        # disc_out = BatchNormalization(name=name + '_bn', mode=bn_mode, axis=axis)(disc_out)
        disc_out = BatchNormalization(name=name + '_bn')(disc_out)
        disc_out = LeakyReLU(alpha=0.2)(disc_out)

    # ------------------------
    # BUILD PATCH GAN
    # this is where we evaluate the loss over each sublayer of the input
    # ------------------------
    patch_gan_discriminator = generate_patch_gan_loss(last_disc_conv_layer=disc_out,
                                                      patch_dim=patch_dim,
                                                      input_layer=input_layer,
                                                      nb_patches=nb_patches)
    return patch_gan_discriminator


def generate_patch_gan_loss(last_disc_conv_layer, patch_dim, input_layer, nb_patches):

    # generate a list of inputs for the different patches to the network
    list_input = [Input(shape=patch_dim, name="patch_gan_input_%s" % i) for i in range(nb_patches)]
    print("--------")
    print("nb_patches:",nb_patches)

    # get an activation
    x_flat = Flatten()(last_disc_conv_layer)
    x = Dense(2, activation='softmax', name="disc_dense")(x_flat)
    
    patch_gan = Model(input_layer, [x, x_flat])
    
    # generate individual losses for each patch
    x = [patch_gan(patch)[0] for patch in list_input]
    x_mbd = [patch_gan(patch)[1] for patch in list_input]
    
    print("x:",x)
    print("x_mbd:",x_mbd)
    print("--------")
    # merge layers if have multiple patches (aka perceptual loss)
    if len(x) > 1:
        x = concatenate(x)
        # x = merge(x, mode="concat", name="merged_features")
    else:
        x = x[0]

    # merge mbd if needed
    # mbd = mini batch discrimination
    # https://arxiv.org/pdf/1606.03498.pdf
    if len(x_mbd) > 1:
        x_mbd = concatenate(x_mbd)
        # x_mbd = merge(x_mbd, mode="concat", name="merged_feature_mbd")
    else:
        x_mbd = x_mbd[0]

    num_kernels = 100
    dim_per_kernel = 5

    M = Dense(num_kernels * dim_per_kernel, bias=False, activation=None)
    MBD = Lambda(minb_disc, output_shape=lambda_output)

    x_mbd = M(x_mbd)
    x_mbd = Reshape((num_kernels, dim_per_kernel))(x_mbd)
    x_mbd = MBD(x_mbd)
    x = concatenate([x, x_mbd])
    #x = merge([x, x_mbd], mode='concat')

    x_out = Dense(1, activation="softmax", name="disc_output")(x)
    print("list_input:",list_input)
    discriminator = Model(name='discriminator_nn', input=list_input, output=[x_out])
    return discriminator


def lambda_output(input_shape):
    # return input_shape[:2]
    return input_shape[:2]


def minb_disc(x):
    diffs = K.expand_dims(x, 3) - K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
    abs_diffs = K.sum(K.abs(diffs), 2)
    x = K.sum(K.exp(-abs_diffs), 2)

    return x
