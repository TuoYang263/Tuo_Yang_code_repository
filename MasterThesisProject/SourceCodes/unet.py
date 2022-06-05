from keras.models import Model
# a series common-used network layers has been defined inside the core, including full-connected layers and activtation layers
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout,\
    BatchNormalization, Permute, add, multiply, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers.core import Lambda
import keras.backend as K


def up_and_concate(down_layer, layer, data_format='channels_first'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    if data_format == "channels_first":
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])
    return concate


def attention_up_and_concate(down_layer, layer, data_format='channels_first'):
    if data_format == 'channels_first':
        in_channel = down_layer.get_shape().as_list()[1]
    else:
        in_channel = down_layer.get_shape().as_list()[3]

    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)

    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel//4, data_format=data_format)

    if data_format == 'channels_first':
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))

    concate = my_concat([up, layer])
    return concate


def attention_block_2d(x, g, inter_channel, data_format='channels_first'):
    # theta_x(?,g_height,g_width,inter_channel)

    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)

    # phi_g(?,g_height,g_width,inter_channel)

    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)

    # f(?,g_height,g_width,inter_channel)

    f = Activation('relu')(add([theta_x, phi_g]))

    # psi_f(?,g_height,g_width,1)

    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)

    rate = Activation('sigmoid')(psi_f)

    # rate(?,x_height,x_width)

    # att_x(?,x_height,x_width,x_channel)

    att_x = multiply([x, rate])

    return att_x


def res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],
              padding='same', data_format='channels_first'):
    if data_format == 'channels_first':
        input_n_filters = input_layer.get_shape().as_list()[1]
    else:
        input_n_filters = input_layer.get_shape().as_list()[3]

    layer = input_layer
    for i in range(2):
        layer = Conv2D(out_n_filters // 4, [1, 1], strides=stride, padding=padding, data_format=data_format)(layer)
        if batch_normalization:
            layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Conv2D(out_n_filters // 4, kernel_size, strides=stride, padding=padding, data_format=data_format)(layer)
        layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(layer)

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer
    out_layer = add([layer, skip_layer])
    return out_layer


# Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net)
def rec_res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],

                  padding='same', data_format='channels_first'):
    if data_format == 'channels_first':
        input_n_filters = input_layer.get_shape().as_list()[1]
    else:
        input_n_filters = input_layer.get_shape().as_list()[3]

    if out_n_filters != input_n_filters:
        skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
            input_layer)
    else:
        skip_layer = input_layer

    layer = skip_layer
    for j in range(2):

        for i in range(2):
            if i == 0:

                layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                    layer)
                if batch_normalization:
                    layer1 = BatchNormalization()(layer1)
                layer1 = Activation('relu')(layer1)
            layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                add([layer1, layer]))
            if batch_normalization:
                layer1 = BatchNormalization()(layer1)
            layer1 = Activation('relu')(layer1)
        layer = layer1

    out_layer = add([layer, skip_layer])
    return out_layer
    
# Define the neural network
def get_unet(n_ch,patch_height,patch_width,n_label):
    inputs = Input(shape=(n_ch,patch_height,patch_width))
    # data_format: string, one of "channels_first" or "channels_last", which indicates channel positions of images
    # take the 128 by 128 RGB image as the example, "channels_first" should organize data as (3,128,128), while
    # "channels_last" should organize data as (128,128,3)
    # default values of this parameter is the value set inside ~/.keras/keras.json
    # if it has not been set up yet, it's channels_last
    # Block1
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)
    #
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    #
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv3)
    conv3 = BatchNormalization()(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2, up1], axis=1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up1)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    conv4 = BatchNormalization()(conv4)

    #
    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1, up2], axis=1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up2)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    conv5 = BatchNormalization()(conv5)

    #
    # the function of 1 cross 1 convolution 1. realize interaction and information melting
    # between different channels 2. proceeding dimensions increasing and decreasing of convolutional channels
    conv6 = Conv2D(n_label, (1, 1), activation='relu', padding='same', data_format='channels_first')(conv5)
    # currently, the output shape is (batchsize,2,patch_height*patch_width)
    conv6 = core.Reshape((n_label, patch_height * patch_width))(
        conv6)
    # currently, the shape of output is (Npatch,patch_height*patch_width,2),  which the dimension of output is
    # (Npatch,2304,2)
    conv6 = core.Permute((n_label, 1))(conv6)
    conv7 = core.Activation('softmax')(conv6)
    model = Model(inputs=inputs, outputs=conv7)
    return model


# Define the neural network
def get_full_unet(n_ch,patch_height,patch_width,n_label):
    inputs = Input(shape=(n_ch,patch_height,patch_width))
    # data_format: string, one of "channels_first" or "channels_last", which indicates channel positions of images
    # take the 128 by 128 RGB image as the example, "channels_first" should organize data as (3,128,128), while
    # "channels_last" should organize data as (128,128,3)
    # default values of this parameter is the value set inside ~/.keras/keras.json
    # if it has not been set up yet, it's channels_last
    # Block1
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(inputs)
    #conv1 = Dropout(0.2)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    # Block2
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool1)
    #conv2 = Dropout(0.2)(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    # Block3
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(pool2)
    #conv3 = Dropout(0.2)(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    # Block4
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool3)
    # conv3 = Dropout(0.2)(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)

    #Block5
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool4)
    # conv3 = Dropout(0.2)(conv3)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv5)
    conv5 = BatchNormalization()(conv5)
    pool5 = MaxPooling2D((2, 2))(conv5)

    conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same', data_format='channels_first')(pool5)
    # conv3 = Dropout(0.2)(conv3)
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv6)
    conv6 = BatchNormalization()(conv6)

    up1 = UpSampling2D(size=(2, 2))(conv6)
    up1 = concatenate([conv5,up1],axis=1)
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same',data_format='channels_first')(up1)
    #conv4 = Dropout(0.2)(conv4)
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv7)
    conv7 = BatchNormalization()(conv7)

    #
    up2 = UpSampling2D(size=(2, 2))(conv7)
    up2 = concatenate([conv4,up2], axis=1)
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_first')(up2)
    #conv5 = Dropout(0.2)(conv5)
    conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same',data_format='channels_first')(conv8)
    conv8 = BatchNormalization()(conv8)

    up3 = UpSampling2D(size=(2, 2))(conv8)
    up3 = concatenate([conv3, up3], axis=1)
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(up3)
    # conv5 = Dropout(0.2)(conv5)
    conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv9)
    conv9 = BatchNormalization()(conv9)

    up4 = UpSampling2D(size=(2, 2))(conv9)
    up4 = concatenate([conv2, up4], axis=1)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(up4)
    # conv5 = Dropout(0.2)(conv5)
    conv10 = BatchNormalization()(conv10)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_first')(conv10)
    conv10 = BatchNormalization()(conv10)

    up5 = UpSampling2D(size=(2, 2))(conv10)
    up5 = concatenate([conv1, up5], axis=1)
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_first')(up5)

    #
    #the function of 1 cross 1 convolution 1. realize interaction and information melting
    #between different channels 2. proceeding dimensions increasing and decreasing of convolutional channels
    conv11 = Conv2D(n_label, (1, 1), activation='relu',padding='same',data_format='channels_first')(conv11)
    # currently, the output shape is (batchsize,2,patch_height*patch_width)
    conv11 = core.Reshape((n_label,patch_height*patch_width))(conv11)
    # currently, the shape of output is (Npatch,patch_height*patch_width,2),  which the dimension of output is
    # (Npatch,2304,2)
    conv11 = core.Permute((n_label,1))(conv11)
    ############
    conv12 = core.Activation('softmax')(conv11)
    model = Model(inputs=inputs, outputs=conv12)
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.3, nesterov=False)
    return model

#Attention U-Net
def att_unet(n_channel, img_h, img_w, n_label, data_format='channels_first'):
    inputs = Input((n_channel, img_w, img_h))
    x = inputs
    depth = 4
    features = 64
    skips = []
    for i in range(depth):
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format='channels_first')(x)
        features = features * 2

    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    for i in reversed(range(depth)):
        features = features // 2
        x = attention_up_and_concate(x, skips[i], data_format=data_format)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    conv6 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
    # currently, the output shape is (batchsize,2,patch_height*patch_width)
    conv6 = core.Reshape((n_label,img_h*img_w))(conv6)
    # currently, the shape of output is (Npatch,patch_height*patch_width,2),  which the dimension of output is
    # (Npatch,2304,2)
    conv6 = core.Permute((n_label, 1))(conv6)
    conv7 = core.Activation('softmax')(conv6)
    model = Model(inputs=inputs, outputs=conv7)
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=Adam(lr=1e-5), loss=[focal_loss()], metrics=['accuracy', dice_coef])
    return model


########################################################################################################
# Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net)
def r2_unet(n_channel, img_h, img_w, n_label, data_format='channels_first'):
    inputs = Input((n_channel, img_w, img_h))
    x = inputs
    depth = 4
    features = 64
    skips = []
    for i in range(depth):
        x = rec_res_block(x, features, data_format=data_format)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)

        features = features * 2

    x = rec_res_block(x, features, data_format=data_format)

    for i in reversed(range(depth)):
        features = features // 2
        x = up_and_concate(x, skips[i], data_format=data_format)
        x = rec_res_block(x, features, data_format=data_format)

    conv6 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
    # currently, the output shape is (batchsize,2,patch_height*patch_width)
    conv6 = core.Reshape((n_label, img_h * img_w))(conv6)
    # currently, the shape of output is (Npatch,patch_height*patch_width,2),  which the dimension of output is
    # (Npatch,2304,2)
    conv6 = core.Permute((n_label, 1))(conv6)
    conv7 = core.Activation('softmax')(conv6)
    model = Model(inputs=inputs, outputs=conv7)
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=Adam(lr=1e-6), loss=[dice_coef_loss], metrics=['accuracy', dice_coef])
    return model


########################################################################################################
#Attention R2U-Net
def att_r2_unet(n_channel, img_h, img_w, n_label, data_format='channels_first'):
    inputs = Input((n_channel, img_w, img_h))
    x = inputs
    depth = 4
    features = 64
    skips = []
    for i in range(depth):
        x = rec_res_block(x, features, data_format=data_format)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format=data_format)(x)

        features = features * 2

    x = rec_res_block(x, features, data_format=data_format)

    for i in reversed(range(depth)):
        features = features // 2
        x = attention_up_and_concate(x, skips[i], data_format=data_format)
        x = rec_res_block(x, features, data_format=data_format)

    conv6 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
    # currently, the output shape is (batchsize,2,patch_height*patch_width)
    conv6 = core.Reshape((n_label, img_h * img_w))(conv6)
    # currently, the shape of output is (Npatch,patch_height*patch_width,2),  which the dimension of output is
    # (Npatch,2304,2)
    conv6 = core.Permute((n_label, 1))(conv6)
    conv7 = core.Activation('softmax')(conv6)
    model = Model(inputs=inputs, outputs=conv7)
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=Adam(lr=1e-6), loss=[dice_coef_loss], metrics=['accuracy', dice_coef])
    return model



