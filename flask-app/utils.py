from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import (Dense, 
                                     BatchNormalization, 
                                     LeakyReLU, 
                                     Reshape, 
                                     Conv2DTranspose,
                                     Conv2D,
                                     Dropout,
                                     Flatten)
from matplotlib import image

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa

IMAGE_SIZE = [256, 256]

def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image

def read_tfrecord(example):
    tfrecord_format = {
        "image": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image'])
    return image

def load_dataset(filenames):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(read_tfrecord)
    return dataset

def down_sample(filters, size, apply_instancenorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    layer = keras.Sequential()
    layer.add(layers.Conv2D(filters, size, strides=2, padding='same', kernel_initializer=initializer, use_bias=False))

    if apply_instancenorm:
        layer.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    layer.add(layers.LeakyReLU())

    return layer

def up_sample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    layer = keras.Sequential()
    layer.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer,use_bias=False))
    layer.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    if apply_dropout:
        layer.add(layers.Dropout(0.5))

    layer.add(layers.ReLU())

    return layer

def Generator():
    inputs = layers.Input(shape=[256,256,3])
    down_stack = [
        down_sample(64, 4, apply_instancenorm=False),# (size, 128, 128, 64)
        down_sample(128, 4),                         # (size, 64, 64, 128)
        down_sample(256, 4),                         # (size, 32, 32, 256)
        down_sample(512, 4),                         # (size, 16, 16, 512)
        down_sample(512, 4),                         # (size, 8, 8, 512)
        down_sample(512, 4),                         # (size, 4, 4, 512)
        down_sample(512, 4),                         # (size, 2, 2, 512)
        down_sample(512, 4),                         # (size, 1, 1, 512)
    ]

    up_stack = [
        up_sample(512, 4, apply_dropout=True),       # (size, 2, 2, 1024)
        up_sample(512, 4, apply_dropout=True),       # (size, 4, 4, 1024)
        up_sample(512, 4, apply_dropout=True),       # (size, 8, 8, 1024)
        up_sample(512, 4),                           # (size, 16, 16, 1024)
        up_sample(256, 4),                           # (size, 32, 32, 512)
        up_sample(128, 4),                           # (size, 64, 64, 256)
        up_sample(64, 4),                            # (size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(3, 4, strides=2, padding='same', kernel_initializer=initializer, activation='tanh') 
    # (size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = layers.Concatenate()([x, skip])

    x = last(x)

    return keras.Model(inputs=inputs, outputs=x)

def make_generator_model():
    model = tf.keras.Sequential()
    model.add(Dense(7*7*512, use_bias=False, input_shape=(256,))) #originally 7*7*7, 256*256
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((7,7,512)))#was 7,7,512
#     assert model.output_shape == (None, 7, 7, 512) # Note: None is the batch size
    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)) #originally 128
#     assert model.output_shape == (None, 7, 7, 128)
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    
    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)) #originally 64
#     assert model.output_shape == (None, 14, 14, 64)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))#originally 3
#     print(model.output_shape)
#     assert model.output_shape == (None, 28, 28, 3)
    return model


def randomNoiseModel(file):
    gen = make_generator_model()
    gen.load_weights('models/generator4.h5')

    noise = tf.random.normal([3, 256])*127.5+100
    plt.imsave('static/input.png', noise.numpy())#.clip(0, 1))

    modelOutput = gen(noise)[0]
    plt.imsave(f'static/{file}', modelOutput.numpy().clip(0, 1))


def monetModel(fileIn, fileOut):

    print(fileIn)
    monet_generator = Generator()
    monet_generator.load_weights('models/vangogh_generator.h5')
    for img in load_dataset(f'static/{fileIn}').batch(1):
        inputImage = img

    inputSaveImage = inputImage[0].numpy().clip(0,1)
    plt.imsave('static/input.png', inputSaveImage)

    #inputImage = image.imread(f'static/test3.jpg')

    #monet_ds = load_dataset(MONET_FILENAMES).batch(1)
    #modelInput = tf.reshape(tf.convert_to_tensor(image.imread(f'static/test3.jpg')), [1, 256, 256, 3])
    #modelInput = tf.random.normal([1, 256, 256, 3])#*127.5+100
    

    #modelOutput = monet_generator(modelInput)[0]
    modelOutput = monet_generator(inputImage, training=False)[0].numpy().clip(0,1)
    plt.imsave(f'static/{fileOut}', modelOutput)