from flask import Flask, request, render_template
import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras.layers import (Dense, 
                                     BatchNormalization, 
                                     LeakyReLU, 
                                     Reshape, 
                                     Conv2DTranspose,
                                     Conv2D,
                                     Dropout,
                                     Flatten)
#from matplotlib import image
import matplotlib.pyplot as plt




app = Flask(__name__, static_url_path='')

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

@app.route("/")
def home():
    return app.send_static_file('index.html')

@app.route("/model/") # model?query
def generateImage():
    # file = request.args.get("query", default = "test", type = str)
    # fileOutName = file + "_output.png"
    fileOutName = 'test.png'

    gen3 = make_generator_model()
    gen3.load_weights('generator2.h5')

    noise = tf.random.normal([3, 256])*127.5+100
    #plt.imshow(gen3(noise)[0,:,:,:])
    #plt.savefig('static/{0}'.format(fileOutName))

    modelOutput = gen3(noise)[0]
    plt.imsave(f'static/{fileOutName}', modelOutput.numpy().clip(0, 1))
 
    return {'output': fileOutName}
    
if __name__ == "__main__":
    app.run()
