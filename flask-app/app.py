from flask import Flask, request, render_template
from tensorflow import keras
from matplotlib import image
from matplotlib import pyplot as plt

app = Flask(__name__, static_url_path='')

@app.route("/")
def home():
    return "Hello, World!"

@app.route("/model/", methods = ['GET', 'POST'])
def generateImage():
    file = request.args.get("query", default = "test", type = str)
    fileOutName = file + "_output.png"
    
    # Load model
    modelPath = ''
    model = keras.models.load_model(modelPath)

    # Generate image and save it in the static folder
    output = model(image.imread(file), training=False)

    fig = plt.figure(figsize=(4,4))
    for i in range(output.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(output[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('static/{0}'.format(fileOutName))

    return render_template("index.html", fileName = fileOutName)
    
if __name__ == "__main__":
    app.run(debug=True)
