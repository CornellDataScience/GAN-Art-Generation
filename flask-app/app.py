from flask import Flask, request, render_template
from utils import *

app = Flask(__name__, static_url_path='')

@app.route("/")
def home():
    return app.send_static_file('index.html')

@app.route("/model") # model?model_type=model_num&?input_file=file
def generateImage():
    modelType = request.args.get("model_type", default = 0, type = int)
    fileInName = request.args.get("input_file")

    fileOutName = 'test.png'

    if (modelType == 0):
        randomNoiseModel(fileOutName)
    elif (modelType == 1):
        monetModel(fileInName, fileOutName)

    return {'output': fileOutName}
    
if __name__ == "__main__":
    app.run()
