from flask import Flask, request, render_template

app = Flask(__name__, static_url_path='')

@app.route("/")
def home():
    return "Hello, World!"

@app.route("/model/", methods = ['GET', 'POST'])
def generateImage():
    file = request.args.get("query", default = "test", type = str)
    fileName = file + ".png"
    
    # Load model

    # Generate image and save it in the static folder

    return render_template("index.html", fileName = fileName)
    
if __name__ == "__main__":
    app.run(debug=True)
