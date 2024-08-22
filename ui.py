from flask import Flask, render_template, request
import cv2
import numpy as np
#from tensorflow.keras.models import load_model
#from nbimporter import NotebookLoader
from DentalDiseaseClassificationpy import graph_cnn,labels
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # get the uploaded image file from the form
        file = request.files['image']
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.resize(image, (32,32))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,32,32,3)
        img = np.asarray(im2arr)
        img = img.astype('float32')
        img = img/255
        preds = graph_cnn.predict(img)
        predict = np.argmax(preds)
        score = np.amax(preds)
        label = labels[predict]
        return render_template('result.html', label=label, score=score)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
