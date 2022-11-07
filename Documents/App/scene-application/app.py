#using resnet50v2 : done
#using vgg16 : done
#using efficientnetb4 : done, this is the best 

# from crypt import methods
import json
from msilib.schema import Directory
import os
from os import path
import random
import numpy as np
from PIL import Image
from glob import glob
from keras.models import load_model
from keras.applications.efficientnet import preprocess_input
# from keras.applications.vgg16 import preprocess_input
# from keras.applications.resnet_v2 import preprocess_input
from pyexpat import model
from flask import Flask, render_template, request
app = Flask(__name__)
app.debug = True

class App : 
    def __init__(self,model) -> None: 
        print("Load the model")
        self.model=load_model(model)

    def prediction(self,img_path):
        new_image = self.load_image(img_path)
        print("Predicting using EfficientNetB4")
        pred = self.model.predict(np.asarray([new_image])) 
        result=np.array(pred)
        return result

    def load_image(self,img_path): 
        img = Image.open(img_path).resize((224,224))
        img_arr = np.asarray(img)
        print("Preprocessing using EfficientNetB4")
        img_arr = preprocess_input(img_arr)
        print(img_arr.shape)
        return img_arr

Cnn = App('./model/efficientnetb4new.h5')

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET","POST"])
def predict():
    if request.method=="POST":
        file=request.files['file']
        filename = file.filename
        file_path = os.path.join(r'C:/Users/User/Documents/App/scene-application/static/images', filename)
        file.save(file_path)
        print(filename)
        pred=Cnn.prediction(file_path)
        print("prediction",pred)
    return render_template("hasil.html",prediction=json.dumps(pred[0].tolist()),user_img="/static/images/"+filename)

# set FLASK_ENV=development 
# location : C:\Users\user\Documents\App\scene-application\static
# set VIRTUAL_ENV=C:\xampp\htdocs\scene-application\.venv di file script/activate.bat