import imp
from operator import mod
from fastcore.meta import method
import PIL
from flask import Flask, render_template, jsonify, request, url_for
from fastai.learner import load_learner
from fastai.vision.all import *
import torchvision.transforms as T
from flask_cors import CORS, cross_origin
import logging
logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)
@app.route('/')
def homepage():
    return render_template('index.html')
model = load_learner(fname='models/export.pkl')

def predict_single(img_file):
    '''function takes image and returns prediction'''
    #logging.debug('This function to take image and return prediction')
    img_pil = PILImage.create(img_file)
    prediction = model.predict(img_pil)
    probs_list = prediction
    #print("Inside predict_single")
    #logging.debug('Before probs_list')
    return probs_list
@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':  
        #logging.debug('Before my_prediction')
        my_prediction = predict_single(request.files['image'])        
        #logging.debug('Before final_pred')
        final_pred = str(my_prediction[0])
        #logging.debug('After final_pred')
    return render_template('results.html', prediction=final_pred, comment='asd')
if __name__ == '__main__':
    app.run(debug=True)