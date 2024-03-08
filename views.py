from flask import Flask, render_template, request, url_for
from PIL import Image
import numpy as np
import cv2
from detector import Detector

app = Flask(__name__)



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        #saving image
        image = request.files['photo']
        file_name = image.filename
        image.save(f'./static/uploads/{file_name}')

        detector = Detector(f'./static/uploads/{file_name}', './Haar Cascade/haarcascade_frontalface_alt.xml')

        #loding image for preprocessing
        detector.preprocess_image()
    
        #detecting with classifier
        detector.detect()

        #drawing boxes
        img = detector.draw_boxes()
        img.save(f'./static/img/{file_name}')
        src = f'img/{file_name}'
        return render_template('index.html', src=src)
    return render_template('index.html', file_name=None)