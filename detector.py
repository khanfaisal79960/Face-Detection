from PIL import Image
import numpy as np
import cv2

class Detector:
    def __init__(self, image_path, classifier_path) -> None:
        self.image_path = image_path
        self.classifier_path = classifier_path
        self.image_arr = None
        self.grey = None
        self.face = None

    def preprocess_image(self):
        image = Image.open(self.image_path)
        image = image.resize((350, 250))
        self.image_arr = np.array(image)
        self.grey = cv2.cvtColor(self.image_arr, cv2.COLOR_BGR2GRAY)

    def detect(self):
        classifier = cv2.CascadeClassifier(self.classifier_path)
        self.face = classifier.detectMultiScale(self.grey, 1.1, 1)

    def draw_boxes(self):
        for (x, y, w, h) in self.face:
            img = cv2.rectangle(self.image_arr, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img = Image.fromarray(img)
        return img

    



    