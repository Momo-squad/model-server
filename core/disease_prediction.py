import numpy as np 
import pickle
import os
from tensorflow import keras
model = pickle.load(open(os.path.realpath(os.path.join(os.getcwd(), "models/dis_classify.pkl")),'rb'))
print(model)

def get_disease_prediction(image):
    np_image = np.array(image, dtype=np.float16) / 225.0
    np_image = np.expand_dims(np_image,0)
    resutlt = np.argmax(model.predict(np_image))
    return resutlt
