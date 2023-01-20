import os 
import pickle
import numpy as np

def get_crop_prediction(data:list):
    models = ["farmap_decisiontree.pkl", "farmap_naivebiass.pkl", "farmap_svm.pkl", "farmap_logisticregression.pkl", "farmap_randomforest.pkl"]
    prediction = []
    for model in models:
        if not model in os.listdir("./../models"):
            print("model payena")
    else:
        for model_ in os.listdir("./../models"):
            name = "./../models/"+model_
            prediction.append(pickle.load(open(name, 'rb')).predict(np.array([data]))[0])
    return most_frequent(prediction)

def most_frequent(List):
    return max(set(List), key = List.count)
