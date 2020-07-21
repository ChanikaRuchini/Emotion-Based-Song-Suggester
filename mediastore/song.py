
from os import path
import os
import librosa
import csv
import pandas as pd
import numpy as np
#Keras
import keras
from keras.models import load_model
from sklearn.externals import joblib
import warnings
warnings.filterwarnings('ignore')


class song:
    def __init__(self, name, url,emotion):
        self.name = name
        self.url = url
        self.emotion = emotion
   
    def getEmotion(self):
        return self.emotion

    def getName(self):
        return self.name

    def getUrl(self):
        return self.url
