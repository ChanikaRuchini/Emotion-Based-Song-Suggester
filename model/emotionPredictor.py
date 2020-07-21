import os
from os import path
import librosa
import pandas as pd
import numpy as np
import csv
import keras
import tkinter as tk
from tkinter import *
from mutagen.mp3 import MP3
from pygame import mixer
from pydub import AudioSegment

import warnings
warnings.filterwarnings('ignore')
from keras.models import load_model
from sklearn.externals import joblib
# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


#predict the emotion of the song using a pretrained model
def detectEmotion(path):
	songname = f'{path}'
	print (songname)
	base=os.path.basename(path)
	print (base)
	name=os.path.splitext(base)[0]
	print (name)
	header = 'chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
	for i in range(1, 21):
    		header += f' mfcc{i}'
	header = header.split()

	file = open('songdata.csv', 'w', newline='')
	with file:
    		writer = csv.writer(file)
    		writer.writerow(header)
	#feature extraction using Librosa library
	y, sr = librosa.load(songname, mono=True, duration=30)
	chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
	spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
	spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
	rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
	zcr = librosa.feature.zero_crossing_rate(y)
	rmse = librosa.feature.rmse(y=y)
	mfcc = librosa.feature.mfcc(y=y, sr=sr)
	to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}' 
	print(to_append)
	for e in mfcc:
		to_append += f' {np.mean(e)}'
	file = open('songdata.csv', 'a', newline='')
	with file:
		writer = csv.writer(file)
		writer.writerow(to_append.split())

	model = load_model('model/new_model.h5')
	data_from_csv = pd.read_csv("songdata.csv")
	data = data_from_csv
	print (data)
	data.shape
	scaler = joblib.load("model/scaler.save") 
	X = scaler.transform(np.array(data.iloc[:,:], dtype = float))
	print(X)
	ynew = model.predict(np.array(X))
	#ynew = model.predict(np.array(data))
	print (ynew)
	if np.argmax(ynew)==0:
		print ("Calm")
		return "calm"
	elif np.argmax(ynew)==1:
		print ("Happy")
		return "happy"
	elif np.argmax(ynew)==2:
		print ("Sad")
		return "sad"
	#return np.argmax(ynew)

#detectEmotion("/home/ruchini/Music/happy/Zedd, Katy Perry - 365 (Official).mp3")