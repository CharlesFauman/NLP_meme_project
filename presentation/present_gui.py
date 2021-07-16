import sys
sys.path.append("..")
import pickle

import os
# comment out line below if using gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd
import numpy as np
from keras.models import load_model

from cleaning import cleaning_functions
from analysis import analysis_functions

from os.path import basename
import glob

from tkinter import Tk, Frame, Canvas, Entry, Button, Text
from PIL import Image, ImageOps, ImageTk

cleaning_meta_dict = pickle.load(open( "../cleaning/meta_dict.p", "rb" ))
analysis_meta_dict = pickle.load(open( "../analysis/meta_dict.p", "rb" ))
model = load_model("../analysis/model.hdf5")

class Application(): #the main class that we call "Game"
	def __init__(self): #setting up the window for the game here
	
		self.root=Tk() #saying this window will use tkinter
		self.root.title("Meme Categorizer")
		self.RUN=True #creating a variable RUN. does nothing yet.
		self.root.protocol("WM_DELETE_WINDOW", self.end)
		
		self.frame = Frame(master= self.root)
		self.frame.grid(row = 3, column = 3)

		self.canvas = Canvas(master = self.frame, width=900, height=600, bg = "white", highlightthickness = 0) #actually creates a window and puts our frame on it
		self.canvas.grid(row = 1,column = 0,rowspan = 2, columnspan = 9) #makes the window called "canvas" complete

		self.text_entry = Text(master= self.frame, height = 10, width = 60, wrap='word')
		self.text_entry.grid(row = 0, column = 0)
		
		self.text_button = Button(master = self.frame, command = self.calculate_matches, height = 10, width = 30)
		self.text_button.grid(row = 0, column = 1)
		
		self.image_holder = ImageHolder(self)

		self.root.mainloop() #starts running the tkinter graphics loop
		
	def calculate_matches(self):
		sentence = self.text_entry.get("1.0", "end")
		cleaned = cleaning_functions.clean_sentence(cleaning_meta_dict, sentence)
		cleaned_as_df = pd.DataFrame(cleaned, columns = ["token", "pos", "sentiment"])
		sentence_matrix = analysis_functions.vectorize_sentence(analysis_meta_dict, cleaned_as_df)
		prediction = model.predict(np.array([sentence_matrix]))[0]
		
		#self.text_entry.delete("1.0", "end")
		#self.text_entry.insert("end", ' '.join([x[0] for x in cleaned]))
		
		scores = list()
		for index, val in enumerate(prediction):
			scores.append(tuple((analysis_meta_dict["meme_names"][index], val)))
			
		meme_predictions = sorted(scores, key=lambda tup: tup[1], reverse=True)
		self.image_holder.set_new(meme_predictions)
	
	def end(self):
		self.root.destroy() #closes the game window and ends the program

	def update(self):
		pass

class ImageHolder():
	def __init__(self, application):
		self.application = application
	
		self.images = dict()
		self.top_pred = list()
		for imagefile in glob.glob("../collection/templates/*.jpg"): #assuming jpg
			image = Image.open(imagefile)
			resized_image = ImageOps.fit(image, (300,300), Image.ANTIALIAS)
			self.images[basename(imagefile)[:-4]] = ImageTk.PhotoImage(resized_image)
			
	def set_new(self, meme_predictions):
		self.top_pred = [x[0] for x in meme_predictions[0:6]]
		self.paint()

	def paint(self):
		for index, meme in enumerate(self.top_pred):
			y = 0
			offset = 0
			if(index >= 3):
				y = 300
				offset = -900
			self.application.canvas.create_image(offset + index*300,y, image = self.images[meme], anchor = 'nw')


if __name__ == '__main__':
	application = Application() #start the application at Class Game()