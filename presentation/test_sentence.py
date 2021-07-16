
import sys
sys.path.append("..")
import pickle

import pandas as pd
import numpy as np
from keras.models import load_model

from cleaning import cleaning_functions
from analysis import analysis_functions

if __name__ == '__main__':
	
	cleaning_meta_dict = pickle.load(open( "../cleaning/meta_dict.p", "rb" ))
	analysis_meta_dict = pickle.load(open( "../analysis/meta_dict.p", "rb" ))
	model = load_model("../analysis/model.hdf5")
	
	while(True):
		sentence = input("sentence: ")
		cleaned = cleaning_functions.clean_sentence(cleaning_meta_dict, sentence)
		cleaned_as_df = pd.DataFrame(cleaned, columns = ["token", "pos", "sentiment"])
		sentence_matrix = analysis_functions.vectorize_sentence(analysis_meta_dict, cleaned_as_df)
		prediction = model.predict(np.array([sentence_matrix]))[0]
		
		scores = list()
		for index, val in enumerate(prediction):
			scores.append(tuple((analysis_meta_dict["meme_names"][index], val)))
			
		for meme in reversed(sorted(scores, key=lambda tup: tup[1], reverse=True)):
			print(str(meme[0]) + ": " + str(meme[1]))
			
		print("prediction: " + analysis_meta_dict["meme_names"][np.argmax(prediction)])
	