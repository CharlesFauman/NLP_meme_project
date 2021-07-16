import os
# comment out line below if using gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd
import numpy as np

import pickle

import tensorflow as tf

from gensim.models import Word2Vec

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.models import load_model
from keras.utils.np_utils import to_categorical
from keras.metrics import top_k_categorical_accuracy


from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.externals import joblib
from sklearn import tree

import analysis_functions

import os.path


if __name__ == '__main__':
	print("preparing to train")
	
	train_data = None
	train_y = None
	test_data = None
	test_y = None
	
	meta_dict = dict()
	
	if(os.path.exists("meta_dict.p") and os.path.exists('data/train_data.npy') and os.path.exists('data/train_y.npy') and os.path.exists('data/test_data.npy') and os.path.exists('data/test_y.npy')):
		meta_dict = pickle.load(open( "meta_dict.p", "rb" ))
		
		train_data = np.load("data/train_data.npy")
		train_y = np.load("data/train_y.npy")
		test_data = np.load("data/test_data.npy")
		test_y = np.load("data/test_y.npy")
	else:
		print("initial formatting data")
		data_df = pd.read_csv("../cleaning/cleaned_memes.tsv", sep='\t')
		data_meme_classes = dict(tuple(data_df.groupby('meme')))
		train_data_dict = dict()
		test_data_dict = dict()
		for meme_name, memes in data_meme_classes.items():
			validation_split = int(np.floor(len(np.unique(memes['meme_id']))*.8))
			ans = dict(tuple(memes.groupby(lambda index: int(memes.loc[index]['meme_id'] > validation_split))))
			train_data_dict[meme_name] = ans[0]
			test_data_dict[meme_name] = ans[1]
			
		train_sentences = analysis_functions.data_dict_to_sents(train_data_dict)
		
		meta_dict["memes"] = analysis_functions.categorical_dict_from_list(np.unique(data_df['meme']))
		meta_dict["meme_names"] = np.unique(data_df['meme'])
		meta_dict["pos"] = analysis_functions.categorical_dict_from_list(np.unique(data_df['pos']))
		meta_dict["sentiment"] = analysis_functions.categorical_dict_from_list(np.unique(data_df['sentiment']))
		meta_dict["embeddings"] = Word2Vec(train_sentences, size=50, min_count=2)
		meta_dict["sentence_size"] = 50
		pickle.dump(meta_dict, open( "meta_dict.p", "wb" ) )
		
		print("reformatting data")
		train_data, train_y = analysis_functions.vectorize_meme_data(meta_dict, train_data_dict)
		test_data, test_y = analysis_functions.vectorize_meme_data(meta_dict, test_data_dict)
		
		np.save("data/train_data", train_data)
		np.save("data/train_y", train_y)
		np.save("data/test_data", test_data)
		np.save("data/test_y", test_y)
		
	print("training")
	model = analysis_functions.create_model(meta_dict)
	
	if(os.path.exists("model.hdf5")): model = load_model("model.hdf5")
	
	tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
	tbCallBack.set_model(model)
	model.fit(train_data, train_y, epochs = 30, batch_size = 200, validation_split = 0.0, callbacks = [tbCallBack])
	
	print("saving model")
	model.save("model.hdf5")
	
	print("predictions:")
	print("train: " + str(model.evaluate(train_data, train_y)))
	print("top 2: " + str(analysis_functions.eval_top_k(model, train_data, train_y, 2)))
	print("top 3: " + str(analysis_functions.eval_top_k(model, train_data, train_y, 3)))
	print("top 5: " + str(analysis_functions.eval_top_k(model, train_data, train_y, 5)))
	print("top 10: " + str(analysis_functions.eval_top_k(model, train_data, train_y, 10)))
	
	
	print("test: " + str(model.evaluate(test_data, test_y)))
	print("top 2: " + str(analysis_functions.eval_top_k(model, test_data, test_y, 2)))
	print("top 3: " + str(analysis_functions.eval_top_k(model, test_data, test_y, 3)))
	print("top 5: " + str(analysis_functions.eval_top_k(model, test_data, test_y, 5)))
	print("top 10: " + str(analysis_functions.eval_top_k(model, test_data, test_y, 10)))
	
	
	