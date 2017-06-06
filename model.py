#!usr/bin/python3
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from os.path import expanduser
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, LSTM
from keras.models import Model
from keras.layers.merge import concatenate
import os
import sys

class quoraModel:
	def __init__(self):
		'''
			The train and test datasets contain an ID
			and question 1/ question 2 which the model is supposed to differentiate
			whether the questions are similar or different
			
			self.RANDOM_STATE = the random_state to initialize the split for reproducible results
			self.MAX_WORDS = gets the maximum number of words for numeric tokenizing
			self.MAX_SEQUENCE_LENGTH = maximum length of a sequence

			self.MAX_NB_WORDS = How many words we need to embedd
			self.EMBEDDING_DIM = number of dimensions in the embedded matrix
		'''
		self.loadData()
		self.RANDOM_STATE = 1
		self.MAX_WORDS = 140000
		self.MAX_SEQUENCE_LENGTH = 238
		self.GLOVE_DIR = '/home/ryan/Documents/wordEmbedding'
		self.MAX_NB_WORDS = 140000
		self.EMBEDDING_DIM = 100

	def loadData(self):
		'''Loads the train/test data from a csv

			self.train = Also contains a label for the ground truth
			-0 if the questions are not the same 1 if they are

		'''
		self.train = pd.read_csv('/home/ryan/Documents/quora/train.csv')
		self.test = pd.read_csv('/home/ryan/Documents/quora/test.csv')


	def loadWordEmbedding(self):
		'''Loads the pre-trained wordEmbedding algorithm

			word embedding is basically finding words that are similar to the ones you are using
			
			Here are two common word embedding algorithms:

			From stanford:
			GloVe= Global Vectors for Word Representation

			From Google:
			word2vec
		'''
		self.embeddings_index = {}
		'''
			Opens and iterates over every line in the file and appends
			to the embeddings_index
		'''
		f = open(os.path.join(self.GLOVE_DIR, 'glove.6B.100d.txt'))
		
		#iterates over each line
		for line in f:
			#each line has a new embedding
			values = line.split()
			
			#first value in the array is the name of the word
			word = values[0]
			
			#coefcients for all other words that are similar to the word we are interested in
			coefs = np.asarray(values[1:], dtype='float32')
			self.embeddings_index[word] = coefs
		f.close()

	def getFeatures(self):
		'''Gets some features for use in the machine learning model

		'''

		'''
			Casting question1 and question2 as strings
		'''
		self.train.loc[:,['question1', 'question2']] = self.train.loc[:,['question1', 'question2']].astype(str)
		self.test.loc[:,['question1', 'question2']] = self.test.loc[:,['question1', 'question2']].astype(str)


	def getWordTokens(self):
		'''Applies the word_tokenize function to train/test 
			The word_tokenize function from the nltk module takes a sentance and returns a dictionary corresponding
			To each space in the sentance.
			
			This will transform self.xTrain/self.yTrain from having a string for each question as a variable to a dictionary
			
			Applies for both question1 and question2
		'''

		'''
			Fitting a tokenizer object, the tokenizer object allows for the X most 
			common words to be given a numeric vector

			The goal here is to employ some word embedding algorithm to get words
			with similiar meanings 
		'''
		self.tokenizer = Tokenizer(num_words=self.MAX_WORDS)
		
		#Applies a numeric vector scale from 0 to self.MAX_WORDS
		self.tokenizer.fit_on_texts(pd.concat([self.train.question1, self.train.question2, self.test.question1, self.test.question2]))

		'''
			Fits the tokenizer on question1/question2 of train/test

			The goal here is to convert each string into a the numeric representations scaled above
			
			Each string will be converted into a sequence of numeric word tokens
		'''
		
		self.train.loc[:,'q1Token'] = self.tokenizer.texts_to_sequences(self.train.loc[:,'question1'])
		self.train.loc[:,'q2Token'] = self.tokenizer.texts_to_sequences(self.train.loc[:,'question2'])
		
		self.test.loc[:,'q1Token'] = self.tokenizer.texts_to_sequences(self.test.loc[:,'question1'])
		self.test.loc[:,'q2Token'] = self.tokenizer.texts_to_sequences(self.test.loc[:,'question2'])
		


	'''

		
		self.train.loc[:,'q1Token'] = self.train.loc[:,'question1'].apply(nltk.word_tokenize)
		self.train.loc[:,'q2Token'] = self.train.loc[:,'question2'].apply(nltk.word_tokenize)
		
		self.test.loc[:,'q1Token'] = self.test.loc[:,'question1'].apply(nltk.word_tokenize)
		self.test.loc[:,'q2Token'] = self.test.loc[:,'question2'].apply(nltk.word_tokenize)
 
	'''


	def getEmbeddedMatrix(self):
		'''Does the Pre-processing necessary to use the pre-trained word embedding
		'''
		'''
			Gets a dictionary where the key is the name of the word and the value is the 
			number of times the token appears in the dataset
		'''		
		self.word_index = self.tokenizer.word_index

		self.num_words = min(self.MAX_NB_WORDS, len(self.word_index)) + 1
		self.embedding_matrix = np.zeros((self.num_words, self.EMBEDDING_DIM))


		'''
			Iterates over every distinct word in the question1/question2 for train/test

			And adds the embedded pre-trained weights to a matrix
		'''
		for word, i in self.word_index.items():
			if i >= self.MAX_NB_WORDS:
				continue
			embedding_vector = self.embeddings_index.get(word)
			if embedding_vector is not None:
				# words not found in embedding index will be all-zeros.
				self.embedding_matrix[i] = embedding_vector



	def getQuantitative(self):
		'''Getting quantitative features from word_tokenize
		
			word_tokenize returns a list for each word split on whitespace
			This function will add some quantitative features to go along with this information

			the first of which is the wordCount which is simply the number of words in each question
		'''
		self.train.loc[:,'q1WordCount'] = self.train.loc[:,'q1Token'].apply(len)
		self.train.loc[:,'q2WordCount'] = self.train.loc[:,'q2Token'].apply(len)

		self.test.loc[:,'q1WordCount'] = self.test.loc[:,'q1Token'].apply(len)
		self.test.loc[:,'q2WordCount'] = self.test.loc[:,'q2Token'].apply(len)


	def getTrainTest(self, percentTrain):
		'''Splits the training dataset into train and test

			self.xTrain = features and id for train
			self.yTrain = one dimension array for the ground truth of train
			self.xTest = features and id for test
			self.yTest = one dimension array for the ground truth of test

			percentTrain = argument passed by the client in the interval (0,1) that gives the
			ratio of the total training set to subset from
		'''


		'''
			splitting into train and test
		'''
		self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(self.train.drop('is_duplicate',axis = 1), self.train.loc[:,'is_duplicate'], train_size = percentTrain, random_state = self.RANDOM_STATE)



	def getDeepModel(self):
		'''Gets a deep learning model

			The first layer of this model will be an embedded layer

		'''

		self.embedding_layer = Embedding(self.num_words, self.EMBEDDING_DIM, weights = [self.embedding_matrix], input_length = self.MAX_SEQUENCE_LENGTH, trainable = False)

		paddedX = pad_sequences(self.xTrain.q1Token, 238)
	
		paddedX2 = pad_sequences(self.xTrain.q2Token, 238)
		
		paddedTest = pad_sequences(self.xTest.q1Token, 238)

		paddedTest2 = pad_sequences(self.xTest.q2Token, 238)
		sequence_input = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
		embedded_sequence = self.embedding_layer(sequence_input)


		sequence_2_input = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
		embedded_sequences_2 = self.embedding_layer(sequence_2_input)
		


		#x = Conv1D(128, 5, activation='relu')(embedded_sequences)

		lstm_layer = LSTM(13)
		x = lstm_layer(embedded_sequence)

		question2 = lstm_layer(embedded_sequences_2)
		merged = concatenate([x,question2])
		merged =Dense(112, activation='sigmoid')(merged)
		preds = Dense(1, activation='softmax')(merged)

		model = Model([sequence_input, sequence_2_input], preds)


		model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

		model.fit([paddedX, paddedX2], self.yTrain, batch_size=1000, epochs=2, validation_data=([paddedTest, paddedTest2], self.yTest))
	
	def getModel(self):
		'''Build an adaboost model using self.xTrain
			
			self.clf = an adaboostclassifier 
		'''
		'''
		self.clf = AdaBoostClassifier(n_estimators=10, n_jobs = 3)
		self.clf.fit(self.xTrain.loc[:,['q1WordCount','q2WordCount']], self.yTrain)
		scores = cross_val_score(self.clf, self.xTrain.loc[:,['q1WordCount','q2WordCount']], self.yTrain, n_jobs = 3)
		'''

		self.nb = GaussianNB()
		self.nb.fit(self.xTrain.loc[:,['q1WordCount','q2WordCount']], self.yTrain)
		scores = cross_val_score(self.nb, self.xTrain.loc[:,['q1WordCount','q2WordCount']], self.yTrain, n_jobs = 3, cv = 5)
		print('Average cross validation score across 5 provisions\n', scores.mean()) 


	def getLogLoss(self, predictArray):
		'''Gets the Log-loss for the training set on the test set
			
			This function takes an arguement of the prediction for whether the questions are a duplicate or not
		'''
		print('Your logloss is :')
		print(metrics.log_loss(self.yTest, predictArray))
		return(metrics.log_loss(self.yTest, predictArray))

	def writeResults(self):
		'''Writes the results of the model to a csv

			Predicts on the full test set and writes the results to a csv
		'''

		self.test.loc[:,'is_duplicate'] = self.nb.predict_proba(self.test.loc[:,['q1WordCount','q2WordCount']])

		self.results = self.test.loc[:,['test_id', 'is_duplicate']]

		'''
			Cross-platform solution for findining the home directory
		'''
		self.results.to_csv(expanduser("~") + '/Documents/quora/results2.csv', index = False)

if __name__ == '__main__':
	quoraObj = quoraModel()

	quoraObj.loadWordEmbedding()

	quoraObj.getFeatures()
	
	quoraObj.getWordTokens()

	quoraObj.getEmbeddedMatrix()

	print(quoraObj.embedding_matrix.shape)


	quoraObj.getTrainTest(.75)
	quoraObj.getDeepModel()
	
	'''
	quoraObj.getQuantitative()
	
	
	logLoss = quoraObj.getLogLoss([0]*quoraObj.yTest.shape[0])

	quoraObj.writeResults()
	'''
