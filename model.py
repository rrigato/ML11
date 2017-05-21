#!usr/bin/python3
import pandas as pd
import numpy as np
import nltk
from sklearn import metrics
from sklearn.cross_validation import train_test_split
class quoraModel:
	def __init__(self):
		'''
			The train and test datasets contain an ID
			and question 1/ question 2 which the model is supposed to differentiate
			whether the questions are similar or different
			
			self.RANDOM_STATE = the random_state to initialize the split for reproducible results
		'''
		self.loadData()
		self.RANDOM_STATE = 1

	def loadData(self):
		'''Loads the train/test data from a csv

			self.train = Also contains a label for the ground truth
			-0 if the questions are not the same 1 if they are

		'''
		self.train = pd.read_csv('/home/ryan/Documents/quora/train.csv')
		self.test = pd.read_csv('/home/ryan/Documents/quora/test.csv')

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
			Casting question1 and question2 as strings
		'''
		self.train.loc[:,['question1', 'question2']] = self.train.loc[:,['question1', 'question2']].astype(str)
		self.test.loc[:,['question1', 'question2']] = self.test.loc[:,['question1', 'question2']].astype(str)

		'''
			splitting into train and test
		'''
		self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(self.train.loc[:, 'id':'question2'], self.train.loc[:,'is_duplicate'], train_size = percentTrain, random_state = self.RANDOM_STATE)

	def getWordTokens(self):
		'''Applies the word_tokenize function to train/test 
			The word_tokenize function from the nltk module takes a sentance and returns a dictionary corresponding
			To each space in the sentance.
			
			This will transform self.xTrain/self.yTrain from having a string for each question as a variable to a dictionary
			
			Applies for both question1 and question2
		'''
		
		self.xTrain.loc[:,'q1Token'] = self.xTrain.loc[:,'question1'].apply(nltk.word_tokenize)
		self.xTrain.loc[:,'q2Token'] = self.xTrain.loc[:,'question2'].apply(nltk.word_tokenize)
		
		self.xTest.loc[:,'q1Token'] = self.xTest.loc[:,'question1'].apply(nltk.word_tokenize)
		self.xTest.loc[:,'q2Token'] = self.xTest.loc[:,'question2'].apply(nltk.word_tokenize)

		print(self.xTest.loc[:,'q1Token':'q2Token'].head())
		print(self.xTrain.loc[:,'q1Token':'q2Token'].head())


	def getLogLoss(self, predictArray):
		'''Gets the Log-loss for the training set on the test set
			
			This function takes an arguement of the prediction for whether the questions are a duplicate or not
		'''
		print('Your logloss is :')
		print(metrics.log_loss(self.yTest, predictArray))
		return(metrics.log_loss(self.yTest, predictArray))

if __name__ == '__main__':
	quoraObj = quoraModel()

	quoraObj.getTrainTest(.75)
	
	quoraObj.getWordTokens()
	
	logLoss = quoraObj.getLogLoss([0]*quoraObj.yTest.shape[0])
