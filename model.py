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
		print(self.train.head())
		print(self.test.head())

	def getTrainTest(self, percentTrain):
		'''Splits the training dataset into train and test

			self.xTrain = features and id for train
			self.yTrain = one dimension array for the ground truth of train
			self.xTest = features and id for test
			self.yTest = one dimension array for the ground truth of test

			percentTrain = argument passed by the client in the interval (0,1) that gives the
			ratio of the total training set to subset from
		'''
		self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(self.train.loc[:, 'id':'question2'], self.train.loc[:,'is_duplicate'], train_size = percentTrain, random_state = self.RANDOM_STATE)



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
	
	logLoss = quoraObj.getLogLoss([0]*quoraObj.yTrain.shape[0])
