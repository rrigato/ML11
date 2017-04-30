#!usr/bin/python3
import pandas as pd
import numpy as np
import nltk
from sklearn import metrics
class quoraModel:
	def __init__(self):
		'''
			The train and test datasets contain an ID
			and question 1/ question 2 which the model is supposed to differentiate
			whether the questions are similar or different
			

		'''
		self.loadData()

	def loadData(self):
		'''Loads the train/test data from a csv

			self.train = Also contains a label for the ground truth
			-0 if the questions are not the same 1 if they are

		'''
		self.train = pd.read_csv('/home/ryan/Documents/quora/train.csv')
		self.test = pd.read_csv('/home/ryan/Documents/quora/test.csv')
		print(self.train.head())
		print(self.test.head())

	def getLogLoss(self, predictArray):
		'''Gets the Log-loss for the training set on the test set
			
			This function takes an arguement of the prediction for whether the questions are a duplicate or not
		'''
		print('Your logloss is :')
		print(metrics.log_loss(self.train.loc[:,'is_duplicate'], predictArray))
		return(metrics.log_loss(self.train.loc[:,'is_duplicate'], predictArray))

if __name__ == '__main__':
	quoraObj = quoraModel()
	
	logLoss = quoraObj.getLogLoss([0]*quoraObj.train.shape[0])
