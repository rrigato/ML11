#!usr/bin/python3
import pandas as pd
import numpy as np
import nltk
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
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
		
		self.train.loc[:,'q1Token'] = self.train.loc[:,'question1'].apply(nltk.word_tokenize)
		self.train.loc[:,'q2Token'] = self.train.loc[:,'question2'].apply(nltk.word_tokenize)
		
		self.test.loc[:,'q1Token'] = self.test.loc[:,'question1'].apply(nltk.word_tokenize)
		self.test.loc[:,'q2Token'] = self.test.loc[:,'question2'].apply(nltk.word_tokenize)

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

		self.test.loc[:,'is_duplicate'] = self.nb.predict(self.test.loc[:,['q1WordCount','q2WordCount']])

		self.test.to_csv('/~/Documents/quora/results1.csv', index = False)

if __name__ == '__main__':
	quoraObj = quoraModel()

	quoraObj.getFeatures()
	
	quoraObj.getWordTokens()

	quoraObj.getQuantitative()
	quoraObj.getTrainTest(.75)
	
	logLoss = quoraObj.getLogLoss([0]*quoraObj.yTest.shape[0])
