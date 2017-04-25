#!usr/bin/python3
import pandas as pd
import numpy as np
class quoraModel:
	def __init__(self):
		self.train = pd.read_csv('/home/ryan/Documents/quora/train.csv')
		self.test = pd.read_csv('/home/ryan/Documents/quora/test.csv')

if __name__ == '__main__':
	quoraObj = quoraModel()
