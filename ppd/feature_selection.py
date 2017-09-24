#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import random

from itertools import chain
from config import Config
from load_origin_data import Load_origin_data

class Feature_selection():

    #select preprocessed and statistics features,    
    #round 1: remove features with super low variance or high correction with other columns
    
	def __init__(self,config):
		self.config=config


###############
#   round 1   #
###############

	def fill_missing_value(self):

		X=pd.read_csv(self.config.path+'final.csv')
		X=np.array(X,dtype=float)#.drop('uid', axis=1)
		train_y=pd.read_csv(self.config.path+'train_y.csv')
		len_train=len(train_y)

		print( 'X shape: ', X.shape, 'len y',len_train)
		m,n=X.shape
		l=[]
		split_features_num=0 #number of removed features due to small variance
		for i in range(n):
	         #nan to 1
			col=list(chain(map(self._deal_nan,X[:,i])))
			col=np.array(col)
			
             #remove features with variance lower than 0.001
			tmp_col=(np.array(col)-np.min(col))/(np.max(col)-np.min(col))
			tmp_col_var=np.var(tmp_col)
			if tmp_col_var<0.001:
				split_features_num+=1
				continue

			# replace values higher than 6 times of std by (mean+6*std)
			col2=col[np.where(col>=0)]#
			self.col_mean=np.mean(col2)
			self.col_std=np.std(col2)*6+self.col_mean
			
			col=list(chain(map(self._deal_std,col)))
			per=float(len(col2))/float(m)
			# replace missing value by mean for those have missing rate under 20%
			if per>0.85:
				col=list(chain(map(self._deal_fill,col)))

			if self.is_choose_col(l, col):
			     l.append(col)

			if i%10==0:
				print(i)

		print ('split feature: ',split_features_num)
		X=np.array(l).transpose()

		X_train=X[:len_train]

		X_predict=X[len_train:]
		print (X.shape)
		return X_train,X_predict

	def is_choose_col(self,l,new_col):
		"""
		cor higher than 0.8 with columns already choosen will be rejected
		"""
		for col in l:
			cor=np.corrcoef(col,new_col)
			if cor[0,1]>0.8:
				return False
		return True
###############
#   round 2   #
###############
#not working yet
	def run_selection(self):
		params={
			'min_cols':100, 
			'max_iter':200, 
			'max_no_change_iter':20,
			'min_sim':0.9, 
			'seed':1, 
			'slient':False
		}
		X_train=pd.read_csv("final_train_select_round_1.csv",header=None)
		X_predict=pd.read_csv("final_test_select_round_1.csv",header=None)
		X=pd.concat([X_train, X_predict], axis=0)
		X=self.col_selection(X, params)
		X_train=X[:len_train]
		X_train=np.hstack((self.train_uids,X_train))

		X_predict=X[len_train:]
		X_predict=np.hstack((self.predict_uids,X_predict))
		print (X.shape)
		return X_train,X_predict



	def col_selection(self,X,params):
		"""
		params={
			'min_cols':500, #min col num
			'max_iter':0, #max rounds to compare
			'max_no_change_iter':,#max rounds no change
			'min_sim':0.99, #choose on from those are very similar
			'seed':3, #random seed
			'slient':False
		}
		
		"""
		random.seed(params['seed'])

		no_change_iter=0
		m,n=X.shape
		last_n=n+1
		for i in range(params['max_iter']):		
			m,n=X.shape
			if n<=params['min_cols']:
				break
			if last_n==n:
				no_change_iter+=1
			if no_change_iter>=params['max_no_change_iter']:
				break
			X=self._deal_col_selection(X, params)
			last_n=n
			if not params['slient']:
				print ('iter: ',i,' col num: ',n,' no_change_iter: ',no_change_iter)

		return X

	def _deal_col_selection(self,X,params):
		m,n=X.shape
		indexes=[i for i in range(n)]
		random.shuffle(indexes)
		X=X.iloc[:,indexes]

		l=[]
		lastChoose=False
		for i in range(n-1):
			isChooseOne=self.is_choose_one(X.iloc[:,i],X.iloc[:,i+1],params['min_sim'])
			if isChooseOne:
				lastChoose=True
			else:
				l.append(i)
				lastChoose=False
		
		if n-1 not in l:
			l.append(n-1)

		X=X.iloc[:,l]
		return X

	def is_choose_one(self,col1,col2,sim):
		cor=np.corrcoef(col1,col2)
		if cor[0,1]>sim:
			print( 'bingo')
			return True
		else:
			return False


	def _deal_nan(self,n):
		if str(n)=='nan':
			return -1
		else:
			return n

	def _deal_std(self,n):
		if n>self.col_std:
			return round(self.col_std,2)
		else:
			return round(n,2)

	def _deal_fill(self,n):
		if n==-1:
			return round(self.col_mean,2)
		return round(n,2)

	def output_selection_1(self,X_train,X_predict):
		pd.DataFrame(X_train).to_csv(self.config.path+"final_train_select_round_1.csv",index=False,header=None)
		pd.DataFrame(X_predict).to_csv(self.config.path+"final_test_select_round_1.csv",index=False,header=None)

	def output_selection_2(self,X_train,X_predict):
		pd.DataFrame(X_train).to_csv(self.config.path+"final_train_select_round_2.csv",index=False,header=None)
		pd.DataFrame(X_predict).to_csv(self.config.path+"final_test_select_round_2.csv",index=False,header=None)

def main():
#round 1
		instance=Feature_selection(Config(''))
		X_train,X_predict=instance.fill_missing_value()
		instance.output_selection_1(X_train, X_predict)
#round 2		
		#X_train,X_predict=instance.run_selection()
		#instance.output_selection_2(X_train, X_predict)

if __name__ == '__main__':
	main()