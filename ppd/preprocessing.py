#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime

from config import Config

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import math
import threading

class Preprocessing():
	"""
	class: Preprocessing
     
    this class is to preprocess the dataset, do a veaiety of transformations on the raw data 
     
     as an independent class Preprocessing could facilitate multi-threading and thus save time in model training

	:min_max: standardize the range of independent variables of data. After min-max: x'∈[0,1]
    
	:standard: Standardization makes the values of each feature in the data have zero-mean and unit-variance. After standard: x' ~ N(0,1)
    
        
	:median: fill na by median
    
	:log: log10 transformation, ignore the negative cases
    
	:log_move: move the entire column to the right by the min value in the column then do log10 trans, this avoids neg values in original columns
    
	:log_move_cor: screen out the feature having too-low correlation with target variable and then do log10 trans
    
	:log_move_standard: do log_move and standard
    
	:log_standard: do log and standard
    
	"""
    
	def __init__(self,config):
		"""
		:type config: Config
		:configure the initial information
		"""
		self.config=config
		pass

	def features_type(self):
		"""
		load feature type file
         return data frame contains info of feature type: numerical or categorical
		"""
		reader=pd.read_csv(self.config.path_feature_type,iterator=False,delimiter=',')
		features=reader
		features.to_csv('check_out_features',index=None)

		return features

	def load_cor_feature(self):
		"""
         load correlations of feature
         return list of features having absolute correlation value higher than 0.01
		"""
		reader=pd.read_csv(self.config.path_cor,iterator=False,delimiter=',',encoding='utf-8',header=None)
		cor_features=set([])
		for i in range(len(reader[0])):
			if abs(reader[1][i])>=0.01:
				cor_features.add(reader[0][i])

		print ('cor_features:',len(cor_features))
		return cor_features

	def scale_X(self):
		"""
         preprocess features according to the value of scale_level and featuer type
         return combination of transformed numeric and category dummy
		"""
		types=self.features_type()
        
          
		use=['uid']#list to store names of numerical features 
                 
		category_use=[]#list to store names of categorical features  
		if self.config.scale_level=='log_move_cor':
			cor_features=self.load_cor_feature()

		for i,t in enumerate(types['type']):
			if t=='category':
				category_use.append(types['feature'][i])#store all categorical features' name into category_use
			else:
				if self.config.scale_level=='log_move_cor':
					if types['feature'][i] in cor_features:
						use.append(types['feature'][i])
				else:
					use.append(types['feature'][i])#store all numerical features' name into use
		print('print use: ',tuple(use))

        #####################
        #      numeric      #
        #####################
		
        #process training and predicted(testing) data 
		train_reader=pd.read_csv(self.config.path_train_x,iterator=False,delimiter=',',usecols=tuple(use))
		test_reader=pd.read_csv(self.config.path_predict_x,iterator=False,delimiter=',',usecols=tuple(use))#,usecols=tuple(use) has problem


		len_train=len(train_reader)
		len_predict=len(test_reader)

		reader=pd.concat([train_reader,test_reader])

		reader.fillna(-1,inplace=True)

		data=np.array(reader)

		X=data[:,1:]
		if self.config.scale_level=='log':
			X=self.log_scale(X)
		elif self.config.scale_level=='log_move':
			X=self.log_scale_move(X)
		elif self.config.scale_level=='standard':	
			X=self.standard_scale(X)
		elif self.config.scale_level=='normalize':
			X=self.normalizer_scale(X)
		elif self.config.scale_level=='min_max':
			X=self.min_max_scale(X)
		elif self.config.scale_level=='median':
			X=self.fill_scale(X,self.median_feature(X))
		elif self.config.scale_level=='log_move_standard':
			X=self.log_scale_move(X)
			X=self.standard_scale(X)
		elif self.config.scale_level=='log_standard':
			X=self.log_scale(X)
			X=self.standard_scale(X)
		elif self.config.scale_level=='log_move_cor':
			X=self.log_scale_move(X)

		uid=np.array(data[:,0],dtype='int')
		uid=uid.astype('str')	
        
        #####################
        #      category     #
        #####################
		#load category features
		category_train_reader=pd.read_csv(self.config.path_train_x,iterator=False,delimiter=',',usecols=tuple(category_use))
		category_reader=pd.read_csv(self.config.path_predict_x,iterator=False,delimiter=',',usecols=tuple(category_use))

		len_train=len(category_train_reader)
		len_predict=len(category_reader)

		#get category dummy
		category_reader=pd.concat([category_train_reader,category_reader])
		category_reader.fillna(-1,inplace=True)

		dummys=pd.DataFrame()
		j=1
		for i in range(len(category_use)):
			temp_dummys=pd.get_dummies(category_reader[category_use[i]])
			if j==1:
				j+=1
				dummys=temp_dummys
			else:
				dummys=np.hstack((dummys,temp_dummys))
		
		#merge numeric and category dummy
		print(len(X),len(dummys),len(uid))

		X=np.column_stack((data[:,0],X))
		X=np.hstack((X,dummys))
		X_train=X[:len_train]
		X_predict=X[len_train:(len_train+len_predict)]

		#output
		pd.DataFrame(X_train).to_csv(self.config.path_train_x_scaled,sep=',',mode='w',header=None,index=False)
		pd.DataFrame(X_predict).to_csv(self.config.path_predict_x_scaled,sep=',',mode='w',header=None,index=False)
		pd.DataFrame(uid).to_csv(self.config.path_uid,sep=',',mode='w',header=None,index=False)
		print(self.config.scale_level+"\n")

	def standard_scale(self,X):
		"""
		:type X: numpy.array feature matrix
		:return type X: numpy.array transformed feature matrix
		"""
		scaler=StandardScaler()
		return scaler.fit_transform(X)

	def min_max_scale(self,X):
		scaler=MinMaxScaler()
		return scaler.fit_transform(X)

	def normalizer_scale(self,X):
		scaler=Normalizer()
		return scaler.fit_transform(X)

	def median_feature(self,X):
		m,n=X.shape
		X_median=[]
		for i in range(n):
			median=np.median(X[:,i])
			X_median.append(median)
		return X_median

	#fill -1 (nan) by median
	def fill_scale(self,X,X_median):
		m,n=X.shape
		for i in range(m):
			for j in range(n):
				if X[i][j]==-1 or X[i][j]==-2:
					X[i][j]=X_median[j]
		return X

	def log_scale(self,X):
		m,n=X.shape
		for i in range(m):
			for j in range(n):
				if X[i][j]>0:
					X[i][j]=math.log10(X[i][j])
		return X

	def log_scale_move(self,X):
		n,m=X.shape
		for i in range(m):
			column=X[:,i]

			c_min=np.min(column)
			for j in range(n):
				column[j]=math.log10(column[j]-c_min+1)

			X[:,i]=column
		return X

def scale_wrapper():
	"""
    configure the variable scale, assign it different values that pre-stored in the list scales
    put the tasks that handle different transforming into threads 
    do multi-threading
	"""
	scales=['log_move_standard']#'log','log_move','standard','normalize','min_max','median',
	threads=[]
	for x in scales:
		config_instance=Config(x)#Configure needs the type of data-transforming as variable
		preprocessing_instance=Preprocessing(config_instance) #instantiate Preprocessing
		threads.append(threading.Thread(target=preprocessing_instance.scale_X))

	for t in threads:
		t.start()

	for t in threads:
		t.join()

def main():

	scale_wrapper()

if __name__ == '__main__':
    
	start=datetime.now()
	main()
	end=datetime.now()
	print("All Run time:"+str(float((end-start).seconds)/60.0)+"min / "+str(float((end-start).seconds))+"s")
