#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from config import Config
from sklearn.cross_validation import train_test_split

class Load_origin_data():
	def __init__(self,config):
		self.config=config

	def load_feature_type(self):
		"""
		load feature type numeric or category
		"""
		feature_type=pd.read_csv(self.config.path_feature_type,header=0)

		categorical_feature=[]
		numerical_feature=[]

		for i,t in enumerate(feature_type['type']):
			if feature_type['feature'][i]=='y':
				continue
			if t=='category':
				categorical_feature.append(feature_type['feature'][i])
			else:
				numerical_feature.append(feature_type['feature'][i])

		return categorical_feature,numerical_feature

	def load_train_X(self):
		"""
         load train
		"""
		categorical_feature,numerical_feature=self.load_feature_type()
		reader_category=pd.read_csv(self.config.path_train_x)[['uid']+categorical_feature]
		reader_numeric=pd.read_csv(self.config.path_train_x)[['uid']+numerical_feature]
		return reader_category,reader_numeric

	def load_predict_X(self):
		"""
         load test
		"""
		categorical_feature,numerical_feature=self.load_feature_type()
		reader_category=pd.read_csv(self.config.path_predict_x)[['uid']+categorical_feature]
		reader_numeric=pd.read_csv(self.config.path_predict_x)[['uid']+numerical_feature]
		return reader_category,reader_numeric

	def load_train_y(self):
		"""
         load label
		"""
		reader_y=pd.read_csv(self.config.path_train_y)
		return reader_y

	def load_train_uid(self):
		"""
		load train uid
		"""
		uid_reader=pd.read_csv(self.config.path_origin_train_x,iterator=False,delimiter=',',usecols=tuple(['uid']))

		uid=np.array(uid_reader,dtype='int')
		uid=np.ravel(uid)
		return uid

	def load_predict_uid(self):
		"""
		load test uid
		"""
		uid_reader=pd.read_csv(self.config.path+'test_uid.csv',iterator=False,delimiter=',',header=None)
		uid=np.array(uid_reader)#,dtype='int'
		uid=np.ravel(uid)
		return uid

	def load_test_y(self):
		"""
         local validation
		"""
		y_reader=pd.read_csv(self.config.path_predict_y)
		return y_reader    

	def load_data_for_statistics_features(self):
		"""
		load data for class Statistics_Features
		"""
		#load feature type
		categorical_feature,numerical_feature=self.load_feature_type()
		#load train and predict
		reader_category_train,reader_numeric_train=self.load_train_X()
		reader_category_predict,reader_numeric_predict=self.load_predict_X()
		y=self.load_train_y()

		reader_category_train = pd.merge(reader_category_train,y,on='uid')
		reader_numeric_train = pd.merge(reader_numeric_train,y,on='uid')

		reader_category_predict['y'] = [-99999 for i in range(len(reader_category_predict))]
		reader_numeric_predict['y'] = [-99999 for i in range(len(reader_numeric_predict))]

		#merge data
		reader_category=pd.concat([reader_category_train,reader_category_predict],ignore_index=True)
		reader_numeric=pd.concat([reader_numeric_train,reader_numeric_predict],ignore_index=True)

		return categorical_feature,numerical_feature,reader_category,reader_numeric    

	def save_final(self):

		statistics_features=pd.read_csv(self.config.path+'reader_statistics_features_output.csv',iterator=False,delimiter=',')
		scaled=pd.read_csv(self.config.path+'train_x_scale_log_move_standard.csv',iterator=False,delimiter=',',header=None)     
		scaled=scaled.rename(columns={0: 'uid'}) 
		final=pd.merge(scaled,statistics_features,on='uid')    		           
		final.to_csv('final.csv')    		           
		return final   		           

	def load_final(self):

		X=pd.read_csv(self.config.path+'final_train_select_round_1.csv',iterator=False,delimiter=',',header=None)
		X=np.array(X,dtype="float32")
                #all nan to zero
		X=np.nan_to_num(X)
		reader_y=self.load_train_y()
		data=np.array(reader_y)
		y=np.ravel(data[:,1:])
		uid=np.array(data[:,0])

		X_0=[]
		X_1=[]
		uid_0=[]
		uid_1=[]
		for i in range(len(y)):
			if y[i]==1: 
				X_1.append(X[i])
				uid_1.append(uid[i])
			else:
				X_0.append(X[i])
				uid_0.append(uid[i])
                
		return X,y,uid,np.array(X_0),np.array(X_1),np.array(uid_0),np.array(uid_1)
         
	def train_test_xy(self,random_state):
		"""
		20% of training is splited as validation in local training
		"""
		X,y,uid,X_0,X_1,uid_0,uid_1=self.load_final()#reverse the original data by label y

		train_X_0,test_X_0,train_uid_0,test_uid_0=train_test_split(X_0,uid_0,test_size=0.2,random_state=1)#split them into train and validation
		train_X_1,test_X_1,train_uid_1,test_uid_1=train_test_split(X_1,uid_1,test_size=0.2,random_state=1)#split them into train and validation
		return train_X_0,test_X_0,train_X_1,test_X_1,train_uid_0,test_uid_0,train_uid_1,test_uid_1

	def load_final_test(self):

		X=pd.read_csv(self.config.path+'final_test_select_round_1.csv',iterator=False,delimiter=',',header=None)
		X=np.array(X)

		uid=self.load_predict_uid()
                
		return X,uid

	def load_final_test_2(self):

		X=pd.read_csv(self.config.path+'final_test_select_round_1.csv',iterator=False,delimiter=',',header=None)
		X=np.array(X,dtype="float32")
         #all nan to zero
		X=np.nan_to_num(X)
		reader_y=self.load_test_y()
		data=np.array(reader_y)
		y=np.ravel(data[:,1:])
		uid=np.array(data[:,0])

		X_0=[]
		X_1=[]
		uid_0=[]
		uid_1=[]
		for i in range(len(y)):
			if y[i]==1: 
				X_1.append(X[i])
				uid_1.append(uid[i])
			else:
				X_0.append(X[i])
				uid_0.append(uid[i])
                
		return X,y,uid,np.array(X_0),np.array(X_1),np.array(uid_0),np.array(uid_1)

	def local_verify_tune(self):

  	
		"""
		20% of training is splited as validation in local training
		"""
		X,y,uid,X_0,X_1,uid_0,uid_1=self.load_final()#reverse the original data by label y

		train_X_0,test_X_0,train_uid_0,test_uid_0=train_test_split(X_0,uid_0,test_size=0.2,random_state=1)#split them into train and validation
		train_X_1,test_X_1,train_uid_1,test_uid_1=train_test_split(X_1,uid_1,test_size=0.2,random_state=1)#split them into train and validation
		return train_X_0,test_X_0,train_X_1,test_X_1,train_uid_0,test_uid_0,train_uid_1,test_uid_1


	def local_verify(self):


		X_train,y_train,uid_train,X_0_train,X_1_train,uid_0_train,uid_1_train=self.load_final()#reverse the original data by label y
		X_test,y_test,uid_test,X_0_test,X_1_test,uid_0_test,uid_1_test=self.load_final_test_2()#reverse the original data by label y
                
		return X_0_train,X_0_test,X_1_train,X_1_test,uid_0_train,uid_0_test,uid_1_train,uid_1_test
