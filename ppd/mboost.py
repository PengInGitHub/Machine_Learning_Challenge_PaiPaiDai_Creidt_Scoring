#! /usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime
from config import Config
from sklearn.cross_validation import KFold
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import xgboost as xgb

class Mboost(object):
	"""
	:class Mboost
	:train model in a cross validation manner
	"""
	def __init__(self,config):
		"""
		:type config: Config
		"""
		self.config=config
		pass

	def fold(self,len_0,len_1,n_folds):
		"""
		:type len_0: int
		:type len_1: int
		:type n_folds: int
		:rtype f0: List[List[int]]
		:rtype f1: List[List[int]]
		:divide the dataset into n fold, return the list of the index
		"""
         #set random seed value
		random_state=self.config.fold_random_state 
         #k-fold for labeled 0
		kf0=KFold(n=len_0, n_folds=n_folds, shuffle=True,random_state=random_state)
         #k-fold for labeled 1
		kf1=KFold(n=len_1,n_folds=n_folds, shuffle=True,random_state=random_state)
		f0=[]
		f1=[]
        
		for train_index_0,test_index_0 in kf0:
			f0.append([train_index_0.tolist(),test_index_0.tolist()])
		for train_index_1,test_index_1 in kf1:
			f1.append([train_index_1.tolist(),test_index_1.tolist()])
		return f0,f1


	def level_train(self,clf,level,name,X_0,X_1,uid_0,uid_1):
		"""
		:type clf: scikit-learn classifier or scikit-learn regressor        
        	:type level: str level of training
		:type name: str name of classifier
		:type X_0: numpy.array feature matrix for X labeled 0
		:type X_1: numpy.array feature matrix for X labeled 1
		:type uid_0: List uid for those labeled 0
		:type uid_1: List uid for those labeled 1        
        
         level_train, function trains models in a cross validation manner
         divide the 0 and 1 labeled features into 5 folds as training set while the ramaining (n-1) fold is used to validate the training
		do the training in a loop way, each fold would be one time used as validation set
		in this sense each training procedure has n models to train
		"""
		n_folds=self.config.n_folds
		f0,f1=self.fold(len(X_0),len(X_1),n_folds)         

		predicts=[]
		test_uids=[]
		scores=[]
		part_uids=[]

		for i in range(n_folds):
			train_index_0,test_index_0=f0[i][0],f0[i][1]
			train_index_1,test_index_1=f1[i][0],f1[i][1]

			train_1=X_1[train_index_1]
			test_1=X_1[test_index_1]

			train_0=X_0[train_index_0]
			test_0=X_0[test_index_0]

			test_uid_1=uid_1[test_index_1]
			test_uid_0=uid_0[test_index_0]

			#obtain labels
			y_train=np.hstack((np.ones(len(train_1)),np.zeros(len(train_0))))
			y_test=np.hstack((np.ones(len(test_1)),np.zeros(len(test_0))))

			#merge test uid
			test_uid=np.hstack((test_uid_1,test_uid_0))

			#merge train regardless of label, test as well
			x_train=np.vstack((train_1,train_0))
			x_test=np.vstack((test_1,test_0))
            
			#let the classifier fit the model
			clf.fit(x_train,y_train)
            
			try:
				y_pred=clf.predict_proba(x_test)#predict, probability as the output
				y_pred=y_pred[:,1]
			except:	
				y_pred=clf.predict(x_test)#otherwise classifiers have probability as output directly

			auc_score=metrics.roc_auc_score(y_test,y_pred)#calculate AUC score in each fold

			predicts.extend((y_pred).tolist())#save result in each fold

			test_uids.extend(test_uid.tolist())

			print(auc_score)
			scores.append(auc_score)

		self.output_level_train(predicts,test_uids,scores,level,name)#save output result
		print(name+" in Mboost.level_train achieves average scores:",np.mean(scores))

	def xgb_level_train(self,level,name,X_0,X_1,uid_0,uid_1,params,round):
		"""
		:type level: str level of training
		:type name: str name of classifier
		:type X_0: numpy.array feature matrix for X labeled 0
		:type X_1: numpy.array feature matrix for X labeled 0
		:type uid_0: List uid for those labeled 0
		:type uid_1: List uid for those labeled 1
        
		:type params: dict XGBoost configure parameters
		:type round: int XGBoost number of iteration
		"""
		n_folds=self.config.n_folds
		f0,f1=self.fold(len(X_0),len(X_1),n_folds)

		predicts=[]
		test_uids=[]
		scores=[]

		for i in range(n_folds):
			train_index_0,test_index_0=f0[i][0],f0[i][1]
			train_index_1,test_index_1=f1[i][0],f1[i][1]

			train_1=X_1[train_index_1]
			test_1=X_1[test_index_1]

			train_0=X_0[train_index_0]
			test_0=X_0[test_index_0]

			test_uid_1=uid_1[test_index_1]
			test_uid_0=uid_0[test_index_0]

			train_1=np.vstack((train_1,train_1))

			y_train=np.hstack((np.ones(len(train_1)),np.zeros(len(train_0))))		
			y_test=np.hstack((np.ones(len(test_1)),np.zeros(len(test_0))))

			test_uid=np.hstack((test_uid_1,test_uid_0))

			x_train=np.vstack((train_1,train_0))
			x_test=np.vstack((test_1,test_0))

			dtest=xgb.DMatrix(x_test)
			dtrain=xgb.DMatrix(x_train,label=y_train)
			watchlist=[(dtrain,'train')]

			model=xgb.train(params,dtrain,num_boost_round=round,evals=watchlist,verbose_eval=False)
			y_pred=model.predict(dtest)

			auc_score=metrics.roc_auc_score(y_test,y_pred)
			predicts.extend((y_pred).tolist())#save result in each fold
			test_uids.extend(test_uid.tolist())
            
			print( auc_score)
			scores.append(auc_score)
            
		self.output_level_train(predicts,test_uids,scores,level,name)
		print (name+" average scores:",np.mean(scores))

	def output_level_train(self,predicts,test_uids,scores,level,name):	
		"""
		:type predicts: List[float] list of predicted values
		:type test_uids: List[str] uid of predict
		:type scores: List[float] AUC score in each fold
		:type level: str level of training
		:type name: str name of classifier
        
		:output is the function to save the predicted result of each classifier in each fold to the underlying file
		"""
		f1=open(self.config.path_train+level+'/'+name+'.csv','w', newline='')
		f2=open(self.config.path_train+level+'/'+name+'_score.csv','w', newline='')
		for i in range(len(test_uids)):
			f1.write(str(test_uids[i])+","+str(predicts[i])+"\n")

		for score in scores:
			f2.write(str(score)+"\n")

		f1.close()
		f2.close()
        
        

	def level_predict(self,clf,level,name,X_0,X_1,predict_X,predict_uid):
		"""
		:type clf: scikit-learn classifier or scikit-learn regressor
		:type level: str 
		:type name: str 
		:type X_0: numpy.array 
		:type X_1: numpy.array
		:type predict_X: 
		:type predict_uid: 
		:not in a cross validation manner, train one model and make one prediction result
		"""
		start=datetime.now()
		x_train=np.vstack((X_1,X_0))
		y_train=np.hstack((np.ones(len(X_1)),np.zeros(len(X_0))))

		clf.fit(x_train,y_train)
		try:
			pred_result=clf.predict_proba(predict_X)
			self.output_level_predict(pred_result[:,1],predict_uid,level,name)
		except:
			pred_result=clf.predict(predict_X)
			self.output_level_predict(pred_result,predict_uid,level,name)
		
		end=datetime.now()
		print( "finish predict:"+name+" Run time:"+str(float((end-start).seconds)/60.0)+"min / "+str(float((end-start).seconds))+"s")

	def xgb_predict(self,level,name,X_0,X_1,predict_X,predict_uid,params,round):
		"""
		:type name: str 
		:type X_0: numpy.array
		:type X_1: numpy.array 
		:type predict_X: 
		:type predict_uid: 
		:type params: dict 
		:type round: int 
		:xgb prediction, not in a cross validation manner, train one model and make one prediction result
		"""
		start=datetime.now()
		x_train=np.vstack((X_1,X_0))
		y_train=np.hstack((np.ones(len(X_1)),np.zeros(len(X_0))))
		dtrain=xgb.DMatrix(x_train,label=y_train)
		watchlist=[(dtrain,'train')]
		model=xgb.train(params,dtrain,num_boost_round=round,evals=watchlist,verbose_eval=False)

		dpredict=xgb.DMatrix(predict_X)
		predict_result=model.predict(dpredict)
		self.output_level_predict(predict_result,predict_uid,level,name)
		end=datetime.now()
		print ("finish predict:"+name+" Run time:"+str(float((end-start).seconds)/60.0)+"min / "+str(float((end-start).seconds))+"s")

	def output_level_predict(self,predicts,test_uids,level,name):	
		"""
		:type predicts: List[float]
		:type test_uids: List[str]
		:type level: str
		:type name: str
		:output the predicted result of each classifier in each fold to the file
		"""

		f1=open(self.config.path_predict+level+'/'+name+'.csv','w', newline='')
		f1.write('"uid","score"\n')
		for i in range(len(test_uids)):
			f1.write(str(test_uids[i])+","+str(predicts[i])+"\n")
		f1.close()