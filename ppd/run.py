#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime
from config import Config
import load_origin_data
import load_train_data
import load_predict_data

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier

from level_train_thread import Level_train_thread
from level_predict_thread import Level_predict_thread
from xgb_level_train_thread import Xgb_level_train_thread
from xgb_level_predict_thread import Xgb_level_predict_thread

from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

"""
run.py
the engine to complete all the taskes

functions list:
scale_wrapper: wrapper function to preprocess dataset
level_one_wrapper: wrapper function of training level one
level_one_predict: wrapper function of prediction level one

level_two_wrapper: wrapper function of training level two
level_two_predict: wrapper function of prediction level two

level_three_wrapper: wrapper function of training level three
level_three_wrapper: wrapper function of prediction level three

compare results among levels, overfitting occurs in level 3 so the trainning stopped at level 3
"""
        
###########################################################################
#                                                                         #
#                           level_one_wrapper                             # 
#                                                                         #
###########################################################################        
		
#level one model fitting
#the major task of this wrapper function is to obtain BBM(best single estimator) and DBM(model differs from BBM from largest scale)
#class BBM_DBM will take inputs from the outputs of this function
#the preprocessed data + additional statistical features are input
#5-fold CV is carried by class Mboost
#output:prediction results of training data and AUC of 5-folds

def level_one_wrapper():
     
	ftype=''#the choosen data preprocessing type
    
	level='level_one'#the level of training
     
	config_instance=Config('')#choose log_move transfered data

	load_data_instance=load_origin_data.Load_origin_data(config_instance)

	X,y,uid,X_0,X_1,uid_0,uid_1=load_data_instance.load_final()


	"""
	append a variety of individual classifiers into Level_train_thread
    obtain multiple results that are based on different classifiers and parameters 
    classifiers include: linear classifiers, random forest, gbm, ada boost, bagging of classifiers and xgb
	"""
     #the list of threads
	threads=[]
     #for all classifiers except for xgb, call Level_train_thread and call Mboost.level_train
	threads.append(Level_train_thread(config_instance,LogisticRegression(solver='sag'),level,'_lr_sag',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,LogisticRegression(max_iter=1000,solver='sag'),level,'_lr_sag_1000',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,LogisticRegression(max_iter=1500,solver='sag'),level,'_lr_sag_1500',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,LogisticRegression(solver='newton-cg'),level,ftype+'_lr_newton',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,LogisticRegression(solver='lbfgs'),level,ftype+'_lr_lbfgs',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,LogisticRegression(solver='liblinear'),level,ftype+'_lr_liblinear',X_0,X_1,uid_0,uid_1))
     
	#try to assign class weights to LogisticRegression, not working
	#threads.append(Level_train_thread(config_instance,LogisticRegression(max_iter=1000,class_weight={0:4092,1:27908},solver='sag'),level,'weighted_lr_sag_1000',X_0,X_1,uid_0,uid_1))
	#threads.append(Level_train_thread(config_instance,LogisticRegression(max_iter=1500,class_weight={0:4092,1:27908},solver='sag-cg'),level,'weighted_lr_sag_1500',X_0,X_1,uid_0,uid_1))
	#threads.append(Level_train_thread(config_instance,LogisticRegression(max_iter=1000,class_weight={0:27908,1:4092},solver='sag'),level,'weighted_lr_sag_1000_reverse',X_0,X_1,uid_0,uid_1))
	#threads.append(Level_train_thread(config_instance,LogisticRegression(max_iter=1500,class_weight={0:27908,1:4092},solver='sag-cg'),level,'weighted_lr_sag_1500_reverse',X_0,X_1,uid_0,uid_1))
	#threads.append(Level_train_thread(config_instance,LogisticRegression(max_iter=1500,class_weight={0:4092,1:27908},n_jobs=-1,solver='newton-cg',verbose=2),level,ftype+'weighted_lr_newton',X_0,X_1,uid_0,uid_1))
	#threads.append(Level_train_thread(config_instance,LogisticRegression(max_iter=1000,class_weight={0:4092,1:27908},n_jobs=-1,solver='lbfgs',verbose=2),level,ftype+'weighted_lr_lbfgs',X_0,X_1,uid_0,uid_1))
	#threads.append(Level_train_thread(config_instance,LogisticRegression(max_iter=1000,class_weight={0:4092,1:27908},n_jobs=-1,solver='liblinear',verbose=2),level,ftype+'weighted_lr_liblinear',X_0,X_1,uid_0,uid_1))
	
	threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=100,max_depth=8,min_samples_split=9),level,ftype+'_rf100',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_split=9),level,ftype+'_rf200',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=500,max_depth=8,min_samples_split=9),level,ftype+'_rf500',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=1000,max_depth=8,min_samples_split=9),level,ftype+'_rf1000',X_0,X_1,uid_0,uid_1))

	#try to assign class weights to RandomForestClassifier, not working
	#threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=100,class_weight={0:4092,1:27908},n_jobs=-1,max_depth=3,min_samples_split=7),level,ftype+'shorter_weighted_rf100',X_0,X_1,uid_0,uid_1))
	#threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=200,class_weight={0:4092,1:27908},n_jobs=-1,max_depth=5,min_samples_split=8),level,ftype+'shorter_weighted_rf200',X_0,X_1,uid_0,uid_1))
	#threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=500,class_weight={0:4092,1:27908},n_jobs=-1,max_depth=6,min_samples_split=9),level,ftype+'shorter_weighted_rf500',X_0,X_1,uid_0,uid_1))
	#threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=1000,class_weight={0:4092,1:27908},n_jobs=-1,max_depth=7,min_samples_split=9),level,ftype+'shorter_weighted_rf1000',X_0,X_1,uid_0,uid_1))

	threads.append(Level_train_thread(config_instance,GradientBoostingClassifier(n_estimators=20,max_depth=8,min_samples_split=9,learning_rate=0.02,subsample=0.7),level,ftype+'_gbdt20',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,GradientBoostingClassifier(n_estimators=50,max_depth=8,min_samples_split=9,learning_rate=0.02,subsample=0.7),level,ftype+'_gbdt50',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,GradientBoostingClassifier(n_estimators=100,max_depth=8,min_samples_split=9,learning_rate=0.02,subsample=0.7),level,ftype+'_gbdt100',X_0,X_1,uid_0,uid_1))

	#threads.append(Level_train_thread(config_instance,GradientBoostingClassifier(n_estimators=200,max_depth=8,min_samples_split=9,learning_rate=0.02,subsample=0.7),level,ftype+'more_esti_gbdt200',X_0,X_1,uid_0,uid_1))

	threads.append(Level_train_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=8,min_samples_split=9),n_estimators=20,learning_rate=0.02),level,ftype+'_ada20',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=8,min_samples_split=9),n_estimators=50,learning_rate=0.02),level,ftype+'_ada50',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=8,min_samples_split=2),n_estimators=100,learning_rate=0.02),level,ftype+'_ada100',X_0,X_1,uid_0,uid_1))

	threads.append(Level_train_thread(config_instance,BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=8,min_samples_split=9),n_estimators=20),level,ftype+'_bag20',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=8,min_samples_split=4),n_estimators=50),level,ftype+'_bag50',X_0,X_1,uid_0,uid_1))

	params={
	    'booster':'gbtree',
	    'objective': 'binary:logistic',
	   	'scale_pos_weight':28000/4000.0,
	    'eval_metric': 'auc',
	    'gamma':0,
	    'max_depth':8,
	    'lambda':700,
	    'subsample':0.7,
	    'colsample_bytree':0.3,
	    'min_child_weight':5,
	    'eta': 0.02,
	    'seed':7,
	    }
    #for all classifiers of xgb, call Xgb_level_train_thread and call Mboost.xgb_level_train
	threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb1000',X_0,X_1,uid_0,uid_1,params,1000))
	threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2000',X_0,X_1,uid_0,uid_1,params,2000))
	threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2500',X_0,X_1,uid_0,uid_1,params,2500))

	params['scale_pos_weight']=28000/2000.0
	threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb1000_2',X_0,X_1,uid_0,uid_1,params,500))
	threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2000_2',X_0,X_1,uid_0,uid_1,params,1000))
	threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2500_2',X_0,X_1,uid_0,uid_1,params,2500))

	params['colsample_bytree']=0.6
	threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb1000_3',X_0,X_1,uid_0,uid_1,params,1000))
	threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2000_3',X_0,X_1,uid_0,uid_1,params,2000))
	threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2500_3',X_0,X_1,uid_0,uid_1,params,2500))

	params['eta']=0.005
	threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb1000_4',X_0,X_1,uid_0,uid_1,params,500))
	threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2000_4',X_0,X_1,uid_0,uid_1,params,2000))
	threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2500_4',X_0,X_1,uid_0,uid_1,params,2500))

	params['eta']=0.01
	params['max_depth']=7
	threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb1000_5',X_0,X_1,uid_0,uid_1,params,1000))
	threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2000_5',X_0,X_1,uid_0,uid_1,params,2000))
	threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2500_5',X_0,X_1,uid_0,uid_1,params,2500))

	params['max_depth']=9
	threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb1000_6',X_0,X_1,uid_0,uid_1,params,1000))
	threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2000_6',X_0,X_1,uid_0,uid_1,params,2000))
	threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2500_6',X_0,X_1,uid_0,uid_1,params,2500))
	
	for thread in threads:
		thread.run()
        
###########################################################################
#                                                                         #
#                           level_two_wrapper                             # 
#                                                                         #
###########################################################################         

#partial stacking ensemble
#take subsample from function level_data_part from Class Load_train_data, use DBM to train the subsample
def level_two_wrapper():
	ftype=''
	config_instance=Config(ftype)
	level='level_two'#in which level right now is
    
    #for normal stacking (call level_data)
	clf_name_level_data=[
		  ftype+'_lr_sag',
		  ftype+'_lr_newton',
		  ftype+'_lr_lbfgs',
		  ftype+'_lr_liblinear',
		  ftype+'_rf100',
		  ftype+'_rf200',
		  ftype+'_rf500',
		  ftype+'_rf1000',
		  ftype+'_gbdt20',
		  ftype+'_gbdt50',
		  ftype+'_gbdt100',
		  ftype+'_ada20',
		  ftype+'_ada50',
		  ftype+'_ada100',
		  ftype+'_xgb2000',
		  ftype+'_xgb2500',
		  ftype+'_xgb2000_2',
		  ftype+'_xgb2500_2'
	]

     #for partial stacking model, call level_data_part, select the BBM, in this case xgb2000
	clf_name=[
		# ftype+'_lr_sag',
		# ftype+'_lr_newton',
		# ftype+'_lr_lbfgs',
		# ftype+'_lr_liblinear',
		# ftype+'_rf100',
		# ftype+'_rf200',
		# ftype+'_rf500',
		# ftype+'_rf1000',
		# ftype+'_gbdt20',
		# ftype+'_gbdt50',
		# ftype+'_gbdt100',
		# ftype+'_ada20',
		# ftype+'_ada50',
		# ftype+'_ada100',
		# ftype+'_bag20',
		# ftype+'_bag50',
		# ftype+'_xgb1000',
		 ftype+'_xgb2000',
		# ftype+'_xgb2500',
		# ftype+'_xgb1000_2',
		# ftype+'_xgb2000_2',
		# ftype+'_xgb2500_2',
		# ftype+'_xgb1000_3',
		# ftype+'_xgb2000_3',
		# ftype+'_xgb2500_3',
		# ftype+'_xgb2500_4',
		# ftype+'_xgb1000_5',
		# ftype+'_xgb2000_5',
		# ftype+'_xgb2500_5',
		# ftype+'_xgb1000_6',
		# ftype+'_xgb2000_6',
		# ftype+'_xgb2500_6'
	]
    ###############################################################################
    #                           choose one below to uncomment                     # 
    ############################################################################### 
     #normal stacking edition
	#load_data_instance=load_train_data.Load_train_data(config_instance,'level_one',clf_name_level_data)#instantiate Load_train_data
	#X_0,X_1,uid_0,uid_1=load_data_instance.level_data()#obtain output of level_one as training data for level two

     #partial stacking edition
	load_data_instance=load_train_data.Load_train_data(config_instance,'level_one',clf_name)#instantiate Load_train_data
	X_0,X_1,uid_0,uid_1=load_data_instance.level_data_part()#subsample of X where their risk to default are supposed to be overestimated by BBM

	threads=[]
	threads.append(Level_train_thread(config_instance,Ridge(),level,ftype+'_ridge',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,LogisticRegression(max_iter=1000,solver='sag'),level,ftype+'_lr_sag_1000',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,LogisticRegression(solver='sag'),level,ftype+'_lr_sag',X_0,X_1,uid_0,uid_1))

	threads.append(Level_train_thread(config_instance,LogisticRegression(solver='newton-cg'),level,ftype+'_lr_newton',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,LogisticRegression(solver='lbfgs'),level,ftype+'_lr_lbfgs',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,LogisticRegression(solver='liblinear'),level,ftype+'_lr_liblinear',X_0,X_1,uid_0,uid_1))
	
	# threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=100,max_depth=2,min_samples_split=10),level,ftype+'_rf100',X_0,X_1,uid_0,uid_1))
	# threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=200,max_depth=3,min_samples_split=10),level,ftype+'_rf200',X_0,X_1,uid_0,uid_1))
	# threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=500,max_depth=3,min_samples_split=10),level,ftype+'_rf500',X_0,X_1,uid_0,uid_1))
	#threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=1000,max_depth=8,min_samples_split=9),level,ftype+'_rf1000',X_0,X_1,uid_0,uid_1))

	# threads.append(Level_train_thread(config_instance,GradientBoostingClassifier(n_estimators=200,max_depth=3,min_samples_split=15,learning_rate=0.005,subsample=0.7),level,ftype+'_gbdt20',X_0,X_1,uid_0,uid_1))
	# threads.append(Level_train_thread(config_instance,GradientBoostingClassifier(n_estimators=50,max_depth=3,min_samples_split=15,learning_rate=0.01,subsample=0.7),level,ftype+'_gbdt50',X_0,X_1,uid_0,uid_1))
	# threads.append(Level_train_thread(config_instance,GradientBoostingClassifier(n_estimators=100,max_depth=3,min_samples_split=15,learning_rate=0.01,subsample=0.7),level,ftype+'_gbdt100',X_0,X_1,uid_0,uid_1))

	# threads.append(Level_train_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=3,min_samples_split=10),n_estimators=20,learning_rate=0.001),level,ftype+'_ada20',X_0,X_1,uid_0,uid_1))
	# threads.append(Level_train_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=3,min_samples_split=10),n_estimators=50,learning_rate=0.02),level,ftype+'_ada50',X_0,X_1,uid_0,uid_1))
	# threads.append(Level_train_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=3,min_samples_split=10),n_estimators=100,learning_rate=0.02),level,ftype+'_ada100',X_0,X_1,uid_0,uid_1))

	# threads.append(Level_train_thread(config_instance,BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=3,min_samples_split=9),n_estimators=20),level,ftype+'_bag20',X_0,X_1,uid_0,uid_1))
	# threads.append(Level_train_thread(config_instance,BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=3,min_samples_split=9),n_estimators=50),level,ftype+'_bag50',X_0,X_1,uid_0,uid_1))

	params={
	    'booster':'gbtree',
	    'objective': 'binary:logistic',
	   	'scale_pos_weight':28000/4000.0,
	    'eval_metric': 'auc',
	    'gamma':0,
	    'max_depth':8,
	    'lambda':700,
	    'subsample':0.7,
	    'colsample_bytree':1,
	    'min_child_weight':5,
	    'eta': 0.001,
	    'seed':1,
	    'nthread':10
	    }
	#threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb1000',X_0,X_1,uid_0,uid_1,params,1000))
	#threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2000',X_0,X_1,uid_0,uid_1,params,2000))    
	#threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2500',X_0,X_1,uid_0,uid_1,params,2500))

	# params['scale_pos_weight']=28000/2000.0
	# threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb1000_2',X_0,X_1,uid_0,uid_1,params,1000))
	# threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2000_2',X_0,X_1,uid_0,uid_1,params,2000))
	# threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2500_2',X_0,X_1,uid_0,uid_1,params,2500))

	# params['eta']=0.001
	# threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb1000_3',X_0,X_1,uid_0,uid_1,params,1000))
	# threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2000_3',X_0,X_1,uid_0,uid_1,params,2000))
	# threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2500_3',X_0,X_1,uid_0,uid_1,params,2500))

	# params['eta']=0.005
	# threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb1000_4',X_0,X_1,uid_0,uid_1,params,1000))
	# threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2000_4',X_0,X_1,uid_0,uid_1,params,2000))
	# threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2500_4',X_0,X_1,uid_0,uid_1,params,2500))

	# params['eta']=0.002
	# params['max_depth']=4
	# threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb1000_5',X_0,X_1,uid_0,uid_1,params,1000))
	# threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2000_5',X_0,X_1,uid_0,uid_1,params,2000))
	# threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2500_5',X_0,X_1,uid_0,uid_1,params,2500))

	# params['max_depth']=5
	# threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb1000_6',X_0,X_1,uid_0,uid_1,params,1000))
	# threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2000_6',X_0,X_1,uid_0,uid_1,params,2000))
	# threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2500_6',X_0,X_1,uid_0,uid_1,params,2500))

	for thread in threads:
		thread.run()
        
###########################################################################
#                                                                         #
#                          level_three_wrapper                            # 
#                                                                         #
########################################################################### 
#level three model training

def level_three_wrapper():
	ftype=''
	config_instance=Config(ftype)
	level='level_three'
	types=[''
	]
	clf_name=[]
	for ftype2 in types:	
		clf_name2=[
			ftype+'_lr_sag',
			ftype+'_lr_newton',
			ftype+'_lr_lbfgs',
			ftype+'_lr_liblinear',
			ftype+'_rf100',
			ftype+'_rf200',
			ftype+'_rf500',
			ftype+'_rf1000',
			ftype+'_gbdt20',
			ftype+'_gbdt50',
			ftype+'_gbdt100',
			ftype+'_ada20',
			ftype+'_ada50',
			ftype+'_ada100',
			ftype+'_bag20',
			ftype+'_bag50',
			ftype+'_xgb1000',
			ftype+'_xgb2000',
			ftype+'_xgb2500',
			ftype+'_xgb1000_2',
			ftype+'_xgb2000_2',
			ftype+'_xgb2500_2',
			ftype+'_xgb1000_3',
			ftype+'_xgb2000_3',
			ftype+'_xgb2500_3',
			ftype+'_xgb2500_4',
			ftype+'_xgb1000_5',
			ftype+'_xgb2000_5',
			ftype+'_xgb2500_5',
			ftype+'_xgb1000_6',
			ftype+'_xgb2000_6',
			ftype+'_xgb2500_6'
		]
		clf_name.extend(clf_name2)

	load_data_instance=load_train_data.Load_train_data(config_instance,'level_two',clf_name)
	X_0,X_1,uid_0,uid_1=load_data_instance.level_data()

	threads=[]
	threads.append(Level_train_thread(config_instance,LogisticRegression(solver='sag'),level,ftype+'_lr_sag',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,LogisticRegression(solver='newton-cg'),level,ftype+'_lr_newton',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,LogisticRegression(solver='lbfgs'),level,ftype+'_lr_lbfgs',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,LogisticRegression(solver='liblinear'),level,ftype+'_lr_liblinear',X_0,X_1,uid_0,uid_1))
	
	threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=100,max_depth=3,min_samples_split=10),level,ftype+'_rf100',X_0,X_1,uid_0,uid_1))

	threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=200,max_depth=3,min_samples_split=10),level,ftype+'_rf200',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=500,max_depth=3,min_samples_split=10),level,ftype+'_rf500',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,RandomForestClassifier(n_estimators=1000,max_depth=3,min_samples_split=10),level,ftype+'_rf1000',X_0,X_1,uid_0,uid_1))

	threads.append(Level_train_thread(config_instance,GradientBoostingClassifier(n_estimators=20,max_depth=3,min_samples_split=15,learning_rate=0.01,subsample=0.7),level,ftype+'_gbdt20',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,GradientBoostingClassifier(n_estimators=50,max_depth=3,min_samples_split=15,learning_rate=0.01,subsample=0.7),level,ftype+'_gbdt50',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,GradientBoostingClassifier(n_estimators=100,max_depth=3,min_samples_split=15,learning_rate=0.01,subsample=0.7),level,ftype+'_gbdt100',X_0,X_1,uid_0,uid_1))

	threads.append(Level_train_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=2,min_samples_split=10),n_estimators=20,learning_rate=0.02),level,ftype+'_ada20',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=3,min_samples_split=10),n_estimators=50,learning_rate=0.02),level,ftype+'_ada50',X_0,X_1,uid_0,uid_1))
	threads.append(Level_train_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=3,min_samples_split=10),n_estimators=100,learning_rate=0.02),level,ftype+'_ada100',X_0,X_1,uid_0,uid_1))

	params={
	    'booster':'gbtree',
	    'objective': 'binary:logistic',
	   	'scale_pos_weight':28000/4000.0,
	    'eval_metric': 'auc',
	    'gamma':0,
	    'max_depth':3,
	    'lambda':700,
	    'subsample':0.9,
	    'colsample_bytree':0.3,
	    'min_child_weight':5,
	    'eta': 0.0005,
	    'seed':1,
	    'nthread':8
	    }
	threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2000',X_0,X_1,uid_0,uid_1,params,2000))
	threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2500',X_0,X_1,uid_0,uid_1,params,2500))
	params={
	    'booster':'gbtree',
	    'objective': 'binary:logistic',
	   	'scale_pos_weight':28000/4000.0,
	    'eval_metric': 'auc',
	    'gamma':0,
	    'max_depth':3,
	    'lambda':700,
	    'subsample':0.9,
	    'colsample_bytree':0.3,
	    'min_child_weight':5,
	    'eta': 0.0005,
	    'seed':1,
	    'nthread':8
	    }
	threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2000_2',X_0,X_1,uid_0,uid_1,params,2000))
	threads.append(Xgb_level_train_thread(config_instance,level,ftype+'_xgb2500_2',X_0,X_1,uid_0,uid_1,params,2500))

	for thread in threads:
		thread.run()
	pass

#predict

###########################################################################
#                                                                         #
#                            level_one_predict                            # 
#                                                                         #
########################################################################### 

def level_one_predict():
	ftype=''
	config_instance=Config(ftype)
	level='level_one'

	config_instance=Config('')#choose log_move transfered data

	load_data_instance=load_origin_data.Load_origin_data(config_instance)

    #for local test
	X,y,uid,X_0,X_1,uid_0,uid_1=load_data_instance.load_final()
	predict_X,predict_uid=load_data_instance.load_final_test()#this is test set
    
    #uncomment the three lines of code when locan verification is in need
    #comment them when you do final prediction and testing
    #scripts below are loading data which is splited into train and validation locally	
	#take 20% of training data as validation set to do local training
	#X_0,test_X_0,X_1,test_X_1,uid_0,test_uid_0,uid_1,test_uid_1=load_data_instance.train_test_xy()
	#predict_X=np.vstack((test_X_0,test_X_1))
	#predict_uid=np.hstack((test_uid_0,test_uid_1))

	threads=[]
     #for all classifiers except for xgb, call Level_predict_thread and call Mboost.level_predict
	threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='sag'),level,ftype+'_lr_sag',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,LogisticRegression(max_iter=1000,solver='sag'),level,ftype+'_lr_sag_1000',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,LogisticRegression(max_iter=1500,solver='sag'),level,ftype+'_lr_sag_1500',X_0,X_1,predict_X,predict_uid))


	threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='newton-cg'),level,ftype+'_lr_newton',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='lbfgs'),level,ftype+'_lr_lbfgs',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='liblinear'),level,ftype+'_lr_liblinear',X_0,X_1,predict_X,predict_uid))

	threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=100,max_depth=8,min_samples_split=9),level,ftype+'_rf100',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_split=9),level,ftype+'_rf200',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=500,max_depth=8,min_samples_split=9),level,ftype+'_rf500',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=1000,max_depth=8,min_samples_split=9),level,ftype+'_rf1000',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,GradientBoostingClassifier(n_estimators=20,max_depth=8,min_samples_split=9,learning_rate=0.02,subsample=0.7),level,ftype+'_gbdt20',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,GradientBoostingClassifier(n_estimators=50,max_depth=8,min_samples_split=9,learning_rate=0.02,subsample=0.7),level,ftype+'_gbdt50',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,GradientBoostingClassifier(n_estimators=100,max_depth=8,min_samples_split=9,learning_rate=0.02,subsample=0.7),level,ftype+'_gbdt100',X_0,X_1,predict_X,predict_uid))

	threads.append(Level_predict_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=8,min_samples_split=9),n_estimators=20,learning_rate=0.02),level,ftype+'_ada20',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=8,min_samples_split=9),n_estimators=50,learning_rate=0.02),level,ftype+'_ada50',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=8,min_samples_split=9),n_estimators=100,learning_rate=0.02),level,ftype+'_ada100',X_0,X_1,predict_X,predict_uid))

	threads.append(Level_predict_thread(config_instance,BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=8,min_samples_split=9),n_estimators=20),level,ftype+'_bag20',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=8,min_samples_split=9),n_estimators=50),level,ftype+'_bag50',X_0,X_1,predict_X,predict_uid))

	params={
	    'booster':'gbtree',
	    'objective': 'binary:logistic',
	   	'scale_pos_weight':28000/4000.0,
	    'eval_metric': 'auc',
	    'gamma':0,
	    'max_depth':8,
	    'lambda':700,
	    'subsample':0.7,
	    'colsample_bytree':0.3,
	    'min_child_weight':5,
	    'eta': 0.02,
	    'seed':7,
	    'nthread':8
	    }
    
    #for all classifiers of xgb, call Xgb_level_predict_thread and call Mboost.output_level_predict
	threads.append(Xgb_level_predict_thread(config_instance,level,ftype+'_xgb1000',X_0,X_1,predict_X,predict_uid,params,1000))
	threads.append(Xgb_level_predict_thread(config_instance,level,ftype+'_xgb2000',X_0,X_1,predict_X,predict_uid,params,2000))
	threads.append(Xgb_level_predict_thread(config_instance,level,ftype+'_xgb2500',X_0,X_1,predict_X,predict_uid,params,2500))

	params['scale_pos_weight']=28000/2000.0
	threads.append(Xgb_level_predict_thread(config_instance,level,ftype+'_xgb1000_2',X_0,X_1,predict_X,predict_uid,params,1000))
	threads.append(Xgb_level_predict_thread(config_instance,level,ftype+'_xgb2000_2',X_0,X_1,predict_X,predict_uid,params,2000))
	threads.append(Xgb_level_predict_thread(config_instance,level,ftype+'_xgb2500_2',X_0,X_1,predict_X,predict_uid,params,2500))

	params['colsample_bytree']=0.6
	threads.append(Xgb_level_predict_thread(config_instance,level,ftype+'_xgb1000_3',X_0,X_1,predict_X,predict_uid,params,1000))
	threads.append(Xgb_level_predict_thread(config_instance,level,ftype+'_xgb2000_3',X_0,X_1,predict_X,predict_uid,params,2000))
	threads.append(Xgb_level_predict_thread(config_instance,level,ftype+'_xgb2500_3',X_0,X_1,predict_X,predict_uid,params,2500))

	params['eta']=0.005
	threads.append(Xgb_level_predict_thread(config_instance,level,ftype+'_xgb1000_4',X_0,X_1,predict_X,predict_uid,params,1000))
	threads.append(Xgb_level_predict_thread(config_instance,level,ftype+'_xgb2000_4',X_0,X_1,predict_X,predict_uid,params,2000))
	threads.append(Xgb_level_predict_thread(config_instance,level,ftype+'_xgb2500_4',X_0,X_1,predict_X,predict_uid,params,2500))

	params['eta']=0.01
	params['max_depth']=7
	threads.append(Xgb_level_predict_thread(config_instance,level,ftype+'_xgb1000_5',X_0,X_1,predict_X,predict_uid,params,1000))
	threads.append(Xgb_level_predict_thread(config_instance,level,ftype+'_xgb2000_5',X_0,X_1,predict_X,predict_uid,params,2000))
	threads.append(Xgb_level_predict_thread(config_instance,level,ftype+'_xgb2500_5',X_0,X_1,predict_X,predict_uid,params,2500))

	params['max_depth']=9
	threads.append(Xgb_level_predict_thread(config_instance,level,ftype+'_xgb1000_6',X_0,X_1,predict_X,predict_uid,params,1000))
	threads.append(Xgb_level_predict_thread(config_instance,level,ftype+'_xgb2000_6',X_0,X_1,predict_X,predict_uid,params,2000))
	threads.append(Xgb_level_predict_thread(config_instance,level,ftype+'_xgb2500_6',X_0,X_1,predict_X,predict_uid,params,2500))
	for thread in threads:
		thread.run()
        
###########################################################################
#                                                                         #
#                            level_two_predict                            # 
#                                                                         #
########################################################################### 
def level_two_predict():
	config_instance=Config('')
	level='level_two'
	ftype=''
	clf_name=[
		ftype+'_lr_sag',
		ftype+'_lr_newton',
		ftype+'_lr_lbfgs',
		ftype+'_lr_liblinear',
		ftype+'_xgb1000',
		ftype+'_xgb2000',
		ftype+'_xgb2500',
		ftype+'_xgb1000_2',
		ftype+'_xgb2000_2',
		ftype+'_xgb2500_2',
		ftype+'_xgb1000_3',
		ftype+'_xgb2000_3',
		ftype+'_xgb2500_3',
		ftype+'_xgb1000_4',
		ftype+'_xgb2000_4',
		ftype+'_xgb2500_4',
		ftype+'_xgb1000_5',
		ftype+'_xgb2000_5',
		ftype+'_xgb2500_5',
		ftype+'_xgb1000_6',
		ftype+'_xgb2000_6',
		ftype+'_xgb2500_6'
	]
    
    #from load_train_data.py module Load Load_train_data class
    #this class has three parameters: config,level,clf_name
    #initialize by calling the class and asgin parameters vales
    #config=config_instance, level='level_one', clf_name=clf_name
	load_data_instance=load_train_data.Load_train_data(config_instance,'level_one',clf_name)
    
    #from load_predict_data.py module Load Load_predict_data class
    #this class has three parameters: config,level,clf_name
    #initialize by calling the class and asgin parameters vales
    #config=config_instance, level='level_one', clf_name=clf_name
	predict_data_instance=load_predict_data.Load_predict_data(config_instance,'level_one',clf_name)
    
    #for local_verify
	X_0,X_1,uid_00,uid_11=load_data_instance.level_data()
	predict_X,predict_uid=predict_data_instance.level_data()



	threads=[]
	threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='sag'),level,'_lr_sag',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,LogisticRegression(max_iter=1000,solver='sag'),level,ftype+'_lr_sag_1000',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,LogisticRegression(max_iter=1500,solver='sag'),level,ftype+'_lr_sag_1500',X_0,X_1,predict_X,predict_uid))

	threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='newton-cg'),level,'_lr_newton',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='lbfgs'),level,'_lr_lbfgs',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='liblinear'),level,'_lr_liblinear',X_0,X_1,predict_X,predict_uid))

	threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=100,max_depth=3,min_samples_split=10),level,'_rf100',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=200,max_depth=3,min_samples_split=10),level,'_rf200',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=500,max_depth=3,min_samples_split=10),level,'_rf500',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=100,max_depth=3,min_samples_split=10),level,'_rf1000',X_0,X_1,predict_X,predict_uid))

	threads.append(Level_predict_thread(config_instance,GradientBoostingClassifier(n_estimators=20,max_depth=3,min_samples_split=15,learning_rate=0.01,subsample=0.7),level,'_gbdt20',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,GradientBoostingClassifier(n_estimators=50,max_depth=3,min_samples_split=15,learning_rate=0.01,subsample=0.7),level,'_gbdt50',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,GradientBoostingClassifier(n_estimators=100,max_depth=3,min_samples_split=15,learning_rate=0.01,subsample=0.7),level,'_gbdt100',X_0,X_1,predict_X,predict_uid))

	threads.append(Level_predict_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=5,max_depth=3,min_samples_split=10),n_estimators=20,learning_rate=0.02),level,'_ada20',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=5,max_depth=3,min_samples_split=10),n_estimators=50,learning_rate=0.02),level,'_ada50',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=5,max_depth=3,min_samples_split=10),n_estimators=100,learning_rate=0.02),level,'_ada100',X_0,X_1,predict_X,predict_uid))

	params={
	    'booster':'gbtree',
	    'objective': 'binary:logistic',
	   	'scale_pos_weight':28000/4000.0,
	    'eval_metric': 'auc',
	    'gamma':15,
	    'max_depth':3,
	    'lambda':600,
	    'subsample':0.40,
	    'colsample_bytree':0.3,
	    'min_child_weight':10,
	    'eta': 0.002,#0.0005
	    'seed':1,
	    'nthread':8
	    }

	threads.append(Xgb_level_predict_thread(config_instance,level,'_xgb2000',X_0,X_1,predict_X,predict_uid,params,2000))
	threads.append(Xgb_level_predict_thread(config_instance,level,'_xgb2500',X_0,X_1,predict_X,predict_uid,params,2500))
	params={
	    'booster':'gbtree',
	    'objective': 'binary:logistic',
	   	'scale_pos_weight':28000/4000.0,
	    'eval_metric': 'auc',
	    'gamma':0,
	    'max_depth':3,
	    'lambda':700,
	    'subsample':0.9,
	    'colsample_bytree':0.3,
	    'min_child_weight':5,
	    'eta': 0.0005,
	    'seed':1,
	    'nthread':8
	    }
	threads.append(Xgb_level_predict_thread(config_instance,level,'_xgb2000_2',X_0,X_1,predict_X,predict_uid,params,2000))
	threads.append(Xgb_level_predict_thread(config_instance,level,'_xgb2500_2',X_0,X_1,predict_X,predict_uid,params,2500))
	for thread in threads:
		thread.run()
	pass

###########################################################################
#                                                                         #
#                          level_three_predict                            # 
#                                                                         #
########################################################################### 
def level_three_predict():
	config_instance=Config('')
	level='level_three'
	clf_name=[
		'log_move_lr_sag',
		'log_move_lr_newton',
		'log_move_lr_lbfgs',
		'log_move_lr_liblinear',
		'log_move_rf100',
		'log_move_rf200',
		'log_move_rf500',
		'log_move_rf1000',
		'log_move_gbdt20',
		'log_move_gbdt50',
		'log_move_gbdt100',
		'log_move_ada20',
		'log_move_ada50',
		'log_move_ada100',
		'log_move_xgb2000',
		'log_move_xgb2500',
		'log_move_xgb2000_2',
		'log_move_xgb2500_2'
	]

	load_data_instance=load_train_data.Load_train_data(config_instance,'level_two',clf_name)
	predict_data_instance=load_predict_data.Load_predict_data(config_instance,'level_two',clf_name)
	X_0,X_1,uid_0,uid_1=load_data_instance.level_data()
	predict_X,predict_uid=predict_data_instance.level_data()

	threads=[]
	threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='sag'),level,'log_move_lr_sag',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='newton-cg'),level,'log_move_lr_newton',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='lbfgs'),level,'log_move_lr_lbfgs',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,LogisticRegression(solver='liblinear'),level,'log_move_lr_liblinear',X_0,X_1,predict_X,predict_uid))

	threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=100,max_depth=3,min_samples_split=10),level,'log_move_rf100',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=200,max_depth=3,min_samples_split=10),level,'log_move_rf200',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=500,max_depth=3,min_samples_split=10),level,'log_move_rf500',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,RandomForestClassifier(n_estimators=1000,max_depth=3,min_samples_split=10),level,'log_move_rf1000',X_0,X_1,predict_X,predict_uid))

	threads.append(Level_predict_thread(config_instance,GradientBoostingClassifier(n_estimators=20,max_depth=3,min_samples_split=15,learning_rate=0.01,subsample=0.7),level,'log_move_gbdt20',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,GradientBoostingClassifier(n_estimators=50,max_depth=3,min_samples_split=15,learning_rate=0.01,subsample=0.7),level,'log_move_gbdt50',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,GradientBoostingClassifier(n_estimators=100,max_depth=3,min_samples_split=15,learning_rate=0.01,subsample=0.7),level,'log_move_gbdt100',X_0,X_1,predict_X,predict_uid))

	threads.append(Level_predict_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=3,min_samples_split=10),n_estimators=20,learning_rate=0.02),level,'log_move_ada20',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=3,min_samples_split=10),n_estimators=50,learning_rate=0.02),level,'log_move_ada50',X_0,X_1,predict_X,predict_uid))
	threads.append(Level_predict_thread(config_instance,AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=50,max_depth=3,min_samples_split=10),n_estimators=100,learning_rate=0.02),level,'log_move_ada100',X_0,X_1,predict_X,predict_uid))

	params={
	    'booster':'gbtree',
	    'objective': 'binary:logistic',
	   	'scale_pos_weight':28000/4000.0,
	    'eval_metric': 'auc',
	    'gamma':0,
	    'max_depth':3,
	    'lambda':700,
	    'subsample':0.9,
	    'colsample_bytree':0.3,
	    'min_child_weight':5,
	    'eta': 0.0005,
	    'seed':1,
	    'nthread':8
	    }
	threads.append(Xgb_level_predict_thread(config_instance,level,'log_move_xgb2000',X_0,X_1,predict_X,predict_uid,params,2000))
	threads.append(Xgb_level_predict_thread(config_instance,level,'log_move_xgb2500',X_0,X_1,predict_X,predict_uid,params,2500))
	params={
	    'booster':'gbtree',
	    'objective': 'binary:logistic',
	   	'scale_pos_weight':28000/4000.0,
	    'eval_metric': 'auc',
	    'gamma':0,
	    'max_depth':3,
	    'lambda':700,
	    'subsample':0.9,
	    'colsample_bytree':0.3,
	    'min_child_weight':5,
	    'eta': 0.0005,
	    'seed':1,
	    'nthread':8
	    }
	threads.append(Xgb_level_predict_thread(config_instance,level,'log_move_xgb2000_2',X_0,X_1,predict_X,predict_uid,params,2000))
	threads.append(Xgb_level_predict_thread(config_instance,level,'log_move_xgb2500_2',X_0,X_1,predict_X,predict_uid,params,2500))
	for thread in threads:
		thread.run()
	pass

#start different tasks here, one by one
def main():

	#level_one_wrapper()
    	#level_one_predict()
	level_two_wrapper()
	#level_two_predict()
	#level_three_wrapper()
	#level_three_predict()

if __name__ == '__main__':
	start=datetime.now()
	main()
	end=datetime.now()
	print( "All Run time:"+str(float((end-start).seconds)/60.0)+"min / "+str(float((end-start).seconds))+"s")
    
