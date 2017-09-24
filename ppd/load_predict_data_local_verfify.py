#! /usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import time

import load_origin_data
from config import Config

class Load_predict_data(object):
	"""
	:class Load_predict_data
     read the prediction result of No.2-n level, use the prediction result as features for next level
	"""
	def __init__(self,config,level,clf_name):
		"""
		:type config: Config 
		:type level: str which level of result to read
		:type clf_name: List[str] name of classifiers on the last level
		"""
		self.config=config
		self.level=level
		self.__clf_name=clf_name

	def load_clf_file(self,level,name):
		"""
        	:type level: str, which leval of data to read
		:type name: str, name of classifier
		read the output of classifers on the last level as a column of dataset in this level
         and do log transfermation to the last-level result so as to have more stable data to use in this level
         return a dict that contains info of log transformed predict score from last level
		"""
		reader=pd.read_csv(self.config.path_predict+level+'/'+name+'.csv',iterator=False,delimiter=',')
		d={}
		for i in range(len(reader['uid'])):
			d[reader['uid'][i]]=np.log10(reader['score'][i])
			#d[reader['uid'][i]]=reader['score'][i]
		return d

	def level_data(self):
		"""
         read the result from last level as the features for the next level
		"""
		level=self.level
		clf_name=self.__clf_name
		config_instance=Config('')#choose log_move transfered data
		load_data_instance=load_origin_data.Load_origin_data(config_instance)

		X,uids=load_data_instance.load_final_test()
		print('X shape: ',X.shape,'uids length',uids.shape)


		d={}
		for name in clf_name:
			column_dict=self.load_clf_file(level,name)
			for uid in uids:
				temp=d.get(uid,[])
				temp.append(column_dict[uid])#change fromtemp.append(column_dict[uid])
				d[uid]=temp
		
		X=[]
		for i in range(len(uids)):
			X.append(d[uids[i]])

		return np.array(X),np.array(uids)

def main():

	config_instance=Config('')
	level='level_one'
	clf_name=[
		'_lr_sag',
		'_lr_newton',
		'_lr_lbfgs',
		'_lr_liblinear',
	#	'_rf100',
	#	'_rf200',
	#	'_rf500',
	##	'_rf1000',
	#	'_gbdt20',
	#	'_gbdt50',
	#	'_gbdt100',
	#	'_ada20',
	#	'_ada50',
	#	'_ada100',
	#	'_xgb2000',
	#	'_xgb2500',
	#	'_xgb2000_2',
	#	'_xgb2500_2'
	]
	predict_data_instance=Load_predict_data(config_instance,level,clf_name)
	predict_X,predict_uid=predict_data_instance.level_data()
	print (predict_X.shape,predict_uid.shape)
	pd.DataFrame(predict_X).to_csv('predict_X.csv',index=None)#to check if it works well
	pd.DataFrame(predict_uid).to_csv('predict_uid.csv',index=None)#to check if it works well

    

	pass

if __name__ == '__main__':
	main()