#! /usr/bin/env python
# -*- coding:utf-8 -*-

from datetime import datetime
import logging
import threading
import mboost


class Xgb_level_train_thread(threading.Thread):
	"""
	:class Level_train_thread
	:multi-threading for training
	"""
	def __init__(self,config,level,name,X_0,X_1,uid_0,uid_1,params,round):
		threading.Thread.__init__(self)
		self.config=config#variable explanation see level_train_thread.py
		self.level=level
		self.name=name
		self.X_0=X_0
		self.X_1=X_1
		self.uid_0=uid_0
		self.uid_1=uid_1
		self.params=params#meta parameters for xgb fitting
		self.round=round#num of iterations

	def run(self):
		logging.info('Begin train '+self.name)
		start=datetime.now()
		boost_instance=mboost.Mboost(self.config)
		boost_instance.xgb_level_train(self.level,self.name,self.X_0,self.X_1,self.uid_0,self.uid_1,self.params,self.round)
		end=datetime.now()
		logging.info('End train '+self.name+", cost time: "+str(float((end-start).seconds)/60.0)+"min / "+str(float((end-start).seconds))+"s")