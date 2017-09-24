#! /usr/bin/env python
# -*- coding:utf-8 -*-

from datetime import datetime
import logging
import threading
import mboost

class Level_train_thread(threading.Thread):
	"""
	:class Level_train_thread
	:multi-threading for training
	"""
	def __init__(self,config,clf,level,name,X_0,X_1,uid_0,uid_1):
		threading.Thread.__init__(self)
		self.config=config#config_instance for instance
		self.clf=clf#LogisticRegression(solver='sag') for instance
		self.level=level#'level_one' for instance
		self.name=name#'log_move_lr_sag' for instance
		self.X_0=X_0#X_0 for instance
		self.X_1=X_1#X_1 for instance
		self.uid_0=uid_0#uid_0 for instance
		self.uid_1=uid_1#uid_1 for instance

	def run(self):
		logging.info('Begin train '+self.name)
		start=datetime.now()
		boost_instance=mboost.Mboost(self.config)
		boost_instance.level_train(self.clf,self.level,self.name,self.X_0,self.X_1,self.uid_0,self.uid_1)#initiate Mboost, call level_train method
		end=datetime.now()
		logging.info('End train '+self.name+", cost time: "+str(float((end-start).seconds)/60.0)+"min / "+str(float((end-start).seconds))+"s")