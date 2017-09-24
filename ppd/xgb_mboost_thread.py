#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os

from config import Config
import threading
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import xgboost as xgb
class Xgb_mboost_thread(threading.Thread):
	def __init__(self,x_train, y_train, x_test, y_test, test_uid,watchlist, params, round):
		threading.Thread.__init__(self)
		self.x_train=x_train
		self.y_train=y_train
		self.x_test=x_test
		self.y_test=y_test
		self.test_uid=test_uid

		self.watchlist=watchlist
		self.params=params
		self.round=round

		self.predict=[]
		self.auc_score=0
		pass

	def run(self):
		self._run()
		pass

	def _run(self):
		x_train=self.x_train
		y_train=self.y_train
		x_test=self.x_test
		y_test=self.y_test
		test_uid=self.test_uid
		watchlist=self.watchlist
		params=self.params
		round=self.round

		model=xgb.train(params,x_train,num_boost_round=round,evals=watchlist,verbose_eval=20)
		y_pred=model.predict(x_test)

		#计算一折的AUC
		auc_score=metrics.roc_auc_score(y_test,y_pred)
		self.predict=y_pred
		self.auc_score=auc_score

def main():
	# threads=[]
	# for i in range(10):
	# 	threads.append(Mboost_thread(i))
	# for thread in threads:
	# 	thread.start()

	# for thread in threads:
	# 	thread.join()

	# for thread in threads:
	# 	print thread.x
	pass

if __name__ == '__main__':
	main()


		
