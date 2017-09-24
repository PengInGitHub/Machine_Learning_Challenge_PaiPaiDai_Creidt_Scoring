#! /usr/bin/env python
# -*- coding:utf-8 -*-

import sys
import os

from config import Config
import threading
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

class Mboost_thread(threading.Thread):
	def __init__(self,clf,x_train, y_train, x_test, y_test, test_uid):
		threading.Thread.__init__(self)
		self.clf=clf
		self.x_train=x_train
		self.y_train=y_train
		self.x_test=x_test
		self.y_test=y_test
		self.test_uid=test_uid

		self.predict=[0 for i in range(len(test_uid))]
		self.auc_score=0
		pass

	def run(self):
		self._run()
		pass

	def _run(self):
		clf=self.clf
		x_train=self.x_train
		y_train=self.y_train
		x_test=self.x_test
		y_test=self.y_test
		test_uid=self.test_uid

		clf.fit(x_train,y_train)
		try:
			#output probability
			y_pred=clf.predict_proba(x_test)
			#print y_pred
			y_pred=y_pred[:,1]
		except:	
			#output for LR that has probability as output directly
			y_pred=clf.predict(x_test)

		#calculate AUC in each fold
		auc_score=metrics.roc_auc_score(y_test,y_pred)
		self.predict=y_pred
		self.auc_score=auc_score
		print (auc_score	)

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


		
