#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
"""
class Config
setting environment
"""
class Config():

	def __init__(self,scale):

		self.scale_level=scale#scale_level is the variable of type of preprocessing

		"""
         configure development environment
		"""
		self.path='/Users/pengchengliu/Documents/submit/data/ppd/'
		self.path_origin_train=self.path+'A_train.csv'#untouched original 

         #feature type
		self.path_feature_type=self.path+'feature_type.csv'#doc contains feature type info

         #train
		self.path_train_x=self.path+'train_x.csv'#train_x before preprocessing
		self.path_train_x_scaled=self.path+'train_x_scale_'+self.scale_level+'.csv' #train_x after preprocessing
       
         #predict
		self.path_predict_x=self.path+'test_x.csv'#test_x before preprocessing
		self.path_predict_x_scaled=self.path+'test_x_scale_'+self.scale_level+'.csv' #test_x after preprocessing

         #y
		self.path_train_y=self.path+'train_y.csv'
		self.path_predict_y=self.path+'test_y.csv'

         #uid
		self.path_uid=self.path+'uid.csv'

		"""
		Analysis:  analysis output
		"""
		self.path_analysis=self.path+'analysis/'


		"""
		Preprocessing: construct features
		"""
		self.path_location=self.path+'location/'
		self.path_coor=self.path+'location/coordinates/'

		"""
		fold random state
		"""
		self.fold_random_state=1
		self.n_folds=5

		"""
         output
		"""
		self.path_train  =self.path+'train/' #output of training
		self.path_predict=self.path+'predict_local/' #path for output result of testing
		self.path_verify=self.path+'verify/'
		self.path_cor=self.path+'statistic/cor_log.csv' #screen out features have correlation with target variable higher than 0.01

	def init_path(self):
		"""
		create folders
		"""
		paths=[self.path_train,self.path_predict,self.path_train+'level_one/',self.path+'statistic/',self.path_train+'level_two/',self.path_predict+'level_one/',self.path_predict+'level_two/']
		for path in paths:
			if not os.path.exists(path):
				os.mkdir(path)
def main():
	instance=Config('')
	instance.init_path()
	pass                

if __name__ == '__main__':
	main()