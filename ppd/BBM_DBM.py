#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from config import Config
from minepy import MINE
from pandas import *
import numpy as np
#from libraries.settings import *
from scipy.stats.stats import pearsonr
import itertools
import csv
class BBM_DBM():
	"""
    choose BBM and DBM
	"""
	def __init__(self,config,level,clf_name):
		"""
		:type config: Config, configuration
		:type level: str, which leval of data to read
		:type clf_name: List[str], a collection of the names of classifiers in the last level
		"""
		self.config=config
		self.level=level
		self.__clf_name=clf_name

	def load_clf_file(self,level,name):

		values=pd.read_csv(self.config.path_train+level+'/'+name+'.csv',iterator=False,delimiter=',',header=None)[1].values

		return values

	def load_clf_score(self,level,name):
		"""
		:type level: str, which leval of data to read
		:type name: str, name of classifier
		read the AUC score of classifier's n-folds traning 
         return average score 
		"""
		reader=pd.read_csv(self.config.path_train+level+'/'+name+'_score.csv',iterator=False,delimiter=',',header=None)
		return np.mean(reader[0])

	def level_data(self):
		"""
        

		"""
		level=self.level
		clf_name=self.__clf_name

		column_important=[]
		result_matrix=[]


		for name in clf_name:

			
			values=self.load_clf_file(level,name)
			result_matrix.append(values)
            
			column_score=self.load_clf_score(level,name)
			column_important.append(column_score)

		print (len(result_matrix))

		return result_matrix,column_important




def  main():

	ftype=''
	config_instance=Config(ftype)
	level='level_one'
    
    #for normal stacking (call level_data)
	clf_name=[
		  '_lr_sag',
		  '_lr_sag_1000',
		  '_lr_sag_1500',
		  '_lr_newton',
		  '_lr_lbfgs',
		  '_lr_liblinear',
		  '_rf100',
		  '_rf200',
		  '_rf500',
		  '_rf1000',
		  '_gbdt20',
		  '_gbdt50',
		  '_gbdt100',
		  '_ada20',
		  '_ada50',
		  '_ada100',
		  '_xgb1000',
		  '_xgb1000_2',
		  '_xgb1000_3',
		  '_xgb1000_4',
		  '_xgb1000_5',
		  '_xgb1000_6',
		  '_xgb2000',
		  '_xgb2000_2',
		  '_xgb2000_3',
		  '_xgb2000_4',
		  '_xgb2000_5',
		  '_xgb2000_6',
		  '_xgb2500',
		  '_xgb2500_2',
		  '_xgb2500_3',
		  '_xgb2500_4',
		  '_xgb2500_5',
		  '_xgb2500_6',

	]
    
###########
#   BBM   #
###########
	bbm_dbm_instance=BBM_DBM(config_instance,level,clf_name)#instantiate class
	res,column_important=bbm_dbm_instance.level_data()#call level_data_part, return data and plot
	column_important,clf_name=zip(*sorted(zip(column_important,clf_name),reverse=True))#https://stackoverflow.com/questions/9764298/is-it-possible-to-sort-two-listswhich-reference-each-other-in-the-exact-same-w

	list(map(lambda X: print(X[0],X[1]), list(zip(clf_name,column_important)))) 


###########
#   DBM   #
###########
	res=list(map(list, zip(*res))) 

	df=pd.DataFrame(res,index=None) 
	df.columns=clf_name
    
	df.to_csv('train_level_one') 

	correlations = {} 
	columns = df.columns.tolist() 

	for col_a, col_b in itertools.combinations(columns, 2): 
		correlations[col_a + '__' + col_b] = pearsonr(df.loc[:, col_a], df.loc[:, col_b])

	result = DataFrame.from_dict(correlations, orient='index') 
	result.columns = ['PCC', 'p-value'] 
	result.to_csv('cor.csv')
 
	print(result.sort_index()) 
	pass

if __name__ == '__main__':
	main()
'''    
	cm = []
	for i in range(len(clf_name)):
		tmp = []
		for j in range(len(clf_name)):
			m = MINE()
			m.compute_score(res[i], res[j])
			tmp.append(m.mic())
		cm.append(tmp)
	
	def plot_confusion_matrix(cm, title, cmap=plt.cm.Blues):

    
		plt.imshow(cm, interpolation='nearest', cmap=cmap)
		plt.title(title)
		plt.colorbar()
		tick_marks = np.arange(len(clf_name))
		plt.xticks(tick_marks, clf_name, rotation=45)
		plt.yticks(tick_marks, clf_name)
		plt.tight_layout()            

	#plot_confusion_matrix(cm, title='mic')
	#plt.show()



'''	

	













    
    
        
        
        
             
     	

	

