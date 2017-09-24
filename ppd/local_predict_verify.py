#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime
import load_origin_data

from config import Config
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class Local_predict_verify():
	"""
	:class Local_predict_verify
	 verify and blend prediction results
     the major task of this class is to find out most appropriate intervals of samples for further blending
    
     In details, this class describes the distribution of observations's differences in ranking of prediction results across classifiers
     based on the distribution of ranking's difference, choose the sample interval for blending 
     th reason of partial blending but by choosing only an interval but not do blending on all samples 
     is because of the fact that the xgb is
     good at classifying the easy cases, which have socres either close to 1 (highly suspected to default) or close to 0 (will very likely not default)
     by contrast, xgb is not quite capable of correctly classifying the cases with score close to 0.5,
     or in other words, hard to differentiate the cases close to decision boundary
     
     Taking this fact into consideration, it is not difficult to figure out a bettersolution to subsitute the blindly overall blending strategy
     the overall blending without assigning appropriate samples to appropriate classifiers,
     for example,easy ones give to xgb while complex ones should be dealt by lr
     in this sense, the new blending is partial stacking: to improve the overall prediction performance,
     select the tough nuts that having scores close
     to 0.5 to crack by simple algorithms like lr while let xgb handle the easy ones ranked at both ends, the overall performance after
     partial blending is supposed to grow
	"""
	def __init__(self,config,level,clf_name):
		self.config=config
		self.level=level
		self.__clf_name=clf_name

	def load_clf_file(self,level,name):
		reader=pd.read_csv(self.config.path_predict+level+'/'+name+'.csv',iterator=False,delimiter=',')#read prediction results
		d={}
		for i in range(len(reader['uid'])):
			d[reader['uid'][i]]=reader['score'][i]
		return d#return dict contains score indexed by uid

	def level_data(self):
		level=self.level
		clf_name=self.__clf_name
		
         #load data and split into training and validation
		config_instance=Config('')#choose log_move transfered data
		load_data_instance=load_origin_data.Load_origin_data(config_instance)
		X_0,test_X_0,X_1,test_X_1,uid_0,test_uid_0,uid_1,test_uid_1=load_data_instance.local_verify()

         #convert format to int and list 
		test_uid_0=test_uid_0.astype('int').tolist()
		test_uid_1=test_uid_1.astype('int').tolist()

         #loop through the classifiers
		for name in clf_name:
			prob=[]
			real=[]
			prob_1=[]
			prob_0=[]

			column_dict=self.load_clf_file(level,name)#obtain score of this classifier
			
			column_dict2=sorted(column_dict.items(),key=lambda d:d[1])#sort the score from small to large

			clf=[
				'_lr_sag',
				#'_lr_newton',
				#'_lr_lbfgs',
				#'_lr_liblinear',
				#'log_move_rf100',
				# 'log_move_rf200',
				# 'log_move_rf500',
				# 'log_move_rf1000',
				# 'log_move_gbdt20',
				# 'log_move_gbdt50',
				#'log_move_gbdt100',
				# 'log_move_ada20',
				# 'log_move_ada50',
				#'log_move_ada100',
				#'_xgb2000',
				#'_xgb2500',
				#'_xgb2000_2',
				#'_xgb2500_2'

			]
            
             #call level_ranks to return the ranking of samples, the smaller num in score, 
             #the smaller num in ranking, since it sorts from small to large
             
			ranks=[]#ranking in level two of another classifier???
			for f_name in clf:
				rank=self.level_ranks('level_one',f_name)#level_two
				ranks.append(rank)

			column_ranks=self.level_ranks(level,name)#ranking in level one


			i=0
			aa=0
			correct_count=0
			strategy_2_region_1_correct_count=0
			strategy_2_region_1_wrong_count=0

			strategy_2_region_2_correct_count=0
			strategy_2_region_3_correct_count=0
			strategy_2_region_2_wrong_count=0
			strategy_2_region_3_wrong_count=0

			AUC_BBM_Level_One=0.743722307172#0.744142151232
			AUC_BBM_Level_two=0.745082297258#_ada100
			wrong_count=0
			r_lr=0
			one_diff=[]
			zero_diff=[]
			one_index=[]
			zero_index=[]
            
			#choose interval of samples to blend

			# xgb_ranks_true=[]
			# xgb_ranks_false=[]
			# lr_ranks_true=[]
			# lr_ranks_false=[]
			# for k in range(21):
			# 	xgb_ranks_true.append(0)
			# 	xgb_ranks_false.append(0)
			# 	lr_ranks_true.append(0)
			# 	lr_ranks_false.append(0)
			# print(xgb_ranks_true)

			for uid, score in column_dict2:
				# if i<2000:
				# 	i+=1
				# 	continue
				diff=0#diff is the diff of observations' ranking in level one than that in level two
				for rank in ranks:
					diff+=column_ranks[uid][0]-rank[uid][0]#column_ranks contains classifier's score ranking of level one, rank contains that in level two


##########################
#  strategy 2: interval  #
########################## auc: 0.754049839922 > auc: 0.753618206126 the benchmark auc, correct 25 good users
				#the first interval
				if i>=9000/4 and i <=14000/4:
					if diff>9000/4:
						column_dict[uid]=0				    
						r_lr+=1

						if uid in test_uid_0:					
						    strategy_2_region_1_correct_count+=1
#
						if uid in test_uid_1:					
						    strategy_2_region_1_wrong_count+=1
		
				#the second interval
				if i>=14000/4 and i <=16000/4:#25000/4 (more radical) will triger
					if diff>12000/4:
						column_dict[uid]=0
						r_lr+=1

						if uid in test_uid_0:					
						    strategy_2_region_2_correct_count+=1
#
						if uid in test_uid_1:					
						    strategy_2_region_2_wrong_count+=1

				#the third interval
				if i>=20000/4 and i <=23000/4:#25000/4 (more radical) will triger
					if diff>15000/4:
						column_dict[uid]=0
						r_lr+=1

						if uid in test_uid_0:					
						    strategy_2_region_3_correct_count+=1

						if uid in test_uid_1:					
						    strategy_2_region_3_wrong_count+=1

#################################
#  strategy 3: subselect train  #
#################################
				#the first interval

                  #choose 
				if diff>2000/4+i*0.4:#or 
					if rank[uid][0]<32:#35 0.000462353399671 ,32 0.00176202757021
						#column_dict[uid]=0

						r_lr+=1
						if uid in test_uid_0:					
						    correct_count+=1
						if uid in test_uid_1:					
						    wrong_count+=1





				

				if uid in test_uid_0:
					zero_diff.append(diff)
					zero_index.append(i)
					aa+=1
					pass

				if uid in test_uid_1:
					one_diff.append(diff)
					one_index.append(i)

					pass
					
				i+=1

			print('hold-out',33)
			print(name)

			print("test uid size: ",(len(test_uid_0)+len(test_uid_1)))
			print(aa)
			print("numbers of estimation fixed: ",r_lr)
			print("correct fixing: ",correct_count)
			print("wrong fixing: ",wrong_count)		
			print("strategy_2_region_1_correct_count: ",strategy_2_region_1_correct_count)
			print("strategy_2_region_1_wrong_count: ",strategy_2_region_1_wrong_count)			
			print("strategy_2_region_2_correct_count: ",strategy_2_region_2_correct_count)
			print("strategy_2_region_2_wrong_count: ",strategy_2_region_2_wrong_count)				
			print("strategy_2_region_3_correct_count: ",strategy_2_region_3_correct_count)
			print("strategy_2_region_3_wrong_count: ",strategy_2_region_3_wrong_count)			

			#calculate AUC after blending
			for uid,score in column_dict.items():
				prob.append(score)
				if uid in test_uid_0:
					real.append(0)
					prob_0.append(score)
				elif uid in test_uid_1:
					real.append(1)
					prob_1.append(score)
				else:
					print("error")

			print( "classifiers :",name)#drop to auc: 0.753348228282 when one bad is estimated as good 

			auc_score=metrics.roc_auc_score(real,prob)#benchmark auc: 0.753618206126,auc: 0.760720180713
			print( "auc :",(auc_score))#drop to auc: 0.753348228282 when one bad is estimated as good 

			print( "auc increase:",(auc_score-AUC_BBM_Level_One))#drop to auc: 0.753348228282 when one bad is estimated as good 
			print( '0:',max(prob_0),min(prob_0))
			print( "1:",max(prob_1),min(prob_1))
 

			#plot the ranking difference among classifiers
			idex=0
			#self.print_diff(zero_diff[idex:],zero_index[idex:],one_diff[idex:],one_index[idex:])
			return


	def print_diff(self,zero_diff,zero_index,one_diff,one_index):
		"""
		:type zero_diff: List[int] difference of ranking for non-default observations
		:type zero_index: List[int] index of observations with label of 0
		:type one_diff: List[int] 1difference of ranking for default observations
		:type one_index: List[int] index of observations with label of 1
         plot the differences
		"""
		x=[]
		y=[]


		for i in range(2000,8000):
			x.append(i)
			y.append(3000/4+i*0.3)

		#plt.plot(x,y,color='yellow',linewidth=4) #comment this line when compare xgb vs. xgb 
		#plt.title('Figure 3.3: The Ranking Differences between BBM(XGB) and DBM(LR)',fontsize=24,fontname='Times New Roman',fontweight="bold")
		plt.xlabel('BBM(XGB) Ranking',fontsize=20,fontweight="bold",fontname='Times New Roman')
		plt.ylabel('Rank Difference',fontsize=20,fontweight="bold",fontname='Times New Roman')
		fig_size = plt.rcParams["figure.figsize"]
		fig_size[0] = 13
		fig_size[1] = 10
		plt.rcParams["figure.figsize"]= fig_size

		plt.scatter(zero_index,zero_diff,c='#00fa9a',label='Regular',s=3)
		plt.scatter(one_index,one_diff,c='#ff0000',label='Default',s=3)
		plt.legend(loc='upper left', prop={'size': 14})
		plt.show()
	def level_ranks(self,level,name):
		"""
         return the ranking of different samples in this classifier
		"""
		config_instance=Config('')#choose log_move transfered data
		load_data_instance=load_origin_data.Load_origin_data(config_instance)
		X_0,test_X_0,X_1,test_X_1,uid_0,test_uid_0,uid_1,test_uid_1=load_data_instance.local_verify()

		test_uid_0=test_uid_0.astype('int').tolist()
		test_uid_1=test_uid_1.astype('int').tolist()

		ranks={}
		column_dict=self.load_clf_file(level,name)
		column_dict2=sorted(column_dict.items(),key=lambda d:d[1])#sort from small num to large num
		i=0

		for uid, score in column_dict2:
			rank=ranks.get(uid,[])
			rank.append(i)
			ranks[uid]=rank
			i+=1

		return ranks

def main():
	config_instance=Config('')
	config_instance.path_predict=config_instance.path+'predict_local/' #file of prediction output
	level='level_one'# level_one
	clf_name=[
		#'_lr_sag',
		#'_lr_sag_1000',
		#'_lr_sag_1500',
		#'_lr_newton',
		# '_lr_lbfgs',
		# '_lr_liblinear',
		# '_rf100',
		 #'_rf200',
		 #'_rf500',
		#'_rf1000',
		 #'_gbdt20',
		 #'_gbdt50',
		 '_gbdt100',
		 #'_ada20',
		 #'_ada50',
		#'_ada100',
		#'_xgb2000',
		#'_xgb2500',
		# '_xgb2000_2',
		 #'_xgb2500_2',

	]
	predict_data_instance=Local_predict_verify(config_instance,level,clf_name)
	predict_data_instance.level_data()
	pass

if __name__ == '__main__':
	main()