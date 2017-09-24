#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from datetime import datetime
from config import Config
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import load_origin_data

class Load_train_data():
	"""
    class Load_train_data --- normal or partial stacking ensemble
    the main idea in this level is to use the classifiers' prediction result in previous level as features for next level
    to train model with hiagher generalization ability
    
    level_data returns prediction results of level_one model fitting (normal stacking)
    level_data_part returns subsamples (partial stacking)
    
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
		"""
		:type level: str, which leval of data to read
		:type name: str, name of classifier
         return log-transformed predictions(probability to default) of this classifier on the previous level
         
		more specifically, read the output(probabilities) of classifers on the last level 
         and do log transfermation to the last-level result so as to have more stable data to use in this level
         return a dict that contains info above
		"""
		reader=pd.read_csv(self.config.path_train+level+'/'+name+'.csv',iterator=False,delimiter=',',header=None)#read uid-score pair csv for the classifier, don't forget header=None
        
		d={} #loop through all the rows in the second column of reader(score from classifiers on the last level) and do log transfer to it, save in dict d

		for i in list(range(len(reader[0]))):
			d[reader[0][i]]=np.log10(reader[1][i])#remove np.log10??
		return d

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
         prepare input for normal stacking, which is the output of classifiers in level_one
         return the  ranking of all observations across predictions of all classifiers on the last level
         for instance: uid  ranking_lr ranking_xgb1 ... ranking_rf2
                       5662       6372       7352         72
                       5663       782         672         673
                       ...         ...

		"""
		level=self.level
		clf_name=self.__clf_name

		config_instance=Config('')#choose log_move transfered data
		load_data_instance=load_origin_data.Load_origin_data(config_instance)

		X,y,uids,X_0,X_1,uid_00,uid_1q=load_data_instance.load_final()

		#uncomment the three lines of code when locan verification is in need
		#comment them when you do final prediction and testing
		#scripts below are loading data which is splited into train and validation locally)
		#take 20% of training data as validation set to do local training
		#X_00,test_X_00,X_11,test_X_11,uid_00,test_uid_00,uid_11,test_uid_11=load_data_instance.train_test_xy()
		#y=np.hstack((np.ones(len(X_00)),np.zeros(len(X_11))))
		#uids=np.hstack((uid_00,uid_11))

		column_important=[]
		d={}
		for name in clf_name:

			column_dict=self.load_clf_file(level,name)#dict contains uid and score of each observation in this specific classifier
			column_score=self.load_clf_score(level,name)#averge auc score this this classifier on the last_level
			column_important.append(column_score)# column_important list contains the average socre of each classifier in clf_name


			for uid in uids:
				temp=d.get(uid,[])
				temp.append(column_dict[uid])#append the uid in column_dict into dict d
				d[uid]=temp#now d contains uid, end of loop
				
                
		X_0=[]
		X_1=[]
		uid_0=[]
		uid_1=[]

		#reverse neg and pos in dict again, return reveresed X_0, X_1, uid_0,uid_1
		for i in range(len(y)):
			if y[i]==1:
				X_1.append(d[uids[i]])
				uid_1.append(uids[i])
			else:
				X_0.append(d[uids[i]])
				uid_0.append(uids[i])

		print( "shape of X_0 is ",(np.array(X_0).shape),"shape of X_1 is ",(np.array(X_1).shape),"shape of uid_0 is ",(np.array(uid_0)).shape,"shape of uid_1 is ",(np.array(uid_1)).shape)
		return np.array(X_0),np.array(X_1),np.array(uid_0),np.array(uid_1)
         
    

    		
	def level_ranks(self,column_dict):
		'''
         sort the observations by predicted value
         return the ranking of observations in terms of predicted prob, asendingly
		'''
		column_dict2=sorted(column_dict.items(),key=lambda d:d[1])#sort from small num to large num
		i=0
		ranks={}
		for uid, score in column_dict2:
			ranks[uid]=i
			i+=1

		return ranks
    #lelve 1
    #level_data_part is the major function for partial stacking ensemble
    #it calculates the change of each observation's ranking in terms of predicited default probability by each classifier from level one
    #choose samples with high difference btw xgb, BBM, which is the best single classifier and lr, DBM, the most different single classifier from BBM,
    #and select samples that have rank are close to default end, which are labeled by 1
    #BBM are over-sensitive to these samples, many of them are actually normal cases but XGB thought they were dangeous
    #select these samples and re-train them by DBM, a much simple model will give them better evaluation
    
    #level 2
    #so these samples are selected and trained individually by lr, in level two, which is the core part of stacking ensemble
    #this is b/c xgb could not predict them well while lr can
   
    #according to the second level training result, top K of them are least probablly going to default will be set directly as 0
    #or draw a line to choose samples above this line, the line y = ax + b has slope a and interval b, these two values are obtained by CV
    #this is so-called partial stacking ensemble, since it chooses only part of samples to do ensemble
                                        
	#function level_data_part instantiates training models with selected samples
    
	def level_data_part(self):
		"""
         select samples that are close to default points and have relative large ranking differences 
         between the predicted scores on BBM(Usually XGB) and DBM(linear model usually)
         use these samples to do next level LR training
		"""
		level=self.level
		clf_name=self.__clf_name
        
        ############################### 
        #      data prepariation      #
        ###############################
        

		config_instance=Config('')
		load_data_instance=load_origin_data.Load_origin_data(config_instance)

		X,y,uids,X_0,X_1,uid_00,uid_11=load_data_instance.load_final()


		#uncomment the three lines of code when locan verification is in need
		#comment them when you do final prediction and testing
		#scripts below are loading data which is splited into train and validation locally)
		#take 20% of training data as validation set to do local training
		#X_00,test_X_00,X_11,test_X_11,uid_00,test_uid_00,uid_11,test_uid_11=load_data_instance.train_test_xy(1)
		#y=np.hstack((np.zeros(len(X_00)),np.ones(len(X_11))))
		#uids=np.hstack((uid_00,uid_11))
	               
        ############################### 
        #       stacking ensemble     #
        ###############################        
        #pick up samples with high volatility on change of predicted auc score ranking among all samples
        
		column_important=[]
		d={}
		diff_uid=set([])#store samples locate above the line

         #begin of looping through the classifiers list
		for name in clf_name:

			column_dict=self.load_clf_file(level,name)#call load_clf_file to obtain log-transformed prediction result on each observation by this classifier
            
			column_score=self.load_clf_score(level,name)#call load_clf_score to obtain the average AUC socre of this classifier 
            
			column_important.append(column_score)#add the average performance of this classifier

			column_rank=self.level_ranks(column_dict)#call level_ranks to obtain the ranking of observations, the smaller score, the higher rank
                              

			lr_dict2=self.load_clf_file('level_two','_lr_sag_1000')#DBM_Sub,#obtain the prediction results of selected samples are re-fitted by lr in Level_Two
			lr_rank2=self.level_ranks(lr_dict2)

			_lr_liblinear_dict=self.load_clf_file(level,'_lr_sag_1000')#_xgb1000, lr_sag, _lr_sag_1500
			_lr_liblinear_rank=self.level_ranks(_lr_liblinear_dict)

			#print('lr_rank2',len(lr_rank2))

			print("classifier ",name, " in level_one model fitting achieved average AUC: ",column_score)
            
			column_dict2=sorted(column_dict.items(),key=lambda d:d[1])#observation ranking again
            
			max_column=max([v for k,v in column_dict.items()])#highest score in this classifier's prediction
			min_column=min([v for k,v in column_dict.items()])#smallest score in this classifier's prediction

			#max_lr=max([v for k,v in lr_dict2.items()])#highest score in DBM's prediction
			#min_lr=min([v for k,v in lr_dict2.items()])#smallest score in  DBM's prediction
            
			print( 'highest score in BBM is: '+str(max_column),' ','lowest score in BBM: '+str(min_column))
			#print( 'highest score in DBM_Sub : '+str(max_lr),' ','lowest score in DBM_Sub: '+str(min_lr))

			i=0
			r_lr=0
			correct_count=0
			wrong_count=0
			prob=[]
			real=[]
			prob_1=[]
			prob_0=[]
			one_diff=[]
			zero_diff=[]
			one_index=[]
			zero_index=[]
			yy=[]
			scores=[]
            
             #start of loops to generate diff_uid
			for uid,score in column_dict2:
				#score=(score-min_column)/(max_column-min_column)#standardization
                 #score is now the probability to default
				temp=d.get(uid,[])
				temp.append(column_dict[uid])
				d[uid]=temp

############################
#  choose benchmarks here  #
############################

                  #calculate the difference of the sample's ranking in by xgb and by benchmark model
				diff=column_rank[uid]-_lr_liblinear_rank[uid]

                  #append the difference value into uid_00, yy appends 0
				if uid in uid_00:
					zero_diff.append(diff)
					zero_index.append(i)
					yy.append(0)
                  #append the difference value into uid_1, yy appends 1 
				else:
					one_diff.append(diff)
					one_index.append(i)
					yy.append(1)
################ 
#  strategy 2  #
################
                  #choose samples above the line with a = 0.4 and b = 2000
				#if diff>3000+i*0.42:#or diff>2500+i*0.2, choose a and b here
				#	diff_uid.add(uid)
				#	if __lr_liblinear_dict[uid]<500:#>200
				#		#score=-100#let the score be very small, 
				#		score=0.7+0.3*((score-min_lr)/(max_lr-min_lr))# 
						#score=-100

################ 
#  strategy 3  #
################
                  #choose samples above the line with a = 0.4 and b = 2000
				if diff>2000+i*0.4:
					diff_uid.add(uid)
					if _lr_liblinear_dict[uid]<50:
						column_dict[uid]=0
						score=10000

						r_lr+=1
						if uid in uid_00:					
						    correct_count+=1
						if uid in uid_11:					
						    wrong_count+=1
				scores.append(score)
				i+=1
             #end of loops

			idex=0

			#calculate AUC after blending
			for uid,score in column_dict.items():
				prob.append(score)
				if uid in uid_00:
					real.append(0)
					prob_0.append(score)
				elif uid in uid_11:
					real.append(1)
					prob_1.append(score)
				else:
					print("error")

			auc_score=metrics.roc_auc_score(real,prob)#benchmark BBM in level_one 0.7505509665787999

			print("num of samples are above the line: ",len(diff_uid))
			print("numbers of estimation fixed: ",r_lr)
			print("correct fixing: ",correct_count)
			print("wrong fixing: ",wrong_count)
			print ('auc increase:',auc_score-column_score)#a pos value will justify this para value
			self.print_diff(zero_diff[idex:],zero_index[idex:],one_diff[idex:],one_index[idex:])
			break
        
         #end of looping through the classifiers list		
		
		X_0=[]
		X_1=[]
		uid_0=[]
		uid_1=[]

		for i in list(range(len(y))):
			if uids[i] in diff_uid:
				if y[i]==0:
					#print i
					X_0.append(X[i])#select samples above the line
					uid_0.append(uids[i])
				else:
					X_1.append(X[i])#select samples above the line
					uid_1.append(uids[i])

		return np.array(X_0),np.array(X_1),np.array(uid_0),np.array(uid_1)
        

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
		#for i in range(9000,12500):
		#	x.append(i)
		#	y.append(8000)

		#plt.plot(x,y,color='yellow',linewidth=4) #comment this line when compare xgb vs. xgb 

		#for i in range(14000,16000):
		#	x.append(i)
		#	y.append(12000)

		#plt.plot(x,y,color='yellow',linewidth=4) #comment this line when compare xgb vs. xgb 

		for i in range(20000,23000):
			x.append(i)
			y.append(15000)

		plt.plot(x,y,color='yellow',linewidth=4) #comment this line when compare xgb vs. xgb 


#		plt.title('Figure 3.3: The Ranking Differences between BBM(XGB) and DBM(LR)',fontsize=24,fontname='Times New Roman',fontweight="bold")
#		plt.title('Figure 5.1: Strategy 1 - Select Top L ',fontsize=24,fontname='Times New Roman',fontweight="bold")
#		plt.title('Figure 5.2: Strategy 2 - Select Top L By Interval',fontsize=24,fontname='Times New Roman',fontweight="bold")
		#plt.title('Figure 5.3: Strategy 3 - Select Top L By Training',fontsize=24,fontname='Times New Roman',fontweight="bold")

		plt.xlabel('BBM Ranking',fontsize=20,fontweight="bold",fontname='Times New Roman')
		plt.ylabel('Rank Difference',fontsize=20,fontweight="bold",fontname='Times New Roman')
		fig_size = plt.rcParams["figure.figsize"]
		fig_size[0] = 13
		fig_size[1] = 10
		plt.rcParams["figure.figsize"]= fig_size

		plt.scatter(zero_index,zero_diff,c='#00fa9a',label='Regular',s=3)
		plt.scatter(one_index,one_diff,c='#ff0000',label='Default',s=3)
		plt.legend(loc='upper left', prop={'size': 14})
		plt.show()

def  main():

	ftype=''
	config_instance=Config(ftype)
	level='level_one'
    #choose BBM
	clf_name=[
		#ftype+'_lr_sag_1000',
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
		#ftype+'_xgb2000',
		 ftype+'_xgb2500',
		# ftype+'_xgb2000_2',
		# ftype+'_xgb2500_2'
	]
    ###############################################################################
    #  choose one  below to uncomment, run run.py level_two_wrapper to see results#
    ###############################################################################
     #normal stacking case, call level data
	#load_data_instance=Load_train_data(config_instance,level,clf_name_level_data)#instantiate class
	#X_0,X_1,uid_0,uid_1=load_data_instance.level_data()#call level_data_part, return data and plot

     #partial stacking case, call level_data_part
	load_data_instance=Load_train_data(config_instance,level,clf_name)#instantiate class
	X_0,X_1,uid_0,uid_1=load_data_instance.level_data_part()#call level_data_part, return data and plot
	print('num of good cases in subsamples: ',len(X_0),'num of bad cases in subsamples: ',len(X_1))

	pass

if __name__ == '__main__':
	main()
	

