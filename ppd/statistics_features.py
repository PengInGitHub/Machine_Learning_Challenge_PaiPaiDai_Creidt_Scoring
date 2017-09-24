#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import Config
from load_origin_data import Load_origin_data


class StatisticsFeatures():
	"""
	class: StatisticsFeatures   
     construct features based on the statistics of raw dataset, sort the features by value for numerical variables and by category rate for categorical variables 
     more specifically,

    for each column, calculate the missing value per row (not per column),
    plot the distribution and do binning based on the distribution, in this dataset missing value per row are partioned into 4 levels   
 
    for each numerical feature, calculate the ranking of each instance in this column, binning into 10 levels, aggregate frequency of 1,2...10
    in case there are N numercial variables in the raw data, N + N + 10 features will be generated
  
    for each categorical feature, calculate the ranking of each instance in this column based on the rate of this category, binning into 5 levels, aggregate frequency of 1,2...5
    in case there are N categorical variables in the raw data, M + M + 5 features will be generated  
    
    output: reader_statistics_features_output, a csv contains generated features

	"""
	def __init__(self,config):
		self.config=config
		self.origin_instance=Load_origin_data(config)


	def load_data(self):
		
		categorical_feature,numerical_feature,reader_category,reader_numeric =self.origin_instance.load_data_for_statistics_features()#load feature type, train and predict

		return categorical_feature,numerical_feature,reader_category,reader_numeric

###############################
#    missing statistics       #
###############################

	def deal_missing(self):
		features_category,features_numeric,reader_category,reader_numeric = self.load_data()
		reader=pd.merge(reader_category,reader_numeric,on='uid')
		reader.fillna(-1,inplace=True)
		reader['n_null'] = (reader<0).sum(axis=1)
		self.deal_missing_plot(reader)

		reader['discret_null']= reader['n_null']
		reader.discret_null[reader.discret_null<=80] = 1
		reader.discret_null[(reader.discret_null>80)&(reader.discret_null<=130)] = 2
		reader.discret_null[(reader.discret_null>130)&(reader.discret_null<=180)] = 3
		reader.discret_null[(reader.discret_null>180)] = 4
		reader_null=reader[['uid','n_null','discret_null']]

		return reader_null
                 
	def deal_missing_plot(self,file):
		"""
		process feature 
		"""
		missing_per_row = np.sum(file<0,axis=1)
		plt.xlabel('Observations')
		plt.ylabel('Missing Vale') 
		plt.plot(missing_per_row.values)
		plt.subplot(211).plot(np.sort(missing_per_row))
		plt.title('sort of missing values per row')
		plt.show()

	def output_deal_missing(self):
		reader_null=self.deal_missing()
		pd.DataFrame(reader_null).to_csv("reader_null.csv",index=None)

##########################################
#       numeric variables ranking        #
##########################################

#calculate ranking for numeric features
	def numeric_variable_ranking(self):
		features_category,features_numeric,reader_category,reader_numeric = self.load_data()

		reader_numeric[reader_numeric.y==0].fillna(reader_numeric[reader_numeric.y==0].median(),inplace=True)
		reader_numeric[reader_numeric.y==1].fillna(reader_numeric[reader_numeric.y==1].median(),inplace=True)
		reader_numeric=reader_numeric.fillna(reader_numeric.median(),inplace=True)

		print('Count of Missing Values: ',reader_numeric.isnull().sum())

		reader_numeric_rank = reader_numeric[['uid']]
		for feature in features_numeric:
			reader_numeric_rank['r_'+feature] = reader_numeric[feature].rank(method='max')/float(len(reader_numeric))

		return reader_numeric_rank

#discretization of num rankings
	def numeric_variable_ranking_dis(self):
		reader_numeric_rank = self.numeric_variable_ranking()

		reader_numeric_rank_x= reader_numeric_rank.drop(['uid'],axis=1)
		#discretization of ranking features
         #each 10% belongs to 1 level
		reader_numeric_rank_x[reader_numeric_rank_x<0.1] = 1
		reader_numeric_rank_x[(reader_numeric_rank_x>=0.1)&(reader_numeric_rank_x<0.2)] = 2
		reader_numeric_rank_x[(reader_numeric_rank_x>=0.2)&(reader_numeric_rank_x<0.3)] = 3
		reader_numeric_rank_x[(reader_numeric_rank_x>=0.3)&(reader_numeric_rank_x<0.4)] = 4
		reader_numeric_rank_x[(reader_numeric_rank_x>=0.4)&(reader_numeric_rank_x<0.5)] = 5
		reader_numeric_rank_x[(reader_numeric_rank_x>=0.5)&(reader_numeric_rank_x<0.6)] = 6
		reader_numeric_rank_x[(reader_numeric_rank_x>=0.6)&(reader_numeric_rank_x<0.7)] = 7
		reader_numeric_rank_x[(reader_numeric_rank_x>=0.7)&(reader_numeric_rank_x<0.8)] = 8
		reader_numeric_rank_x[(reader_numeric_rank_x>=0.8)&(reader_numeric_rank_x<1)] = 9
		reader_numeric_rank_x[reader_numeric_rank_x==1] = 10

         #rename      
         #nameing rules for ranking discretization features, add "d" in front of orginal features
         #for instance "x1" would have discretization feature of "dx1"
		rename_dict = {s:'d_'+s[0:] for s in reader_numeric_rank_x.columns.tolist()}
		reader_numeric_rank_x = reader_numeric_rank_x.rename(columns=rename_dict)
		reader_numeric_rank_x['uid'] = reader_numeric_rank.uid
		return reader_numeric_rank_x

    #frequency of ranking discretization 
	def numeric_variable_ranking_dis_freq(self):
		reader_numeric_rank_x = self.numeric_variable_ranking_dis()

		reader_numeric_rank_x['nn1'] = (reader_numeric_rank_x==1).sum(axis=1)
		reader_numeric_rank_x['nn2'] = (reader_numeric_rank_x==2).sum(axis=1)
		reader_numeric_rank_x['nn3'] = (reader_numeric_rank_x==3).sum(axis=1)
		reader_numeric_rank_x['nn4'] = (reader_numeric_rank_x==4).sum(axis=1)
		reader_numeric_rank_x['nn5'] = (reader_numeric_rank_x==5).sum(axis=1)
		reader_numeric_rank_x['nn6'] = (reader_numeric_rank_x==6).sum(axis=1)
		reader_numeric_rank_x['nn7'] = (reader_numeric_rank_x==7).sum(axis=1)
		reader_numeric_rank_x['nn8'] = (reader_numeric_rank_x==8).sum(axis=1)
		reader_numeric_rank_x['nn9'] = (reader_numeric_rank_x==9).sum(axis=1)
		reader_numeric_rank_x['nn10'] = (reader_numeric_rank_x==10).sum(axis=1)
		train_num_rank_dis_count = reader_numeric_rank_x[['uid','nn1','nn2','nn3','nn4','nn5','nn6','nn7','nn8','nn9','nn10']]

		return train_num_rank_dis_count

	def output_numeric_variable_ranking(self):
        
		reader_numeric_rank=self.numeric_variable_ranking()
		reader_numeric_rank.to_csv('reader_numeric_rank.csv',index=None)

		reader_numeric_rank_dis=self.numeric_variable_ranking_dis()
		reader_numeric_rank_dis.to_csv('reader_numeric_rank_dis.csv',index=None)

		reader_numeric_rank_dis_count=self.numeric_variable_ranking_dis_freq()	
		reader_numeric_rank_dis_count.to_csv('reader_numeric_rank_dis_count.csv',index=None)

		reader_numeric=pd.merge(pd.merge(reader_numeric_rank,reader_numeric_rank_dis,on='uid'),reader_numeric_rank_dis_count,on='uid')

		return reader_numeric
##########################################
#       category variables ranking        #
##########################################

#calculate ranking for category features
	def category_variable_ranking(self):
		features_category,features_numeric,reader_category,reader_numeric = self.load_data()

		reader_category.fillna(-1,inplace=True)

		reader_category_rank = reader_category[['uid']]
		for feature in features_category:
			reader_category_rank['r'+feature] = reader_category.groupby(feature)[feature].transform('count')/len(reader_category_rank)

		reader_category_rank.describe()
		return reader_category_rank


#discretization of category rankings
	def category_variable_ranking_dis(self):
		reader_category_rank = self.category_variable_ranking()

		reader_category_rank_x= reader_category_rank.drop(['uid'],axis=1)
		#discretization of ranking features
         #each 10% belongs to 1 level
		reader_category_rank_x[reader_category_rank_x<0.05] = 1
		reader_category_rank_x[(reader_category_rank_x>=0.05)&(reader_category_rank_x<0.1)] = 2
		reader_category_rank_x[(reader_category_rank_x>=0.1)&(reader_category_rank_x<0.25)] = 3
		reader_category_rank_x[(reader_category_rank_x>=0.25)&(reader_category_rank_x<0.7)] = 4
		reader_category_rank_x[(reader_category_rank_x>=0.7)&(reader_category_rank_x<1)] = 5
		
         #rename      
         #nameing rules for ranking discretization features, add "d" in front of orginal features
         #for instance "x1" would have discretization feature of "dx1"
		rename_dict = {s:'d_'+s[0:] for s in reader_category_rank_x.columns.tolist()}
		reader_category_rank_x = reader_category_rank_x.rename(columns=rename_dict)
		reader_category_rank_x['uid'] = reader_category_rank.uid
		return reader_category_rank_x

#frequency of ranking discretization 
	def category_variable_ranking_dis_freq(self):
		reader_category_rank_x = self.category_variable_ranking_dis()

		reader_category_rank_x['cn1'] = (reader_category_rank_x==1).sum(axis=1)
		reader_category_rank_x['cn2'] = (reader_category_rank_x==2).sum(axis=1)
		reader_category_rank_x['cn3'] = (reader_category_rank_x==3).sum(axis=1)
		reader_category_rank_x['cn4'] = (reader_category_rank_x==4).sum(axis=1)
		reader_category_rank_x['cn5'] = (reader_category_rank_x==5).sum(axis=1)
	
		reader_category_rank_dis_count = reader_category_rank_x[['uid','cn1','cn2','cn3','cn4','cn5']]

		return reader_category_rank_dis_count

	def output_category_variable_ranking(self):
        
		reader_category_rank=self.category_variable_ranking()
		reader_category_rank.to_csv('reader_category_rank.csv',index=None)

		reader_category_rank_dis=self.category_variable_ranking_dis()
		reader_category_rank_dis.to_csv('reader_category_rank_dis.csv',index=None)

		reader_category_rank_dis_count=self.category_variable_ranking_dis_freq()	
		reader_category_rank_dis_count.to_csv('reader_category_rank_dis_count.csv',index=None)


		reader_category=pd.merge(pd.merge(reader_category_rank,reader_category_rank_dis,on='uid'),reader_category_rank_dis_count,on='uid')

		return reader_category

	def output_all(self):
        
		reader_category=self.output_category_variable_ranking()

		reader_numeric=self.output_numeric_variable_ranking()

		reader_null=self.deal_missing()	

		reader_statistics_features_output=pd.merge(pd.merge(reader_category,reader_numeric,on='uid'),reader_null,on='uid')

		reader_statistics_features_output.to_csv('reader_statistics_features_output.csv',index=None)

def main():
    instance=StatisticsFeatures(Config(''))
    #instance.load_data()
    #instance.output_deal_missing()
    #instance.output_numeric_variable_ranking()
    #instance.output_category_variable_ranking()
    instance.output_all()

pass

if __name__ == '__main__':
	main()
        