
################################
#       1.prepare data         #
################################
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import random
import _pickle as cPickle
import os
from sklearn.model_selection import train_test_split

#craete a dicrectory if it doesn't exist
#https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
#if not makes this more robust
if not os.path.exists("featurescore"):
    os.makedirs("featurescore")
if not os.path.exists("model"):
    os.makedirs("model")    
if not os.path.exists("preds"):
    os.makedirs("preds")    

train =pd.read_csv("ppd_train_withid.csv")

train.rename(columns={'target':'y'}, inplace=True)
train.rename(columns={'Idx':'uid'}, inplace=True)

original = train

sum(train.target)/(len(train.target)-sum(train.target))

################################
#   2.num variable rankings    #
################################

feature_type = pd.read_csv("ppd_feature_type.csv")

numeric_feature = list(feature_type[feature_type.feature_type=='Numerical'].feature_name)


#clean up the list: remove y 
if 'y' in numeric_feature: numeric_feature.remove('y')
if 'WeblogInfo_1' in numeric_feature: numeric_feature.remove('WeblogInfo_1')
if 'WeblogInfo_3' in numeric_feature: numeric_feature.remove('WeblogInfo_3')

type(numeric_feature)

train_numeric = train[['uid']+numeric_feature]
train_rank = pd.DataFrame(train_numeric.uid,columns=['uid'])

for feature in numeric_feature:
    train_rank['r'+feature] = train_numeric[feature].rank(method='max')
train_rank.to_csv('train_x_rank.csv',index=None)
train_rank.shape



#####################################
#   3.discretization of rankings    #
#####################################

train_x = train_rank.drop(['uid'],axis=1)

#discretization of ranking features
#each 10% belongs to 1 level
train_x[train_x<7000] = 1
train_x[(train_x>=7000)&(train_x<14000)] = 2
train_x[(train_x>=14000)&(train_x<21000)] = 3
train_x[(train_x>=21000)&(train_x<28000)] = 4
train_x[(train_x>=28000)&(train_x<35000)] = 5
train_x[(train_x>=35000)&(train_x<42000)] = 6
train_x[(train_x>=42000)&(train_x<49000)] = 7
train_x[(train_x>=49000)&(train_x<56000)] = 8
train_x[(train_x>=56000)&(train_x<63000)] = 9
train_x[train_x>=63000] = 10

#rename      
#nameing rules for ranking discretization features, add "d" in front of orginal features
#for instance "x1" would have discretization feature of "dx1"
rename_dict = {s:'d'+s[1:] for s in train_x.columns.tolist()}
train_x = train_x.rename(columns=rename_dict)
train_x['uid'] = train_rank.uid
      
train_x.to_csv('train_x_discretization.csv',index=None)      

#############################################
#   4.frequency of ranking discretization   #
#############################################


train_x['n1'] = (train_x==1).sum(axis=1)
train_x['n2'] = (train_x==2).sum(axis=1)
train_x['n3'] = (train_x==3).sum(axis=1)
train_x['n4'] = (train_x==4).sum(axis=1)
train_x['n5'] = (train_x==5).sum(axis=1)
train_x['n6'] = (train_x==6).sum(axis=1)
train_x['n7'] = (train_x==7).sum(axis=1)
train_x['n8'] = (train_x==8).sum(axis=1)
train_x['n9'] = (train_x==9).sum(axis=1)
train_x['n10'] = (train_x==10).sum(axis=1)
train_x[['uid','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']].to_csv('train_x_nd.csv',index=None)


##############################################
#   5.feature importance of rank features    #
##############################################
#generate a variety of xgboost models to have rank feature importance


#craete a dicrectory if it doesn't exist
#https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
#if not makes this more robust
if not os.path.exists("featurescore"):
    os.makedirs("featurescore")
if not os.path.exists("model"):
    os.makedirs("model")    
if not os.path.exists("preds"):
    os.makedirs("preds")    

##################################################
#                 split dataset                  #
##################################################
#load data
train_x = pd.read_csv("train_x_rank.csv")
train_y = train[['uid']+['y']]
train_xy = pd.merge(train_x,train_y,on='uid')

train_xy, test_xy = train_test_split(train_xy, test_size = 0.2)
#train_xy 32,000*158
#test_xy 8,000*158
train_y = train_xy[['uid']+['y']]
test_y = test_xy[['uid']+['y']]
#label or xgb
y=train_xy.y


###########
#  train  #
###########
#leave features only
train_x= train_xy.drop(["uid",'y'],axis=1)
#convert to percentage 
X = train_x/68294
#to xgb.DMatrix format
dtrain = xgb.DMatrix(X, label=y)

###########
#   test  #
###########
#do the same to test table    
test = test_xy
test_uid = test.uid
test = test.drop(["uid",'y'],axis=1)
test_x = test/68294
dtest = xgb.DMatrix(test_x)

##########################
#   feature importance   #
##########################
#define an xgb model to do calculate feature importance 
def pipeline(iteration,random_seed,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight):
    params={
    	'booster':'gbtree',
    	'objective': 'binary:logistic',
    	'scale_pos_weight': float(len(y)-sum(y))/float(sum(y)),
        'eval_metric': 'auc',
    	'gamma':gamma,
    	'max_depth':max_depth,
    	'lambda':lambd,
        'subsample':subsample,
        'colsample_bytree':colsample_bytree,
        'min_child_weight':min_child_weight, 
        'eta': 0.04,
    	'seed':random_seed,
        }
    
    watchlist  = [(dtrain,'train')]
    model = xgb.train(params,dtrain,num_boost_round=700,evals=watchlist)
    model.save_model('./model/xgb{0}.model'.format(iteration))
    
    #predict test set
    test_y = model.predict(dtest)
    test_result = pd.DataFrame(columns=["uid","score"])
    test_result.uid = test_uid
    test_result.score = test_y
    test_result.to_csv("./preds/xgb{0}.csv".format(iteration),index=None,encoding='utf-8')
    
    #save feature score
    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
    fs = []
    for (key,value) in feature_score:
        fs.append("{0},{1}\n".format(key,value))
    
    with open('./featurescore/feature_score_{0}.csv'.format(iteration),'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)

if __name__ == "__main__":
    random_seed = list(range(1000,2000,10))
    gamma = [i/1000.0 for i in list(range(100,200,1))]
    max_depth = [6,7,8]
    lambd = list(range(100,200,1))
    subsample = [i/1000.0 for i in list(range(500,700,2))]
    colsample_bytree = [i/1000.0 for i in list(range(250,350,1))]
    min_child_weight = [i/1000.0 for i in list(range(200,300,1))]
    random.shuffle(random_seed)
    random.shuffle(gamma)
    random.shuffle(max_depth)
    random.shuffle(lambd)
    random.shuffle(subsample)
    random.shuffle(colsample_bytree)
    random.shuffle(min_child_weight)
    
#save params for reproducing
    with open('params.pkl','wb') as f:
        cPickle.dump((random_seed,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight),f)
    

#train 100 xgb
    for i in list(range(100)):
        pipeline(i,random_seed[i],gamma[i],max_depth[i%3],lambd[i],subsample[i],colsample_bytree[i],min_child_weight[i])

#run from here
##################################
#   average feature importance   #
##################################
#calculate average feature score for ranking features

#get rank feature importance info from the xgboost models
import pandas as pd 
import os

#featurescore folder contains csv files called feature_score_* that tells feature importance

files = os.listdir('featurescore')
#save into a dict
fs = {}
for f in files:
    t = pd.read_csv('featurescore/'+f)
    t.index = t.feature
    t = t.drop(['feature'],axis=1)
    d = t.to_dict()['score']
    for key in d:
        if key in fs:
            fs[key] += d[key]
        else:
            fs[key] = d[key] 
       
#sort and organize the dict            
fs = sorted(fs.items(), key=lambda x:x[1],reverse=True)

t = []
type(t)
for (key,value) in fs:
    t.append("{0},{1}\n".format(key,value))

#save the overall importance scores of ranking features into csv
with open('rank_feature_score.csv','w') as f:
    f.writelines("feature,score\n")
    f.writelines(t)
train.ListingInfo
##############################################
#     6.feature importance of raw features   #
##############################################


if not os.path.exists("featurescore"):
    os.makedirs("featurescore")
if not os.path.exists("model"):
    os.makedirs("model")    
if not os.path.exists("preds"):
    os.makedirs("preds") 
##load data

train_xy = original

train_xy, test_xy = train_test_split(train_xy, test_size = 0.2)

train_y = train_xy[['uid']+['y']]
test_y = test_xy[['uid']+['y']]
#flabel or xgb
y=train_xy.y


###########
#  train  #
###########
#leave features only
X= train_xy.drop(["uid",'y'],axis=1)
#to xgb.DMatrix format
dtrain = xgb.DMatrix(X, label=y)

###########
#   test  #
###########
#do the same to test table    
test = test_xy
test_uid = test.uid
test = test.drop(["uid",'y'],axis=1)
dtest = xgb.DMatrix(test_x)

##########################
#   feature importance   #
##########################
#define an xgb model to do calculate feature importance 
def pipeline(iteration,random_seed,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight):
    params={
    	'booster':'gbtree',
    	'objective': 'binary:logistic',
    	'scale_pos_weight': float(len(y)-sum(y))/float(sum(y)),
        'eval_metric': 'auc',
    	'gamma':gamma,
    	'max_depth':max_depth,
    	'lambda':lambd,
        'subsample':subsample,
        'colsample_bytree':colsample_bytree,
        'min_child_weight':min_child_weight, 
        'eta': 0.04,
    	'seed':random_seed,
        }
    
    watchlist  = [(dtrain,'train')]
    model = xgb.train(params,dtrain,num_boost_round=8,evals=watchlist)
    model.save_model('./model/xgb{0}.model'.format(iteration))
    
    #predict test set
    test_y = model.predict(dtest)
    test_result = pd.DataFrame(columns=["uid","score"])
    test_result.uid = test_uid
    test_result.score = test_y
    test_result.to_csv("./preds/xgb{0}.csv".format(iteration),index=None,encoding='utf-8')
    
    #save feature score
    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
    fs = []
    for (key,value) in feature_score:
        fs.append("{0},{1}\n".format(key,value))
    
    with open('./featurescore/feature_score_{0}.csv'.format(iteration),'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)
        

if __name__ == "__main__":
    random_seed = list(range(1000,2000,10))
    gamma = [i/1000.0 for i in list(range(100,200,1))]
    max_depth = [6,7,8]
    lambd = list(range(100,200,1))
    subsample = [i/1000.0 for i in list(range(500,700,2))]
    colsample_bytree = [i/1000.0 for i in list(range(250,350,1))]
    min_child_weight = [i/1000.0 for i in list(range(200,300,1))]
    random.shuffle(random_seed)
    random.shuffle(gamma)
    random.shuffle(max_depth)
    random.shuffle(lambd)
    random.shuffle(subsample)
    random.shuffle(colsample_bytree)
    random.shuffle(min_child_weight)
    
#save params for reproducing
    with open('params.pkl','wb') as f:
        cPickle.dump((random_seed,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight),f)
    


#train 100 xgb
    for i in list(range(100)):
        pipeline(i,random_seed[i],gamma[i],max_depth[i%3],lambd[i],subsample[i],colsample_bytree[i],min_child_weight[i])

#run from here
##################################
#   average feature importance   #
##################################

#get rank feature importance info from the xgboost models
import pandas as pd 
import os

#featurescore folder contains csv files called feature_score_* that tells feature importance

files = os.listdir('featurescore')
#save into a dict
fs = {}
for f in files:
    t = pd.read_csv('featurescore/'+f)
    t.index = t.feature
    t = t.drop(['feature'],axis=1)
    d = t.to_dict()['score']
    for key in d:
        if key in fs:
            fs[key] += d[key]
        else:
            fs[key] = d[key] 
       
#sort and organize the dict            
fs = sorted(fs.items(), key=lambda x:x[1],reverse=True)

t = []
type(t)
for (key,value) in fs:
    t.append("{0},{1}\n".format(key,value))

#save the overall importance scores of ranking features into csv
with open('raw_feature_score.csv','w') as f:
    f.writelines("feature,score\n")
    f.writelines(t)


#start from here
##################################################
#     7.feature importance of discret features   #
##################################################

if not os.path.exists("featurescore"):
    os.makedirs("featurescore")
if not os.path.exists("model"):
    os.makedirs("model")    
if not os.path.exists("preds"):
    os.makedirs("preds")    

##################################################
#                 split dataset                  #
##################################################
#load data

train = original
train.rename(columns={'flag':'y'}, inplace=True)
train.rename(columns={'no':'uid'}, inplace=True)

train_x = pd.read_csv("train_x_discretization.csv")
train_y = train[['uid']+['y']]
train_xy = pd.merge(train_x,train_y,on='uid')

train_xy, test_xy = train_test_split(train_xy, test_size = 0.2)
#train_xy 32,000*158
#test_xy 8,000*158
train_y = train_xy[['uid']+['y']]
test_y = test_xy[['uid']+['y']]
#flabel or xgb
y=train_xy.y


###########
#  train  #
###########
#leave features only
train_x= train_xy.drop(["uid",'y'],axis=1)
#convert to percentage 
X = train_x/len(train_x)
#to xgb.DMatrix format
dtrain = xgb.DMatrix(X, label=y)

###########
#   test  #
###########
#do the same to test table    
test = test_xy
test_uid = test.uid
test = test.drop(["uid",'y'],axis=1)
test_x = test/len(test_x)
dtest = xgb.DMatrix(test_x)

##########################
#   feature importance   #
##########################
#define an xgb model to do calculate feature importance 
def pipeline(iteration,random_seed,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight):
    params={
    	'booster':'gbtree',
    	'objective': 'binary:logistic',
    	'scale_pos_weight': float(len(y)-sum(y))/float(sum(y)),
        'eval_metric': 'auc',
    	'gamma':gamma,
    	'max_depth':max_depth,
    	'lambda':lambd,
        'subsample':subsample,
        'colsample_bytree':colsample_bytree,
        'min_child_weight':min_child_weight, 
        'eta': 0.04,
    	'seed':random_seed,
        }
    
    watchlist  = [(dtrain,'train')]
    model = xgb.train(params,dtrain,num_boost_round=1350,evals=watchlist)
    model.save_model('./model/xgb{0}.model'.format(iteration))
    
    #predict test set
    test_y = model.predict(dtest)
    test_result = pd.DataFrame(columns=["uid","score"])
    test_result.uid = test_uid
    test_result.score = test_y
    test_result.to_csv("./preds/xgb{0}.csv".format(iteration),index=None,encoding='utf-8')
    
    #save feature score
    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
    fs = []
    for (key,value) in feature_score:
        fs.append("{0},{1}\n".format(key,value))
    
    with open('./featurescore/feature_score_{0}.csv'.format(iteration),'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)



if __name__ == "__main__":
    random_seed = list(range(1000,2000,10))
    gamma = [i/1000.0 for i in list(range(100,200,1))]
    max_depth = [6,7,8]
    lambd = list(range(100,200,1))
    subsample = [i/1000.0 for i in list(range(500,700,2))]
    colsample_bytree = [i/1000.0 for i in list(range(250,350,1))]
    min_child_weight = [i/1000.0 for i in list(range(200,300,1))]
    random.shuffle(random_seed)
    random.shuffle(gamma)
    random.shuffle(max_depth)
    random.shuffle(lambd)
    random.shuffle(subsample)
    random.shuffle(colsample_bytree)
    random.shuffle(min_child_weight)
    
#save params for reproducing
    with open('params.pkl','wb') as f:
        cPickle.dump((random_seed,gamma,max_depth,lambd,subsample,colsample_bytree,min_child_weight),f)
    

#train 100 xgb
    for i in list(range(100)):
        pipeline(i,random_seed[i],gamma[i],max_depth[i%3],lambd[i],subsample[i],colsample_bytree[i],min_child_weight[i])

#run from here
##################################
#   average feature importance   #
##################################
#calculate average feature score for ranking features

#get rank feature importance info from the xgboost models
import pandas as pd 
import os

#featurescore folder contains csv files called feature_score_* that tells feature importance

files = os.listdir('featurescore')
#save into a dict
fs = {}
for f in files:
    t = pd.read_csv('featurescore/'+f)
    t.index = t.feature
    t = t.drop(['feature'],axis=1)
    d = t.to_dict()['score']
    for key in d:
        if key in fs:
            fs[key] += d[key]
        else:
            fs[key] = d[key] 
       
#sort and organize the dict            
fs = sorted(fs.items(), key=lambda x:x[1],reverse=True)

t = []
type(t)
for (key,value) in fs:
    t.append("{0},{1}\n".format(key,value))

#save the overall importance scores of ranking features into csv
with open('discreet_feature_score.csv','w') as f:
    f.writelines("feature,score\n")
    f.writelines(t)




#####################################################
#      normalization of numeric variables           #
#####################################################

train=train.fillna(-1)


plt.figure(1)
train_numeric.UserInfo_213.apply(lambda x: x/10).hist()
plt.figure(2)
luk = train_numeric.UserInfo_213.apply(lambda x: np.log(x+1).round()).hist()

def numpy_minmax(X):
    xmin =  X.min(axis=0)
    return (X - xmin) / (X.max(axis=0) - xmin)

plt.figure(1)
train_numeric.UserInfo_213.apply(lambda x: x/10).hist()
plt.figure(2)
numpy_minmax(train_numeric.UserInfo_213).hist()
from sklearn import preprocessing

standardized_X = preprocessing.scale(train_numeric.UserInfo_213)
train_numeric.UserInfo_5.apply(lambda x: x/10).hist()

luk = train_numeric.UserInfo_213.apply(lambda x: np.log(x+1).round()).hist()



##################################################
#                 8.xgb bagging                  #
##################################################

###########################
#      prepare data       #
###########################

import pandas as pd
import xgboost as xgb
import random
import _pickle as cPickle
import os

os.mkdir('featurescore')
os.mkdir('model')
os.mkdir('preds')



#count features of ranking discretion
train_nd = pd.read_csv('train_x_nd.csv')[['uid','n1','n2','n3','n4','n5','n6','n7','n8','n9','n10']]

#discret features of count of missing values
train_dnull = pd.read_csv('train_x_null.csv')[['uid','discret_null']]

#considering the size of the features above (only 11) it is not necessary to do feature selection on them
#so they are merged and left alone in the feature selection process
eleven_feature = ['n1','n2','n3','n4','n5','n6','n7','n8','n9','n10','discret_null']
train_eleven = pd.merge(train_nd,train_dnull,on='uid')



#discreet features
discreet_feature_score = pd.read_csv('discreet_feature_score.csv')
#80 of 122
fs = list(discreet_feature_score.feature[0:80])
discreet_train = pd.read_csv("train_x_discretization.csv")[['uid']+fs]
#40,000*81

#ranking features
rank_feature_score = pd.read_csv('rank_feature_score.csv')
rank_feature_score.shape
fs = list(rank_feature_score.feature[0:110])
rank_train = pd.read_csv("train_x_rank.csv")[['uid']+fs]
#40,000*111

rank_train = rank_train[fs] / float(len(rank_train))
rank_train['uid'] = pd.read_csv("train_x_rank.csv").uid

#raw feature
raw_feature_score = pd.read_csv('raw_feature_score.csv')
fs = list(raw_feature_score.feature[0:400])
raw_train = original[['uid']+fs+['y']]


#merge raw, ranking, discret and other 11 features
train = pd.merge(raw_train,rank_train,on='uid')
train = pd.merge(train,discreet_train,on='uid')
#
train = pd.merge(train,train_eleven,on='uid')
#unify all missing records to -1
train=train.fillna(-1)

#train 15,000 * 1,513
train, test_xy = train_test_split(train, test_size = 0.2)
test_y = test_xy[['uid']+['y']]
test_uid=test_xy['uid']
test= test_xy.drop(["uid",'y'],axis=1)


######################
#    xgb bagging     #
###################### 
#create randomness in the number of raw,ranking and discreet features 
#create randomness in the meta parameters of  

#by setting the number of feature from a random number from 300 to 500
#feature_num is such a variable

    ####################
    #       xgb        #
    ####################
def pipeline(iteration,random_seed,feature_num,rank_feature_num,discret_feature_num,gamma,max_depth,lambda_,subsample,colsample_bytree,min_child_weight):
    #define number of features as a variable feature_num
    raw_feature_selected = list(raw_feature_score.feature[0:feature_num])
    rank_feature_selected = list(rank_feature_score.feature[0:rank_feature_num])
    discreet_feature_selected = list(discreet_feature_score.feature[0:discret_feature_num])

    #construct training dataset from the randomly selected top features from raw, ranking, discret plus untouched 11
    train_xy = train[eleven_feature+raw_feature_selected+rank_feature_selected+discreet_feature_selected+['y']]

    test_x = test[eleven_feature+raw_feature_selected+rank_feature_selected+discreet_feature_selected]

    y = train_xy.y
    X = train_xy.drop(['y'],axis=1)
    

    dtest = xgb.DMatrix(test_x)
    dtrain = xgb.DMatrix(X, label=y)
    
    params={
    	'booster':'gbtree',
    	'objective': 'binary:logistic',
    	'scale_pos_weight': float(len(y)-sum(y))/float(sum(y)),
        'eval_metric': 'auc',
    	'gamma':gamma,
    	'max_depth':max_depth,
    	'lambda':lambda_,
        'subsample':subsample,
        'colsample_bytree':colsample_bytree,
        'min_child_weight':min_child_weight, 
        'eta': 0.08,
    	'seed':random_seed,
    	'nthread':8
        }
    
    watchlist  = [(dtrain,'train')]
    model = xgb.train(params,dtrain,num_boost_round=1500,evals=watchlist)
    model.save_model('./model/xgb{0}.model'.format(iteration))
    
    #predict test set
    test_y = model.predict(dtest)
    test_result = pd.DataFrame(test_uid,columns=["uid"])
    test_result["score"] = test_y
    test_result.to_csv("./preds/xgb{0}.csv".format(iteration),index=None,encoding='utf-8')
    
    #save feature score
    feature_score = model.get_fscore()
    feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
    fs = []
    for (key,value) in feature_score:
        fs.append("{0},{1}\n".format(key,value))
    
    with open('./featurescore/feature_score_{0}.csv'.format(iteration),'w') as f:
        f.writelines("feature,score\n")
        f.writelines(fs)



if __name__ == "__main__":
    
    random_seed = list(range(1000,2000,10))
    feature_num = list(range(200,400,2))
    rank_feature_num = list(range(60,110,2))
    discret_feature_num = list(range(50,80,1))
    gamma = [i/1000.0 for i in list(range(0,300,3))]
    max_depth = [6,7,8]
    lambda_ = list(range(500,700,2))
    subsample = [i/1000.0 for i in list(range(500,700,2))]
    colsample_bytree = [i/1000.0 for i in list(range(250,350,1))]
    min_child_weight = [i/1000.0 for i in list(range(250,550,3))]
    random.shuffle(rank_feature_num)
    random.shuffle(random_seed)
    random.shuffle(feature_num)
    random.shuffle(discret_feature_num)
    random.shuffle(gamma)
    random.shuffle(max_depth)
    random.shuffle(lambda_)
    random.shuffle(subsample)
    random.shuffle(colsample_bytree)
    random.shuffle(min_child_weight)
    
    with open('params.pkl','wb') as f:
        cPickle.dump((random_seed,feature_num,rank_feature_num,discret_feature_num,gamma,max_depth,lambda_,subsample,colsample_bytree,min_child_weight),f)
    
    
    for i in list(range(36)):
        print ("iter:",i)
        pipeline(i,random_seed[i],feature_num[i],rank_feature_num[i],discret_feature_num[i],gamma[i],max_depth[i%3],lambda_[i],subsample[i],colsample_bytree[i],min_child_weight[i])

    ##################################
    #  take average of xgb models    #
    ##################################

from sklearn.metrics import roc_auc_score

files = os.listdir('./preds')
pred = pd.read_csv('./preds/'+files[0])
uid = pred.uid
score = pred.score
for f in files[1:]:
    pred = pd.read_csv('./preds/'+f)
    score += pred.score

score /= len(files)

pred = pd.DataFrame(uid,columns=['uid'])
pred['score'] = score
pred.to_csv('avg_preds.csv',index=None,encoding='utf-8')

####cal auc

val_set = test_y
val_pred = pred
auc = roc_auc_score(val_set.y, val_pred.score.values)
print(auc)  




































