import pandas as pd
import numpy as np
import re
import tokenize
import sklearn
from sklearn.naive_bayes import *
import string
from pandas.compat import StringIO
from sklearn.metrics import f1_score

def read_file_by_rate(filename):
    with open(filename,encoding="utf8") as f:
        data = f.read()
        translator = str.maketrans('','', string.punctuation)
        data = data.translate(translator)
        train = pd.read_csv(StringIO(data),delimiter="\t",engine='python',header= None, names = ["Review","Class"])
        return train.astype(str).apply(lambda x: x.str.lower())

def convert_to_binary_BOW_representation(data):

        x_star = np.zeros((len(data),10000),dtype=int)
    
        for i in range(len(data)):
                x= np.fromstring(data[i],dtype=int , sep=" ")
                for j in range(len(x)):
                        x_star[i][int(x[j])-1]=1
        return x_star

def compute_f1_score(traintokenlist,testtokenlist):
        
        np_train_x = np.asarray(traintokenlist.iloc[:]["Review"])
        np_train_y = np.asarray(traintokenlist.iloc[:]["Class"])
        x_binary_train = convert_to_binary_BOW_representation(np_train_x)

        clf = BernoulliNB()
        clf.fit(x_binary_train,np_train_y)
        BernoulliNB(alpha = 1)

        np_test_x = np.asarray(testtokenlist.iloc[:]["Review"])
        np_test_y = np.asarray(testtokenlist.iloc[:]["Class"])
        x_binary_test = convert_to_binary_BOW_representation(np_test_x)

        predict_y = clf.predict(x_binary_test)
        #print(predict_y)
        return f1_score(np_test_y,predict_y,labels=None,average='micro')

if __name__ == '__main__':
    yelp_train_data_filename = "./yelp-train.txt"
    yelp_train_token_list = read_file_by_rate(yelp_train_data_filename)
    yelp_data_valid_filename = "./yelp-valid.txt"
    yelp_valid_token_list = read_file_by_rate(yelp_data_valid_filename)

    yelp_valid_f1_score_uniform = compute_f1_score(yelp_train_token_list,yelp_valid_token_list)
    print("the unifrom classifier f1 score is %f" %yelp_valid_f1_score_uniform)

    '''
    #print(dataset)
    yelp_train_x, yelp_train_y = process_data(dataset)
    #print(yelp_train_x)
    clf = DummyClassifier(strategy='uniform',random_state=0)
    clf.fit(yelp_train_x,yelp_train_y)
    #print(type(yelp_train_y))
    #clf.score(yelp_train_x,yelp_train_y)
    #predict_y = clf.score
    '''