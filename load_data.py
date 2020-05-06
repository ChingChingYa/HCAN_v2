# coding=utf-8
import pickle
import numpy as np 
import random
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


def load_data(filename):
    with open(filename, 'rb') as file:
        a_dict1 = pickle.load(file)

    return a_dict1

def load_data3(filename):
    with open(filename, 'rb') as file:
        result, doc, sents, words = pickle.load(file)

    return result, doc, sents, words  

def cut_data_len(data, field):
    result = []
    for d in data:
        if field == 'reviewerID':
            result.append(d[0])
        elif field == 'asin':
            result.append(d[1])
        else:
            result.append(d[2])

    return int(len(np.unique(result)))


def cut_data(data, field):
    result = []
    for d in data:
        if field == '0':
            result.append(d[0])
        elif field == '1':
            result.append(d[1])
        else:
            result.append(d[2])

    return result


def cut_dataset(filename):
    with open(filename, 'rb') as f:
        kf = KFold(n_splits=5)
        data = np.array(pickle.load(f))
    f.close()
    name = filename.split('.')
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        s = '.'+name[1]+str(i+1)+'.pickle'
        pickle.dump((data[train_index], data[test_index]),open(s, 'wb'))
        # train_x, dev_x = data_x[:int(length*0.8)], data_x[int(length*0.8)+1 :]
        # train_y, dev_y = data_y[:int(length*0.8)], data_y[int(length*0.8)+1 :]


def cut_dataset2(filename):
    x_data = []
    y_data = []
    with open(filename, 'rb') as f:
        kf = KFold(n_splits=5)
        a, b = (pickle.load(f))
        x_data = np.array(a)
        y_data = np.array(b)
    f.close()
    name = filename.split('.')
    for i, (train_index, test_index) in enumerate(kf.split(x_data)):
        s = '.'+name[1]+str(i+1)+'.pickle'
        pickle.dump((x_data[train_index], x_data[test_index], y_data[train_index], y_data[test_index]),open(s, 'wb'))
        # train_x, dev_x = data_x[:int(length*0.8)], data_x[int(length*0.8)+1 :]
        # train_y, dev_y = data_y[:int(length*0.8)], data_y[int(length*0.8)+1 :]


def read_dataset(filename):
    with open(filename, 'rb') as f:
        train, test = pickle.load(f)
    f.close()
    return train, test


def read_dataset2(filename):
    with open(filename, 'rb') as f:
        x_train, x_test, y_train, y_test = pickle.load(f)
        
    f.close()
    return x_train, x_test, y_train, y_test

