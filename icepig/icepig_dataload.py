
import tensorflow as tf
import numpy as np

import time
import datetime
import scipy.io as scio
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def Dataload_v3_Standard(filemath,train_size,test_size,random_state):
    '''本程序是为了用minmax方式归一化数据，单目标输出'''
    tempdata = scio.loadmat( filemath )  # 给文件内字典赋值定义
    tempdata_son1 = tempdata['DATASETS']  ##将字典tempdata内数据表n取出定义新变量
    # tempdata_son1 = tempdata_son1.T
    tf.cast(tempdata_son1, dtype=tf.float32, name=None)   #转换类型

    consize = np.shape(tempdata_son1)
    n1 = consize[0]  # 行数
    s1 = consize[1]  # 列数

    x_in = tempdata_son1[:, 0:(s1 - 1)]
    y_in = tempdata_son1[:, -1]

    train_x_disorder, test_x_disorder, train_y_disorder, test_y_disorder = train_test_split(x_in, y_in,
                                                                                            train_size=train_size,
                                                                                            test_size=test_size,
                                                                                            random_state=random_state)

    test_y_true = test_y_disorder
    train_y_disorder = train_y_disorder.reshape(-1, 1)
    test_y_disorder = test_y_disorder.reshape(-1, 1)


    scaler_train_x = StandardScaler()
    scaler_train_x.fit(train_x_disorder)  # 计算数据均值mean_ 和方差 var_  别忘记后面的  '_')
    #scaler_train_x_mean = scaler_train_x.mean_
    #scaler_train_x_var = scaler_train_x.var_
    train_x_disorder = scaler_train_x.transform(train_x_disorder)
    test_x_disorder = scaler_train_x.transform(test_x_disorder)

    scaler_train_y = StandardScaler()
    scaler_train_y.fit(train_y_disorder)
    #scaler_train_y_mean = scaler_train_y.mean_
    #scaler_train_y_var = scaler_train_y.var_
    train_y_disorder = scaler_train_y.transform(train_y_disorder)
    test_y_disorder = scaler_train_y.transform(test_y_disorder)



    x_train, y_train = train_x_disorder, train_y_disorder
    x_test, y_test = test_x_disorder, test_y_disorder

    x_val, x_val_rest, y_val, y_val_rest = train_test_split(x_train, y_train,
                                                            train_size=0.25,
                                                            test_size=0.1, random_state=None)
    scaler_x = scaler_train_x
    scaler_y = scaler_train_y
    time_lable = datetime.datetime.now().strftime('%Y-%m-%d-%H')
    unit = [x_train,y_train, x_test, y_test, x_val, y_val, scaler_x, scaler_y]

    return (unit)


def Dataload_v3_MinMax(filemath,train_size,test_size,random_state):
    '''本程序是为了用minmax方式归一化数据，单目标输出'''
    tempdata = scio.loadmat( filemath )  # 给文件内字典赋值定义
    tempdata_son1 = tempdata['DATASETS']  ##将字典tempdata内数据表n取出定义新变量
    # tempdata_son1 = tempdata_son1.T
    tf.cast(tempdata_son1, dtype=tf.float32, name=None)

    consize = np.shape(tempdata_son1)
    n1 = consize[0]  # 行数
    s1 = consize[1]  # 列数

    x_in = tempdata_son1[:, 0:(s1 - 1)]
    y_in = tempdata_son1[:, -1]

    train_x_disorder, test_x_disorder, train_y_disorder, test_y_disorder = train_test_split(x_in, y_in,
                                                                                            train_size=train_size,
                                                                                            test_size=test_size,
                                                                                            random_state=random_state)

    test_y_true = test_y_disorder
    train_y_disorder = train_y_disorder.reshape(-1, 1)
    test_y_disorder = test_y_disorder.reshape(-1, 1)


    scaler_train_x = MinMaxScaler()
    scaler_train_x.fit(train_x_disorder)  # 计算数据均值mean_ 和方差 var_  别忘记后面的  '_')
    #scaler_train_x_mean = scaler_train_x.mean_
    #scaler_train_x_var = scaler_train_x.var_
    train_x_disorder = scaler_train_x.transform(train_x_disorder)
    test_x_disorder = scaler_train_x.transform(test_x_disorder)

    scaler_train_y = MinMaxScaler()
    scaler_train_y.fit(train_y_disorder)
    #scaler_train_y_mean = scaler_train_y.mean_
    #scaler_train_y_var = scaler_train_y.var_
    train_y_disorder = scaler_train_y.transform(train_y_disorder)
    test_y_disorder = scaler_train_y.transform(test_y_disorder)


    x_train, y_train = train_x_disorder, train_y_disorder
    x_test, y_test = test_x_disorder, test_y_disorder

    x_val, x_val_rest, y_val, y_val_rest = train_test_split(x_train, y_train,
                                                            train_size=0.25,
                                                            test_size=0.1, random_state=None)
    scaler_x = scaler_train_x
    scaler_y = scaler_train_y
    time_lable = datetime.datetime.now().strftime('%Y-%m-%d-%H')

    unit = [x_train,y_train, x_test,  y_test, x_val, y_val, scaler_x, scaler_y]

    return (unit)


def Dataload_v3_None(filemath,train_size,test_size,random_state):
    '''本程序是为没有归一化数据，单目标输出'''
    tempdata = scio.loadmat( filemath )  # 给文件内字典赋值定义
    tempdata_son1 = tempdata['DATASETS']  ##将字典tempdata内数据表n取出定义新变量
    # tempdata_son1 = tempdata_son1.T
    tf.cast(tempdata_son1, dtype=tf.float32, name=None)

    consize = np.shape(tempdata_son1)
    n1 = consize[0]  # 行数
    s1 = consize[1]  # 列数

    x_in = tempdata_son1[:, 0:(s1 - 1)]
    y_in = tempdata_son1[:, -1]

    train_x_disorder, test_x_disorder, train_y_disorder, test_y_disorder = train_test_split(x_in, y_in,
             train_size=train_size,  test_size=test_size,     random_state=random_state)


    x_train, y_train = train_x_disorder, train_y_disorder
    x_test, y_test = test_x_disorder, test_y_disorder

    x_val, x_val_rest, y_val, y_val_rest = train_test_split(x_train, y_train,
                                                            train_size=0.25,
                                                            test_size=0.1, random_state=None)

    time_lable = datetime.datetime.now().strftime('%Y-%m-%d-%H')
    unit = [x_train,y_train, x_test,  y_test, x_val, y_val]

    return (unit)




def Dataload_mul_MinMax(filemath,aim_num, train_size,test_size,random_state):
    '''本程序是为了用minmax方式归一化数据，可以多目标输出'''
    tempdata = scio.loadmat( filemath )  # 给文件内字典赋值定义
    tempdata_son1 = tempdata['DATASETS']  ##将字典tempdata内数据表n取出定义新变量
    # tempdata_son1 = tempdata_son1.T
    tf.cast(tempdata_son1, dtype=tf.float32, name=None)

    consize = np.shape(tempdata_son1)
    n1 = consize[0]  # 行数
    s1 = consize[1]  # 列数

    x_in = tempdata_son1[:, 0:(s1 - aim_num)]
    y_in = tempdata_son1[:, (s1 - aim_num):]

    train_x_disorder, test_x_disorder, train_y_disorder, test_y_disorder = train_test_split(x_in, y_in,
              train_size=train_size,      test_size=test_size,      random_state=random_state)



    scaler_train_x = MinMaxScaler()
    scaler_train_x.fit(train_x_disorder)  # 计算数据均值mean_ 和方差 var_  别忘记后面的  '_')
    #scaler_train_x_mean = scaler_train_x.mean_
    #scaler_train_x_var = scaler_train_x.var_
    train_x_disorder = scaler_train_x.transform(train_x_disorder)
    test_x_disorder = scaler_train_x.transform(test_x_disorder)

    scaler_train_y = MinMaxScaler()
    scaler_train_y.fit(train_y_disorder)
    #scaler_train_y_mean = scaler_train_y.mean_
    #scaler_train_y_var = scaler_train_y.var_
    train_y_disorder = scaler_train_y.transform(train_y_disorder)
    test_y_disorder = scaler_train_y.transform(test_y_disorder)


    x_train, y_train = train_x_disorder, train_y_disorder
    x_test, y_test = test_x_disorder, test_y_disorder

    x_val, x_val_rest, y_val, y_val_rest = train_test_split(x_train, y_train,
           train_size=0.25,     test_size=0.25, random_state=None)

    scaler_x = scaler_train_x
    scaler_y = scaler_train_y
    time_lable = datetime.datetime.now().strftime('%Y-%m-%d-%H')

    unit = [x_train, y_train, x_test,  y_test, x_val, y_val, scaler_x, scaler_y]

    return (unit)


def Dataload_mul_Standard(filemath,aim_num, train_size,test_size,random_state):
    '''本程序是为了用Standard方式归一化数据，可以多目标输出'''
    tempdata = scio.loadmat( filemath )  # 给文件内字典赋值定义
    tempdata_son1 = tempdata['DATASETS']  ##将字典tempdata内数据表n取出定义新变量
    # tempdata_son1 = tempdata_son1.T
    tf.cast(tempdata_son1, dtype=tf.float32, name=None)

    consize = np.shape(tempdata_son1)
    n1 = consize[0]  # 行数
    s1 = consize[1]  # 列数

    x_in = tempdata_son1[:, 0:(s1 - aim_num)]
    y_in = tempdata_son1[:, (s1 - aim_num):]

    train_x_disorder, test_x_disorder, train_y_disorder, test_y_disorder = train_test_split(x_in, y_in,
              train_size=train_size,      test_size=test_size,      random_state=random_state)



    scaler_train_x = StandardScaler()
    scaler_train_x.fit(train_x_disorder)  # 计算数据均值mean_ 和方差 var_  别忘记后面的  '_')
    #scaler_train_x_mean = scaler_train_x.mean_
    #scaler_train_x_var = scaler_train_x.var_
    train_x_disorder = scaler_train_x.transform(train_x_disorder)
    test_x_disorder = scaler_train_x.transform(test_x_disorder)

    scaler_train_y = StandardScaler()
    scaler_train_y.fit(train_y_disorder)
    #scaler_train_y_mean = scaler_train_y.mean_
    #scaler_train_y_var = scaler_train_y.var_
    train_y_disorder = scaler_train_y.transform(train_y_disorder)
    test_y_disorder = scaler_train_y.transform(test_y_disorder)


    x_train, y_train = train_x_disorder, train_y_disorder
    x_test, y_test = test_x_disorder, test_y_disorder

    x_val, x_val_rest, y_val, y_val_rest = train_test_split(x_train, y_train,
           train_size=0.25,     test_size=0.25, random_state=None)

    scaler_x = scaler_train_x
    scaler_y = scaler_train_y
    time_lable = datetime.datetime.now().strftime('%Y-%m-%d-%H')

    unit = [x_train, y_train, x_test,  y_test, x_val, y_val, scaler_x, scaler_y]

    return (unit)



def Dataload_v3_mul_None(filemath,aim_num, train_size,test_size,random_state):
    '''本程序是为没有归一化数据，多目标输出'''
    tempdata = scio.loadmat( filemath )  # 给文件内字典赋值定义
    tempdata_son1 = tempdata['DATASETS']  ##将字典tempdata内数据表n取出定义新变量
    # tempdata_son1 = tempdata_son1.T
    tf.cast(tempdata_son1, dtype=tf.float32, name=None)

    consize = np.shape(tempdata_son1)
    n1 = consize[0]  # 行数
    s1 = consize[1]  # 列数

    x_in = tempdata_son1[:, 0:(s1 - aim_num)]
    y_in = tempdata_son1[:, (s1 - aim_num):]

    train_x_disorder, test_x_disorder, train_y_disorder, test_y_disorder = train_test_split(x_in, y_in,
             train_size=train_size,  test_size=test_size,     random_state=random_state)


    x_train, y_train = train_x_disorder, train_y_disorder
    x_test, y_test = test_x_disorder, test_y_disorder

    x_val, x_val_rest, y_val, y_val_rest = train_test_split(x_train, y_train,
                                                            train_size=0.25,
                                                            test_size=0.1, random_state=None)

    time_lable = datetime.datetime.now().strftime('%Y-%m-%d-%H')
    unit = [x_train,y_train, x_test,  y_test, x_val, y_val]

    return (unit)