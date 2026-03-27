import sys
sys.path.append('../../ICMLA2020_12-lead-ECG-main/core')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import torch
from torch.utils.data import DataLoader as torch_dataloader
from torch.utils.data import Dataset as torch_dataset
from ClassBalancedSampler import ClassBalancedSampler
from sklearn.preprocessing import MaxAbsScaler
#%%
import os

def read_data(path,l):
    fileList = os.listdir(path)
    #print(fileList)
    used_name=path+fileList[0]
    s = pd.read_csv(used_name).values.shape[0]
    label=[l for i in range(s)]
    label=np.array(label)
    ecg_data=[[] for i in range(s)]
    fileList=sorted(fileList)
    for file in fileList:
            try:
                used_name = path + file
                data = pd.read_csv(used_name)
                data = data.values[:, 3:]
                # print(data)
                for i in range(s):
                    ecg_data[i].append(list(data[i]))
            except:
                pass
    for i in range(s):
        ecg_data[i]=np.mat(ecg_data[i]).T
    ecg_data=np.array(ecg_data)
    #print(path,ecg_data,ecg_data.shape)
    return ecg_data,label

def read_test_4class():
    path1,path2,path3,path4 = './hnu/test/4class/AVNRT/','./hnu/test/4class/AVRT-L/','./hnu/test/4class/AVRT-R/','./hnu/test/4class/N/'
    l1,l2,l3,l4=[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]
    train_4class_AVNRT,label_4class_AVNRT=read_data(path1,l1)
    train_4class_AVRTL, label_4class_AVRTL = read_data(path2, l2)
    train_4class_AVRTR, label_4class_AVRTR = read_data(path3, l3)
    train_4class_N, label_4class_N = read_data(path4, l4)
    label=np.vstack((label_4class_AVNRT, label_4class_AVRTL))
    label=np.vstack((label, label_4class_AVRTR))
    label = np.vstack((label, label_4class_N))
    train_set=np.vstack((train_4class_AVNRT,train_4class_AVRTL))
    train_set=np.vstack((train_set,train_4class_AVRTR))
    train_set = np.vstack((train_set, train_4class_N))
    return train_set,label

#return ecg_data,label
def read_data1(path,l):
    #path='./hundata2/test/4class/AVRT-L/'
    fileList = os.listdir(path)
    ecg_data=[]

    count1=0
    for file in fileList:
            try:
                used_name = path + file
                #data = pd.read_csv(used_name, encoding='gbk',usecols=['Ⅱ','V1','aVF', 'EB'])
                index_strings=['EB','Ⅰ','Ⅱ','Ⅲ','V1','V2','V3','V4','V5','V6','aVF','aVL','aVR']
                data = pd.read_csv(used_name, encoding='gbk',usecols=['Ⅰ','Ⅱ','Ⅲ','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6','EB'])[index_strings]
                a = data.values.shape[0]
                #print(z,used_name,a//5000)
                v = data.values
                count = 0
                for i in range(0, a, 5000):
                    count = count + 1
                    ecg_data.append(v[i:i+5000])
                count1=count1+count
            except:
                pass
    ecg_data = np.array(ecg_data)
    label=[l for i in range(count1)]
    label=np.array(label)
    return ecg_data,label


def read_test_4class1():
    path1,path2,path3,path4 = './hundata2/test/4class/AVNRT/','./hundata2/test/4class/AVRT-L/','./hundata2/test/4class/AVRT-R/','./hundata2/test/4class/N/'
    l1,l2,l3,l4=[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]
    train_4class_AVNRT,label_4class_AVNRT=read_data1(path1,l1)
    train_4class_AVRTL, label_4class_AVRTL = read_data1(path2, l2)
    train_4class_AVRTR, label_4class_AVRTR = read_data1(path3, l3)
    train_4class_N, label_4class_N = read_data1(path4, l4)
    label=np.vstack((label_4class_AVNRT, label_4class_AVRTL))
    label=np.vstack((label, label_4class_AVRTR))
    label = np.vstack((label, label_4class_N))
    train_set=np.vstack((train_4class_AVNRT,train_4class_AVRTL))
    train_set=np.vstack((train_set,train_4class_AVRTR))
    train_set = np.vstack((train_set, train_4class_N))
    return train_set,label


def read_data2(path,l):
    fileList = os.listdir(path)
    #print(fileList)
    used_name=path+fileList[0]
    s = pd.read_csv(used_name).values.shape[0]
    label=[l for i in range(s)]
    label=np.array(label)
    ecg_data=[[] for i in range(s)]
    fileList=sorted(fileList)
    for file in fileList:
            try:
                used_name = path + file
                data = pd.read_csv(used_name)
                data = data.values[:, 3:]
                # print(data)
                for i in range(s):
                    ecg_data[i].append(list(data[i]))
            except:
                pass
    for i in range(s):
        ecg_data[i]=np.mat(ecg_data[i]).T
    ecg_data=np.array(ecg_data)
    #print(path,ecg_data,ecg_data.shape)
    return ecg_data,label

def read_test_4class2():
    path1,path2,path3,path4 = './hnu/train/4class/AVNRT/','./hnu/train/4class/AVRT-L/','./hnu/train/4class/AVRT-R/','./hnu/train/4class/N/'
    l1,l2,l3,l4=[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]
    train_4class_AVNRT,label_4class_AVNRT=read_data2(path1,l1)
    train_4class_AVRTL, label_4class_AVRTL = read_data2(path2, l2)
    train_4class_AVRTR, label_4class_AVRTR = read_data2(path3, l3)
    train_4class_N, label_4class_N = read_data2(path4, l4)
    label=np.vstack((label_4class_AVNRT, label_4class_AVRTL))
    label=np.vstack((label, label_4class_AVRTR))
    label = np.vstack((label, label_4class_N))
    train_set=np.vstack((train_4class_AVNRT,train_4class_AVRTL))
    train_set=np.vstack((train_set,train_4class_AVRTR))
    train_set = np.vstack((train_set, train_4class_N))
    return train_set,label

#return ecg_data,label
def read_data3(path,l):
    #path='./hundata2/test/4class/AVRT-L/'
    fileList = os.listdir(path)
    ecg_data=[]

    count1=0
    for file in fileList:
            try:
                used_name = path + file
                #data = pd.read_csv(used_name, encoding='gbk',usecols=['Ⅱ','V1','aVF', 'EB'])
                index_strings=['EB','Ⅰ','Ⅱ','Ⅲ','V1','V2','V3','V4','V5','V6','aVF','aVL','aVR']
                data = pd.read_csv(used_name, encoding='gbk',usecols=[ 'Ⅰ', 'Ⅱ', 'Ⅲ', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6','EB'])[index_strings]
                a = data.values.shape[0]
                #print(z,used_name,a//5000)
                v = data.values
                count = 0
                for i in range(0, a, 5000):
                    count = count + 1
                    ecg_data.append(v[i:i+5000])
                count1=count1+count
            except:
                pass
    ecg_data = np.array(ecg_data)
    label=[l for i in range(count1)]
    label=np.array(label)
    return ecg_data,label


def read_test_4class3():
    path1,path2,path3,path4 = './hundata2/train/4class/AVNRT/','./hundata2/train/4class/AVRT-L/','./hundata2/train/4class/AVRT-R/','./hundata2/train/4class/N/'
    l1,l2,l3,l4=[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]
    train_4class_AVNRT,label_4class_AVNRT=read_data3(path1,l1)
    train_4class_AVRTL, label_4class_AVRTL = read_data3(path2, l2)
    train_4class_AVRTR, label_4class_AVRTR = read_data3(path3, l3)
    train_4class_N, label_4class_N = read_data3(path4, l4)
    label=np.vstack((label_4class_AVNRT, label_4class_AVRTL))
    label=np.vstack((label, label_4class_AVRTR))
    label = np.vstack((label, label_4class_N))
    train_set=np.vstack((train_4class_AVNRT,train_4class_AVRTL))
    train_set=np.vstack((train_set,train_4class_AVRTR))
    train_set = np.vstack((train_set, train_4class_N))
    return train_set,label

#%%
class MyDataset(torch_dataset):
    def __init__(self, signal, label, rand_pad):
        self.signal=signal
        self.label=label
        self.rand_pad=rand_pad
    def __len__(self):
        return len(self.signal)
    def __getitem__(self, idx):
        x = self.signal[idx].T
       
        # remove lead3,4,5,6
        #print('x,y:',x.shape,idx,self.label[idx]) #(12, 15435) 1109 8
        #m = [False, False, False,False,False, False, False, False,False,False,True,False,False]
        m = [True, True, True,True,True, True, True, True,True,True,True,True,True]
        x = x[m]
        #x = scaler.fit_transform(x.T).T
        x=torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.label[idx], dtype=torch.int64)
        #print('aaa',x.shape,y,mask.shape,idx) torch.Size([8, 33792]) tensor(2) torch.Size([1, 33792]) 2068
        #print('bbb',y[0],y.shape) #bbb tensor([1, 0, 0, 0, 0, 0, 0, 0, 0]) torch.Size([9])
        return x, y,idx
        #return x, y



#%%
def get_dataloader(batch_size=64, num_workers=0, rand_pad=False, path='../../data/CPSC2018/'):
    #df_train = pd.concat([df_train_0, df_test_0], ignore_index=True)
    print('load signal:', path)
    X1,y1=read_test_4class2()
    X2,y2=read_test_4class3()
    signal_train=np.vstack((X1,X2))
    label_train=np.vstack((y1,y2))
    print(signal_train.shape)
    X1,y1=read_test_4class()
    X2,y2=read_test_4class1()
    signal_test=np.vstack((X1,X2))
    label_test=np.vstack((y1,y2))
    print(signal_test.shape)
    print('load signal: completed')
    
    dataset_train = MyDataset(signal_train, label_train, rand_pad=rand_pad)
    dataset_test = MyDataset(signal_test, label_test, rand_pad=False)
    
 

    sampler_train = ClassBalancedSampler(label_train, True)
    loader_train = torch_dataloader(dataset_train, batch_size=batch_size,
                                    num_workers=num_workers, sampler=sampler_train, pin_memory=True)
    loader_test = torch_dataloader(dataset_test, batch_size=batch_size,
                                  num_workers=num_workers, shuffle=False, pin_memory=True)
    return loader_train,loader_test

#%%