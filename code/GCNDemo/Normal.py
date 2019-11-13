'''
Created on 2018-12-5

@author: 南城
'''
from openpyxl import workbook
import numpy as np
import pandas as pd
import scipy.io as scio
import time
# dataSets=pd.read_excel('feature.xlsx')
# dataSets=np.array(dataSets)
def createAdj(dataSets):
    A=np.zeros([dataSets.shape[0],dataSets.shape[0]])
    for i in range(dataSets.shape[0]):
        for j in range(i+1):
            distance1=(dataSets[i]-dataSets[j])
            distance2=distance1.transpose()
            temp=np.matmul(distance2,distance1)+1
            A[i,j]=1/temp
#             A[i,j]=np.mat(temp).I
            A[j,i]=A[i,j]       
    return A
def createDeg(Mat):
    A=np.zeros(Mat.shape)
    for item in range(Mat.shape[0]):
        A[item,item]=np.sum(Mat[item])
    return A
def createLaplacian(Mat):
    I=np.eye(Mat.shape[0])
    A_=Mat+I
    D=createDeg(A_)
    D_=np.mat(D**(0.5)).I
    L=np.dot(np.dot(D_,A_),D_)
    return L
def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)].T
def loadData():
#     A=pd.read_excel('../excel/features-corn.xlsx')
#     A=np.array(A)
#     labels=pd.read_excel('../excel/labelscorn.xlsx')
#     labels=np.array(labels)
#     adj=createAdj(A)
#     A=pd.read_excel('../excel/Esc-animaldata.xlsx')
#     A = np.array(A)
#     x= scio.loadmat('melspectrograms/ResNet101_bot_melfea_pre_train.mat')
#     print(x)
    x= scio.loadmat('melspectrograms/ResNet101_bot_melfea_pre_train.mat')['arry'][ : ,0:4096]
    tx= scio.loadmat('melspectrograms/ResNet101_bot_melfea_pre_test.mat')['arry'][ : ,0:4096]
    allx=np.vstack((x,tx))
#     allx=scio.loadmat('melspectrograms/')['zongdata']
#     print(allx.shape)
#     ally=scio.loadmat('melspectrograms/zonglabel.mat')['zonglabel']
#     print(ally.shape)
#     ally=np.eye(ally.shape[1],10)[y][0]
#     print(y[0])
#     y=y.transpose()
    y=scio.loadmat('melspectrograms/trainlabel.mat')['arry']
    y=np.eye(y.shape[1],10)[y][0]
    ty=scio.loadmat('melspectrograms/testlabel.mat')['arry']
# #     print(ty.shape)
    ty=np.eye(ty.shape[1],10)[ty][0]
# #     ty=ty.transpose()
    ally=np.vstack((y,ty))
    print(ally,ally.shape,allx,allx.shape)
#     A=A['allvggmelfeature']
#     labels=pd.read_excel('../excel/Esc-animallabel2.xlsx')
#     labels = scio.loadmat('../excel/alllabel.mat')
#     labels=np.array(labels)
#     #labels = labels['a']
#     print(A.shape[0])
    time_begin=time.localtime(time.time())
    print('begin time:','{0}-{1}-{2} {3}:{4}'.format(list(time_begin)[0],list(time_begin)[1],list(time_begin)[2],list(time_begin)[3],list(time_begin)[4]))
    adj=createAdj(allx)
    time_end=time.localtime(time.time())
    print('end time:','{0}-{1}-{2} {3}:{4}'.format(list(time_end)[0],list(time_end)[1],list(time_end)[2],list(time_end)[3],list(time_end)[4]))
    L=createLaplacian(adj)
    return L,allx,ally
# A=createAdj(dataSets)
# L=createLaplacian(A)
# print(L[4,5],L[5,4])
# L,A,labels=loadData()
# print(L,A,labels)
L,allx,ally=loadData()
# print(L.shape,allx.shape,ally.shape )