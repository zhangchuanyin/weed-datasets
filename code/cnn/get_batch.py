'''
Created on 2018-9-11

@author: 74510
'''
import os
import tensorflow as tf
from scipy.misc import imread,imresize
import numpy as np
import skimage
import skimage.io
import skimage.transform
import cv2 as cv
from PIL import Image
train_dir='train'

def get_files(file_dir):
    classDir=[]
    classLabel=[]
#     className=['Aslan','Esek','Inek','Kedi-Part1','Kopek-Part1','Koyun','Kurbaga','Kus-Part1','Maymun','Tavuk']
    className=['cier','huicai','suocao','yumi','zaoshuhe']
    for i in range(len(className)):
        classDir.append([])
        classLabel.append([])
    for i ,file in enumerate(os.listdir(file_dir)):
        for filename in os.listdir(os.path.join(file_dir,file,'train')):
            classDir[i].append(os.path.join(file_dir,file,'train',filename))
            classLabel[i].append(i)
#     print(classDir,'\n',classLabel)
    for i in range(len(classDir)-1):
        if i>0:
            classDir[i]=image_list
            classLabel[i]=label_list
        image_list=np.hstack((classDir[i],classDir[i+1]))
        label_list=np.hstack((classLabel[i],classLabel[i+1]))
    temp=np.array((image_list,label_list))
    temp=temp.transpose()#将矩阵翻转
#     np.random.shuffle(temp)
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(i) for i in label_list] 
    return image_list,label_list
def get_batch(image,label,image_w,image_h,batch_size,capacity):
    image=tf.cast(image,tf.string)
    label=tf.cast(label,tf.int32)
#从tensor列表中按顺序或随机抽取一个tensor
    input_queue=tf.train.slice_input_producer([image,label])
    label=input_queue[1]
    #将图片进行编码，以进行图片胡处理
    #将图像解码，不同类型的图像不能混在一起，要么只用jpeg，要么只用png等。
    image_content=tf.read_file(input_queue[0])
    image=tf.image.decode_jpeg(image_content,channels=3)
    image=tf.image.resize_image_with_crop_or_pad(image,image_w,image_h)
    image=tf.image.resize_images(image,[image_w,image_h])
    image=tf.image.per_image_standardization(image)  
#     image=tf.cast(image,dtype=tf.float32)
    image_batch,label_batch=tf.train.batch([image,label], batch_size,num_threads=16, capacity=capacity)  
    
    label_batch=tf.reshape(label_batch,[batch_size])
    return image_batch,label_batch
# image,label=get_files(train_dir)
# print(image,label)
#         
# print(len(image),image,len(label),label)
# def Z_ScoreNormalization(x):
#     mu = np.average(x)#均值
#     #sigma = np.std(x)#方差
#     #x = (x - mu) / sigma
#     #dis = x.max() - x.min()
#     #x = x / dis
#     x = x - mu
#     return x
# def normalimage(img):
#     # load image
#  
#     img = img / 255.0
#     assert (0 <= img).all() and (img <= 1.0).all()
#     # print "Original Image Shape: ", img.shape
#     # we crop image from center
#     short_edge = min(img.shape[:2])
#     yy = int((img.shape[0] - short_edge) / 2)
#     xx = int((img.shape[1] - short_edge) / 2)
#     crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
#     # resize to 224, 224
#     resized_img = skimage.transform.resize(crop_img, (224, 224))
#     return resized_img

