import os
import tensorflow as tf
from scipy.misc import imread,imresize
import numpy as np
import skimage
import skimage.io
import skimage.transform
import cv2 as cv
from PIL import Image
train_dir=r'D:\workspace\net\raw\44'
def get_files_carrort(file_dir):
    for file in os.walk(file_dir):
        print(file)  #获取file_dir的全部目录
    carrort=[]
    weed=[]
    for file in os.listdir(file_dir):
        if file=='carrort':
            base_file=file_dir+file
            for file in os.listdir(base_file):
                carrort.append(base_file+'/'+file)
                img=Image.open(base_file+'/'+file)
                img_xz1=img.rotate(35)
                img_xz2=img.rotate(70)
                img_xz3=img.rotate(105)
                img_xz4=img.rotate(145)
                fname=file.split('.')[0]
                print(fname)
                img_xz1.save(base_file+'/'+fname+'_xz1'+'.png')
                img_xz2.save(base_file+'/'+fname+'_xz2'+'.png')
                img_xz3.save(base_file+'/'+fname+'_xz3'+'.png')
                img_xz4.save(base_file+'/'+fname+'_xz4'+'.png')
                print(fname,'saving...........')
        elif file=='weeds':
            base_file=file_dir+file
            for file in os.listdir(base_file):
                weed.append(base_file+'/'+file)
                img=Image.open(base_file+'/'+file)
                img_xz1=img.rotate(45)
                img_xz2=img.rotate(90)
                img_xz3=img.rotate(135)
                img_xz4=img.rotate(180)
                fname=file.split('.')[0]
                print(fname)
                img_xz1.save(base_file+'/'+fname+'_xz1'+'.png')
                img_xz2.save(base_file+'/'+fname+'_xz2'+'.png')
                img_xz3.save(base_file+'/'+fname+'_xz3'+'.png')
                img_xz4.save(base_file+'/'+fname+'_xz4'+'.png')
                print(fname,'saving...........')
def enhance(data_dir):
    for file in os.listdir(data_dir):
#                 carrort.append(data_dir+'/'+file)
                img=Image.open(data_dir+'/'+file)
                img_xz1=img.rotate(45)
                img_xz2=img.rotate(90)
                img_xz3=img.rotate(135)
                img_xz4=img.rotate(180)
                fname=file.split('.')[0]
                print(fname)
                img_xz1.save(data_dir+'/'+'xz1'+fname+'.tif')
                img_xz2.save(data_dir+'/'+'xz2'+fname+'.tif')
                img_xz3.save(data_dir+'/'+'xz3'+fname+'.tif')
                img_xz4.save(data_dir+'/'+'xz4'+fname+'.tif')
                print(fname,'saving...........')          
       
enhance(train_dir)