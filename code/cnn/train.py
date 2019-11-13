'''
Created on 2018-9-11

@author: 74510
'''
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import alexnet 
import vgg16
import resnet_101
from get_batch import *
N_CLASSES =5
IMG_W =224
IMG_H =224
BATCH_SIZE =8
CAPACITY = 100
MAX_STEP =50000
learning_rate = 0.0001

x=tf.placeholder(tf.float32,shape=[BATCH_SIZE,IMG_W,IMG_H,3],name='input')
# x=tf.convert_to_tensor(x)
y=tf.placeholder(tf.int32,shape=[BATCH_SIZE],name='input')
def run_training():
    train_dir = 'train/'
    logs_train_dir = 'logresnet_101/'
    train,train_label = get_files(train_dir)
    print(len(train),train_label)
    train_batch,train_label_batch = get_batch(train,train_label,
                                                         IMG_W,
                                                         IMG_H, 
                                                         BATCH_SIZE,
                                                         CAPACITY)
    train_logits = resnet_101.inference(train_batch,BATCH_SIZE,N_CLASSES,train=True)
    train_loss = resnet_101.losses(train_logits,train_label_batch)
    train_op = resnet_101.training(train_loss,learning_rate)
    train_acc = resnet_101.evaluation(train_logits,train_label_batch)
     
    summary_op = tf.summary.merge_all()#自动管理
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir,sess.graph)
    saver = tf.train.Saver()
    
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess,coord = coord)
    los=[]
    epoch=[]
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _,tra_loss,tra_acc = sess.run([train_op,train_loss,train_acc])
            epoch.append(step)
            los.append(tra_loss)
#             with open('loss.txt' ,'a') as f:
#                 f.write('Step %d,train loss = %.2f,train occuracy = %.2f%%'%(step,tra_loss,tra_acc*100)+'\n')
            if step %  50 == 0:
                print('Step %d,train loss = %.2f,train occuracy = %.2f%%'%(step,tra_loss,tra_acc*100))
                summary_str = sess.run(summary_op)
                #将所有的日志写入文件，TensorFlow程序就可以那这次运行日志文件，进行各种信息的可视化
                train_writer.add_summary(summary_str,step)
            if step % 100 ==0 or (step +1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir,'model.ckpt')
                saver.save(sess,checkpoint_path,global_step = step)
    except tf.errors.OutOfRangeError:
        print('Done training epoch limit reached')
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()
    plt.plot(epoch,los)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('train loss')
    plt.show()
if __name__ == '__main__':
    run_training()