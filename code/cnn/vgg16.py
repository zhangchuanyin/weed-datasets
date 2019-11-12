'''
Created on 2018-9-11

@author: 74510
'''
import tensorflow as tf

def get_variableConv(input,name,shape):
    with tf.variable_scope(name) as scop:
        weights=tf.get_variable('weights',shape=shape ,#前两个为过滤器尺寸，第三个为当前层深度，第四个为过滤器纵深高度
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))#正态分布
        biases=tf.get_variable('biases',shape=shape[-1],dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))
        conv=tf.nn.conv2d(input,weights,strides=[1,1,1,1],padding='SAME')#实现卷积累重要的核心函数，四维为步数
        pre_activation=tf.nn.bias_add(conv,biases)
        conv=tf.nn.relu(pre_activation,name=name)
    return conv
def get_variablePool(input,name):
    with tf.variable_scope(name)as scop:
        #ksize第一个跟最后i一个必须为1 ，通常给的过滤器尺寸为[1，2，2，1],[1,3,3,1]第二个四维为步长
        pool=tf.nn.max_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
        #防止激励函数之后过度耦合的现象
        pool=tf.nn.lrn(pool, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name=name)
    return  pool 
def get_variableFull(input,name,batch_size,shape):
    with tf.variable_scope(name) as scop:
         
        weights = tf.get_variable("weights",
                                  shape=shape,
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable("biases",
                                 shape=shape[-1],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fc=tf.matmul(input,weights)
        fc=tf.add(fc,biases)
        return fc
def inference(input,batch_size,class_num,train=True):
    #卷积层1
    conv1_1= get_variableConv(input,'conv1_1',shape=[3,3,3,64])
    conv1_1=tf.layers.batch_normalization(conv1_1,training=train)
    conv1_2= get_variableConv(conv1_1,'conv1_2',shape=[3,3,64,64])
    conv1_2=tf.layers.batch_normalization(conv1_2,training=train)    #池化层1
    pooling1=get_variablePool(conv1_2,'pooling1')
    #卷积层2
    conv2_1= get_variableConv(pooling1,'conv2_1',shape=[3,3,64,128])
    conv2_1=tf.layers.batch_normalization(conv2_1,training=train)
    conv2_2= get_variableConv(conv2_1,'conv2_2',shape=[3,3,128,128])
    conv2_2=tf.layers.batch_normalization(conv2_2,training=train)
    #池化层2
    pooling2=get_variablePool(conv2_2,'pooling2')
    #卷积层3
    conv3_1= get_variableConv(pooling2,'conv3_1',shape=[3,3,128,256])
    conv3_1=tf.layers.batch_normalization(conv3_1,training=train)
    conv3_2= get_variableConv(conv3_1,'conv3_2',shape=[3,3,256,256])
    conv3_2=tf.layers.batch_normalization(conv3_2,training=train)
    conv3_3= get_variableConv(conv3_2,'conv3_3',shape=[3,3,256,256])
    conv3_3=tf.layers.batch_normalization(conv3_3,training=train)
#     #池化层3
    pooling3=get_variablePool(conv3_3,'pooling3')
#     #卷积层4
    conv4_1= get_variableConv(pooling3,'conv4_1',shape=[3,3,256,512])
    conv4_1=tf.layers.batch_normalization(conv4_1,training=train)
    conv4_2= get_variableConv(conv4_1,'conv4_2',shape=[3,3,512,512])
    conv4_2=tf.layers.batch_normalization(conv4_2,training=train)
    conv4_3= get_variableConv(conv4_2,'conv4_3',shape=[3,3,512,512])
    conv4_3=tf.layers.batch_normalization(conv4_3,training=train)
#     
#     #池化层4
    pooling4=get_variablePool(conv4_3,'pooling4')
#     #卷积层5
    conv5_1= get_variableConv(pooling4,'conv5_1',shape=[3,3,512,512])
    conv5_1=tf.layers.batch_normalization(conv5_1,training=train)
    conv5_2= get_variableConv(conv5_1,'conv5_2',shape=[3,3,512,512])
    conv5_2=tf.layers.batch_normalization(conv5_2,training=train)
    conv5_3= get_variableConv(conv5_2,'conv5_3',shape=[3,3,512,512])
    conv5_3=tf.layers.batch_normalization(conv5_3,training=train)
#     #池化层5
    pooling5=get_variablePool(conv5_3,'pooling5')
    #全连接层full-connection
    pooling5=tf.reshape(pooling5,[batch_size,-1])
    print(pooling5.get_shape())
    dim=pooling5.get_shape()[1].value
    fc1=get_variableFull(pooling5, 'fullc1',batch_size,shape=[dim,4096])
    fc1=tf.nn.relu(fc1)
    if train==True:
        fc1=tf.nn.dropout(fc1, 0.5, name="fc1_drop")
    fc2=get_variableFull(fc1, 'fullc2',batch_size,shape=[4096,4096])
    fc2=tf.nn.relu(fc2)
    if train==True:
        fc2=tf.nn.dropout(fc2, 0.5, name="fc2_drop")
    fc3=get_variableFull(fc2, 'fullc3',batch_size,shape=[4096,1000])
    if train==True:
        fc3=tf.nn.dropout(fc3, 0.5, name="fc3_drop")
    fc4=get_variableFull(fc3, 'fullc4',batch_size,shape=[1000,class_num])
#     fc4=tf.nn.dropout(fc4, 0.5, name="fc4_drop")

#     prob=tf.nn.softmax(fc4)
    return fc4
def losses(logits,label):
    with tf.variable_scope("loss") as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=label, name="xentropy_per_example")
        loss = tf.reduce_mean(cross_entropy, name="loss")
        tf.summary.scalar(scope.name + "loss", loss)
    return loss
def training(loss,learning_rate):
    with tf.variable_scope('optimizer') as scope:
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step=tf.Variable(0,name='global_step',trainable=False)
        train_op=optimizer.minimize(loss, global_step=global_step)    
    return train_op
def evaluation(logits, labels):
    with tf.variable_scope("accuracy") as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + "accuracy", accuracy)
    return accuracy
