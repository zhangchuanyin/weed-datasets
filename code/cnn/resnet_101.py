'''
Created on 2019-9-4

@author: 南城
'''
'''
Created on 2018-9-11

@author: 74510
'''
import tensorflow as tf
parameters=[]
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list()) 
def get_variableConv(input,name,shape):
    with tf.variable_scope(name) as scop:
        weights=tf.get_variable('weights',shape=shape ,#前两个为过滤器尺寸，第三个为当前层深度，第四个为过滤器纵深高度
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))#正态分布
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(1e-4)(weights))
        biases=tf.get_variable('biases',shape=shape[-1],dtype=tf.float32,
                               initializer=tf.constant_initializer(0.1))
        conv=tf.nn.conv2d(input,weights,strides=[1,2,2,1],padding='SAME')#实现卷积累重要的核心函数，四维为步数
        pre_activation=tf.nn.bias_add(conv,biases)
        conv=tf.layers.batch_normalization(conv)
        conv=tf.nn.relu(pre_activation,name=name)
        print_activations(conv)
       
    return conv
def get_variablePool(input,name):
    with tf.variable_scope(name)as scop:
        #ksize第一个跟最后i一个必须为1 ，通常给的过滤器尺寸为[1，2，2，1],[1,3,3,1]第二个四维为步长
        pool=tf.nn.max_pool(input,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
        #防止激励函数之后过度耦合的现象
        pool=tf.nn.lrn(pool, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name=name)
        print_activations(pool)
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
def resnetBlock(input,name,shape):
    a,b,c=name
    shape1,shape2,shape3=shape
    conv_a=get_variableConv(input,a,shape1)
    conv_a=tf.layers.batch_normalization(conv_a)
    conv_b=get_variableConv(conv_a,b,shape2)
    conv_b=tf.layers.batch_normalization(conv_b)
    conv_c=get_variableConv(conv_b, c, shape3)
    conv_c=tf.layers.batch_normalization(conv_c)
    conv=tf.add(conv_c,input)
    conv=tf.nn.relu(conv)
    return conv
    
def inference(input,batch_size,class_num,train=True):
   
    #conv1
    conv1_1= get_variableConv(input,'conv1_1',shape=[7,7,3,64])
    
    #conv2 
    pooling1=get_variablePool(conv1_1, 'pool_1')
    conv2= get_variableConv(pooling1,'conv2_1',shape=[3,3,64,64])
    conv2=tf.layers.batch_normalization(conv2)
    conv2= get_variableConv(conv2,'conv2_2',shape=[3,3,64,64])
    conv2=tf.layers.batch_normalization(conv2)
    conv2= get_variableConv(conv2,'conv2_3',shape=[3,3,64,64])
    conv2=tf.layers.batch_normalization(conv2)
    conv2= get_variableConv(conv2,'conv2_4',shape=[3,3,64,64])
    conv2=tf.layers.batch_normalization(conv2)
    pooling2=get_variablePool(conv2, 'poo2_1')
    conv2= get_variableConv(pooling2,'conv2_5',shape=[3,3,64,64])
    conv2=tf.layers.batch_normalization(conv2)
    #conv3
    conv3_1=resnetBlock(conv2,['conv3_1_a','conv3_1_b','conv3_1_c'],[[1,1,64,64],[3,3,64,256],[1,1,256,64]])
    conv3_2=resnetBlock(conv3_1,['conv3_2_a','conv3_2_b','conv3_2_c'],[[1,1,64,64],[3,3,64,256],[1,1,256,64]])
    conv3_3=resnetBlock(conv3_2,['conv3_3_a','conv3_3_b','conv3_3_c'],[[1,1,64,64],[3,3,64,256],[1,1,256,64]])
    #conv4
    conv4= get_variableConv(conv3_3,'con4_0',shape=[3,3,64,128])
    conv4=tf.layers.batch_normalization(conv4)
    conv4_1=resnetBlock(conv4,['conv4_1_a','conv4_1_b','conv4_1_c'],[[1,1,128,128],[3,3,128,512],[1,1,512,128]])
    conv4_2=resnetBlock(conv4_1,['conv4_2_a','conv4_2_b','conv4_2_c'],[[1,1,128,128],[3,3,128,512],[1,1,512,128]])
    conv4_3=resnetBlock(conv4_2,['conv4_3_a','conv4_3_b','conv4_3_c'],[[1,1,128,128],[3,3,128,512],[1,1,512,128]])
    conv4_4=resnetBlock(conv4_3,['conv4_4_a','conv4_4_b','conv4_4_c'],[[1,1,128,128],[3,3,128,512],[1,1,512,128]])
    #conv5
    conv5= get_variableConv(conv4_4,'con5_0',shape=[3,3,128,256])
    conv5=tf.layers.batch_normalization(conv5)
    for i in range(1,2):
        conv5=resnetBlock(conv5,['conv5_{}_a'.format(i),'conv5_{}_b'.format(i),'conv5_{}_c'.format(i)],[[1,1,256,256],[3,3,256,1024],[1,1,1024,256]])
    #conv6
    conv6= get_variableConv(conv5,'con6_0',shape=[3,3,256,512])
    conv6=tf.layers.batch_normalization(conv6)
    for i in range(1,2):
        conv6=resnetBlock(conv6,['conv6_{}_a'.format(i),'conv6_{}_b'.format(i),'conv6_{}_c'.format(i)],[[1,1,512,512],[3,3,512,2048],[1,1,2048,512]])

    # full-connection
    conv6=tf.reshape(conv6,[batch_size,-1])
    print('conv6',conv6.get_shape())
    dim=conv6.get_shape()[1].value
    fc1=get_variableFull(conv6, 'fullc1',batch_size,shape=[dim,512])
    fc1=tf.nn.relu(fc1)
    if train==True:
        fc1=tf.nn.dropout(fc1, 0.5, name="fc1_drop")
    fc2=get_variableFull(fc1, 'fullc2',batch_size,shape=[512,class_num])
#     fc2=tf.nn.relu(fc2)
#     if train==True:
#         fc2=tf.nn.dropout(fc2, 0.5, name="fc2_drop")
#     fc3=get_variableFull(fc2, 'fullc3',batch_size,shape=[256,64])
#     if train==True:
#         fc3=tf.nn.dropout(fc3, 0.5, name="fc3_drop")
#     fc4=get_variableFull(fc3, 'fullc4',batch_size,shape=[64,class_num])
#     fc4=tf.nn.dropout(fc4, 0.5, name="fc4_drop")
 
#     prob=tf.nn.softmax(fc4)
    return fc2
def losses(logits,label):
    with tf.variable_scope("loss") as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                       labels=label, name="xentropy_per_example")
        loss = tf.reduce_mean(cross_entropy, name="loss")
        tf.add_to_collection('losses',loss)
        loss = tf.add_n(tf.get_collection('losses'))
        tf.summary.scalar(scope.name + "loss", loss)
    return loss
def training(loss,learning_rate):
    with tf.variable_scope('optimizer') as scope:
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step=tf.Variable(0,name='global_step',trainable=True)
        train_op=optimizer.minimize(loss, global_step=global_step)    
    return train_op
def evaluation(logits, labels):
    with tf.variable_scope("accuracy") as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + "accuracy", accuracy)
    return accuracy
