'''
Created on 2018-12-1

@author: 南城
'''
import tensorflow as tf
loss=0
# tf.reset_default_graph()
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list()) 
def get_variableFull(input,name,shape):
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE) as scop:
         
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
def inference(images,batch_size,class_num,train=True):
    parameters = []
# conv1
#    tf.reset_default_graph()
    with tf.name_scope('conv1') as scope:
        
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 96], dtype=tf.float32,
                                             stddev=0.1), name='weights')
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(1e-4)(kernel))
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1, shape=[96], dtype=tf.float32),
                         trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]
    pool1 = tf.nn.max_pool(conv1,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool1')
    print_activations(pool1)
    pool1=tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75)
    

# conv2
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 96, 256], dtype=tf.float32,
                                             stddev=0.1), name='weights')
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(1e-4)(kernel))
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv2)

  # pool2
    pool2 = tf.nn.max_pool(conv2,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool2')
    print_activations(pool2)
    pool2=tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75)
  # conv3
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 384],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(1e-4)(kernel))
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1, shape=[384], dtype=tf.float32),
                         trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv3)

  # conv4
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 384],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(1e-4)(kernel))
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1, shape=[384], dtype=tf.float32),
                         trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
    print_activations(conv4)

  # conv5
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                             dtype=tf.float32,
                                             stddev=0.1), name='weights')
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(1e-4)(kernel))
        conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv5)

  # pool5
    pool5 = tf.nn.max_pool(conv5,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool5')
    print_activations(pool5)
    pooling5=tf.reshape(pool5,[batch_size,-1])
    print(pooling5.get_shape())
    dim=pooling5.get_shape()[1].value
    fc1=get_variableFull(pooling5, 'fullc1',shape=[dim,4096])
    fc1=tf.nn.relu(fc1)
    if train==True:
        fc1=tf.nn.dropout(fc1, 0.5 , name="fc1_drop")
    fc2=get_variableFull(fc1, 'fullc2',shape=[4096,4096])
    fc2=tf.nn.relu(fc2)
    if train==True:
        fc2=tf.nn.dropout(fc2,0.5, name="fc2_drop")
    fc3=get_variableFull(fc2, 'fullc3',shape=[4096,1000])
    fc3=tf.nn.relu(fc3)
    if train==True:
        fc3=tf.nn.dropout(fc3,0.5 , name="fc3_drop")
    fc4=get_variableFull(fc3, 'fullc4',shape=[1000,class_num])
#     fc4=tf.nn.dropout(fc4, 0.5, name="fc4_drop")
  
    
    return fc4
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

