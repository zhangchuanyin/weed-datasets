'''
Created on 2018-11-9

@author: 南城
'''
import tensorflow as tf
import numpy as np
from tensorflow.python.training.moving_averages import assign_moving_average
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.layers.python.layers import batch_norm
flags = tf.app.flags
FLAGS = flags.FLAGS


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
#     print(mask,preds.get_shape()[1])
#     pred=np.zeros([len(mask),preds.get_shape()[1]])
#     pred= tf.convert_to_tensor(pred)
#     pred[:,:]=preds[mask,:]
 
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
#     loss=-labels * tf.log(tf.clip_by_value(preds, 1e-10, 1.0))
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
     
    return tf.reduce_mean(loss) 


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
   
#     pred=np.zeros(preds.shape)
#     pred= tf.convert_to_tensor(pred)
#     pred[mask,:]=preds[mask,:]

    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
#      
    return tf.reduce_mean(accuracy_all)
    
class GCN(object):
    """docstring for GCN"""
    def __init__(self, learning_rate, num_input, num_classes, hidden_dimensions=[64],  act=tf.nn.relu):

        self.adj_hat = tf.placeholder(tf.float32, name='adjacency_matrix')
        self.input_x = tf.placeholder(tf.float32, shape=[None, num_input], name="input_x")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.dropout_keep_probrl = tf.placeholder(tf.float32, name="dropout_keep_probrl")
        self.oh_labels = tf.placeholder(tf.float32, shape=[None, num_classes], name="oh_labels")
        self.labels_mask = tf.placeholder(tf.int32, name="labels_mask")
        self.loss = 0.0
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.bias=tf.get_variable(name='bias',shape=[num_classes],initializer=tf.contrib.layers.xavier_initializer())
        with tf.variable_scope("graph_convo"):
            # scope for graph convolutional layer
#             x =tf.nn.dropout(self.input_x,self.dropout_keep_prob)
            x=self.input_x
           
            layer_input_dimensionality = num_input
    
            for indx,h_dimensionality in enumerate(hidden_dimensions):
                x=tf.nn.dropout(x,self.dropout_keep_prob)
              
                W = tf.get_variable("WGC_"+str(indx), shape=[layer_input_dimensionality,h_dimensionality],initializer=tf.contrib.layers.xavier_initializer())
               
                # compute the first matrix multiplication between x and the weight
                pre_h = tf.matmul(x,W)

                # compute the second matrix multiplication with A hat
                h = tf.matmul(self.adj_hat, pre_h)
                batch_norm_layer(h)

                # add non-linearity
                x = act(h)                    
               
                # update parameter
                layer_input_dimensionality = h_dimensionality
#                 x=tf.nn.dropout(x,self.dropout_keep_probrl) 
                self.loss +=tf.contrib.layers.l2_regularizer(1e-4)(W)
#                 compute the l2 loss for l2 regularization later
                self.loss += tf.nn.l2_loss(W)
            self.h_activation = x 


        with tf.variable_scope("output"):
            # scope for output
            W = tf.get_variable("WFC", shape=[hidden_dimensions[-1],num_classes],initializer=tf.contrib.layers.xavier_initializer())
            h = tf.matmul(self.h_activation, W)
            self.output = tf.matmul(self.adj_hat,h)
#             self.output=tf.nn.relu(self.output)
            self.output=tf.nn.dropout(self.output,self.dropout_keep_probrl)
            self.output_prob = tf.nn.softmax(self.output)

            # compute the l2 loss for l2 regularization later
            self.loss = FLAGS.weight_decay * self.loss

        with tf.variable_scope("loss"):
            # Sum over the l2_loss and the cross entropy loss
            self.loss += masked_softmax_cross_entropy(self.output, self.oh_labels,self.labels_mask)
            self.opt_op = self.optimizer.minimize(self.loss)


        with tf.variable_scope("accuracy"):
            self.accuracy = masked_accuracy(self.output, self.oh_labels,self.labels_mask)
   
def batch_norm_layer(value,train = None, name = 'batch_norm'):
    if train is not None:       
        return batch_norm(value, decay = 0.9,updates_collections=None, is_training = True)
    else:
        return batch_norm(value, decay = 0.9,updates_collections=None, is_training = False)
