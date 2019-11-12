'''
This script will train the GCN model and evaluate its performance on three datasets.
The code is adapted from https://github.com/tkipf/gcn.
'''

from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from gcn import *

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 20000, 'Number of epochs to train.')
flags.DEFINE_float('dropout', 0.3, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-3, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')

def main():
    # Load data
    adj_hat, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data()
    print(adj_hat.shape, features.shape, y_train.shape, y_val.shape, y_test.shape, train_mask.shape, val_mask.shape, test_mask.shape)
    # pre-process features
#     features = preprocess_features(features)
# 
#     # normalize the adjacency matrix
#     adj_hat = preprocess_adj(adj)

    # import pdb;pdb.set_trace()
    
    # create a GCN model object
    num_classes=y_train.shape[1]
    num_examples, num_input=features.shape
    model = GCN(FLAGS.learning_rate, num_input, num_classes, hidden_dimensions=[64], act=tf.nn.relu)

    # Initialize session
    sess = tf.Session()

    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()

        # Construct feed dictionary
        train_feed_dict = {
            model.adj_hat: adj_hat,
            model.input_x: features,
            model.oh_labels: y_train,
            model.labels_mask: train_mask,
            model.dropout_keep_prob: FLAGS.dropout,
            model.dropout_keep_probrl:0.3
#             model.num_features_nonzero: features[1].shape
        }
        # import pdb;pdb.set_trace()
        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=train_feed_dict)

        # Validation
        val_feed_dict = {
            model.adj_hat: adj_hat,
            model.input_x: features,
            model.oh_labels: y_val,
            model.labels_mask: val_mask,
            model.dropout_keep_prob: 1.0,
            model.dropout_keep_probrl:1.0
#             model.num_features_nonzero: features[1].shape
        }
        cost, acc = sess.run([model.loss, model.accuracy], feed_dict=val_feed_dict)
        cost_val.append(cost)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
# 
#         if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
#             print("Early stopping...")
#             break

    print("Optimization Finished!")

    # Testing
    test_feed_dict = {
        model.adj_hat: adj_hat,
        model.input_x: features,
        model.oh_labels: y_test,
        model.labels_mask: test_mask,
        model.dropout_keep_prob: 1.0,
        model.dropout_keep_probrl:1.0
#         model.num_features_nonzero: features[1].shape
    }
    t = time.time()
    test_cost, test_acc = sess.run([model.loss, model.accuracy], feed_dict=test_feed_dict)
    test_duration = time.time() - t
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

    sess.close()


if __name__ == '__main__':
    main()

