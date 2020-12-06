
# coding: utf-8


import tensorflow as tf
import numpy as np
import time
#import layers
import csv
import sys
from GCN import *
import scipy.sparse as sp
import random
from models import GAT
from utils import process
import evalue


check_file = 'DBLP/compairGCNGAT/premodel/'

tf.set_random_seed(1)

#fold = 20
data_size = 4057   #same_size
node_size = 4057
node_embedding = 334
node_encoding = 64
meta_size = 3

#test_size = int(data_size/fold)
#train_size = data_size - test_size

#GCN
gcn_para = [256,128]

#train
batch_size = 4057
epoch_num = 1000000000
learning_rate = 0.001
momentum = 0.9
hid_units = [8]
n_heads = [8, 1]
residual = False
nonlinearity = tf.nn.elu

modeL2 = GAT


# def read_data():
#     label_ = np.load('one_hot_lable.npy')
#     index = [i for i in range(data_size)]
#     np.random.shuffle(index)
#     label = label_[index]
#     return label_,index[:train_size], label[:train_size], index[train_size:], label[train_size:]
# alllabel,train_data, train_label, test_data, test_label = read_data()

alllabel = np.load('DBLP/compairGCNGAT/one_hot_labels.npy')
train_data = np.load('DBLP/compairGCNGAT/train_idx.npy')
test_data = np.load('DBLP/compairGCNGAT/test_idx.npy')
train_label = alllabel[train_data]
test_label = alllabel[test_data]

train_size = train_data.shape[0]
test_size = test_data.shape[0]

print('alllabel_shape:',alllabel.shape)
print('train_data_shape:',train_data.shape)
print('test_data_shape:',test_data.shape)
print('train_label_shape:',train_label.shape)
print('test_label_shape:',test_label.shape)
print('train_size:',train_size)
print('test_size:',test_size)

def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

test_mask = sample_mask(test_data,alllabel.shape[0])

def get_data(ix, int_batch):
    if ix + int_batch >= train_size:
        ix = train_size - int_batch
        end = train_size
    else:
        end = ix + int_batch
    return train_data[ix:end], train_label[ix:end]


def preprocess_features_v1(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1,).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return np.matrix(features)

# In[8]:

class GCNPair(object):
    def __init__(self, session,
                 meta,
                 nodes,
                 class_size,
                 gcn_output1,
                 gcn_output2, 
                 embedding,
                 encoding,
                 residual,
                 nonlinearity,
                 hid_units,
                 n_heads):
        self.meta = meta
        self.nodes = nodes
        self.class_size = class_size
        self.gcn_output1 = gcn_output1
        self.gcn_output2 = gcn_output2
        self.embedding = embedding
        self.encoding_size = encoding
        self.residual = residual
        self.nonlinearity = nonlinearity
        self.n_heads = n_heads
        self.hid_units = hid_units
        
        self.build_placeholders()
        
        self.loss, self.probabilities, self.features = self.forward_propagation()
        self.l2 = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.005), tf.trainable_variables())


        self.pred = tf.one_hot(tf.argmax(self.probabilities,1),class_size)
        correct_prediction = tf.equal(tf.argmax(self.probabilities,1), tf.argmax(self.t,1))


        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print('Forward propagation finished.')
        
        self.sess = session
        #self.optimizer = tf.train.MomentumOptimizer(self.lr, self.mom).minimize(self.loss+self.l2)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.gradients = self.optimizer.compute_gradients(self.loss+self.l2)
        #gradients = self.optimizer.compute_gradients(self.loss)
        #capped_gradients = [(tf.clip_by_value(grad*1000, -15., 15.), var) for grad, var in self.gradients if grad is not None]
        #self.train_op = self.optimizer.apply_gradients(capped_gradients)
        self.train_op = self.optimizer.apply_gradients(self.gradients)
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(tf.global_variables())
        print('Backward propagation finished.')
        
    def build_placeholders(self):
        self.a = tf.placeholder(tf.float32, [self.meta, self.nodes, self.nodes], 'adj')
        self.x = tf.placeholder(tf.float32, [self.nodes, self.embedding], 'nxf')
        self.batch_index = tf.placeholder(tf.int32, [None], 'index')
        self.t = tf.placeholder(tf.int32, [None, self.class_size], 'labels')
        self.lr = tf.placeholder(tf.float32, [], 'learning_rate')
        self.mom = tf.placeholder(tf.float32, [], 'momentum')

        self.attn_drop = tf.placeholder(dtype=tf.float32, shape=())
        self.ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
        self.is_train = tf.placeholder(dtype=tf.bool, shape=())
        
    def forward_propagation(self):
        # input ==> (2187, 2187)
        # self.x ==> (2187, 1433)
        # self.t ==> (2187, 7)
        with tf.variable_scope('weights_n'):
            A = tf.reshape(self.a, [self.meta, self.nodes*self.nodes])
            A_ = tf.transpose(A, [1, 0]) #dimention exchange
            WW = tf.nn.sigmoid(tf.get_variable('W', shape=[self.meta, 1], initializer=tf.contrib.layers.xavier_initializer()))
            weighted_adj = tf.matmul(A_, WW)# add all meta based matrix
        
            weighted_adj = tf.reshape(weighted_adj, [1, self.nodes, self.nodes])

            
            # Wone = tf.constant([[1.0], [1.0], [1.0]], dtype=tf.float32, )
            # weighted_oneadj = tf.matmul(A_, Wone)
            # weighted_oneadj = tf.reshape(weighted_oneadj, [1, self.nodes, self.nodes])

            
            WL2 = tf.nn.sigmoid(tf.get_variable('WL2', shape=[self.meta, 1], initializer=tf.contrib.layers.xavier_initializer()))
            weighted_L2adj = tf.matmul(A_, WL2)
            weighted_L2adj = tf.reshape(weighted_L2adj, [1, self.nodes, self.nodes])

        with tf.variable_scope('spectral_gcn'):

            #L1out = GCN(tf.expand_dims(self.x, 0), weighted_adj, [self.gcn_output1, self.gcn_output2, self.encoding_size]).build()
            L1out = GCN(tf.expand_dims(self.x, 0), weighted_adj,[self.gcn_output2]).build()
            print('self.x_type:',type(self.x))
            print('self.x:',tf.expand_dims(self.x,0))
            _, L2out = modeL2.inference(L1out, self.class_size, self.nodes, self.is_train,
                                           self.attn_drop, self.ffd_drop,
                                          bias_mat=weighted_L2adj,
                                          hid_units=self.hid_units, n_heads=self.n_heads,
                                          residual=self.residual, activation=self.nonlinearity)


            #gcn_out = GCN(tf.expand_dims(self.x, 0), weighted_adj, [self.gcn_output1, self.gcn_output2, self.encoding_size]).build()

            # logits, out = modeL2.inference(gcn_out[0], self.class_size, self.nodes, self.is_train,
            #                                self.attn_drop, self.ffd_drop,
            #                               bias_mat=weighted_adj,
            #                               hid_units=self.hid_units, n_heads=self.n_heads,
            #                               residual=self.residual, activation=self.nonlinearity)
        
        with tf.variable_scope('classification'):
            batch_data = tf.matmul(tf.one_hot(self.batch_index, self.nodes), L2out[0])
            W = tf.get_variable(name='weights', shape=[self.encoding_size, self.class_size], initializer=tf.contrib.layers.xavier_initializer())
            #W = tf.get_variable(name='weights', shape=[self.encoding_size, self.class_size], initializer=tf.random_normal_initializer())
            b = tf.get_variable(name='bias', shape=[1, self.class_size], initializer=tf.zeros_initializer())
            logits = tf.matmul(batch_data, W) + b
            #loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.t, logits=logits)
            loss = tf.losses.softmax_cross_entropy(onehot_labels=self.t, logits=logits)
        
        return loss, tf.nn.softmax(logits), L2out[0]


    def train(self, x, a, t, b, learning_rate = 1e-2, momentum = 0.9,ffd_drop=0.2,attn_drop=0.2,is_train=True):
        feed_dict = {
            self.x: x,
            self.a: a,
            self.t: t, 
            self.batch_index: b,
            self.lr: learning_rate,
            self.mom: momentum,
            self.ffd_drop:ffd_drop,
            self.attn_drop:attn_drop,
            self.is_train:is_train
        }
        _, loss, acc, pred, prob, gra= self.sess.run([self.train_op, self.loss, self.accuracy, self.pred, self.probabilities, self.gradients], feed_dict = feed_dict)
        
        return loss, acc, pred, prob, gra

    def test(self, x, a, t, b,ffd_drop=0,attn_drop=0,is_train=False):
        feed_dict = {
            self.x : x,
            self.a : a,
            self.t : t,
            self.batch_index : b,
            self.ffd_drop: ffd_drop,
            self.attn_drop: attn_drop,
            self.is_train: is_train
        }
        acc, pred, features = self.sess.run([self.accuracy, self.pred, self.features], feed_dict = feed_dict)
        return acc, pred, features
        
def com_f1(pred,label):
    MI_F1 = []
    l = len(pred)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    f1 = 0
    for i in range(len(pred[0])):
        if pred[i] == 1 and label[i] == 1:
            TP += 1
        elif pred[i] == 1:
            FP += 1
        elif label[i] == 1:
            FN += 1
        else:
            TN += 1
    if TP+FP == 0:
       pre = 0
    else:
       pre = TP/(TP + FP)
    if TP+FN == 0:
       rec = 0
    else:
       rec = TP/(TP + FN)
    acc = (TP+TN)/l
    if (pre + rec) != 0:
        f1 = 2*pre*rec/(pre+rec)
    return pre,rec,acc,f1

if __name__ == "__main__":

    xdata = np.load("DBLP/compairGCNGAT/features.npy")
    #xdata = preprocess_features_v1(xdata)
    adj_data = np.load("DBLP/compairGCNGAT/small_adj_data.npy")

    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
#config=config

    with tf.Session() as sess:
        net = GCNPair(session=sess, class_size=4, gcn_output1=gcn_para[0],
                gcn_output2=gcn_para[1], meta=meta_size,nodes=node_size, embedding=node_embedding, encoding=node_encoding,
                      hid_units=hid_units, n_heads=n_heads, residual=residual, nonlinearity=nonlinearity)
        sess.run(tf.global_variables_initializer())

        min_loss = 15061162
        max_acc = -1
        maxtrain_acc = -1
        loss_upper_bound = 100
        for epoch in range(epoch_num):
            train_loss = 0
            train_acc = 0
            count = 0

            #print(sess.run(WW))

            for index in range(0, train_size, batch_size):
                batch_data, batch_label = get_data(index, batch_size)
                loss, acc, pred, prob,gra = net.train(xdata, adj_data, batch_label, batch_data, learning_rate, momentum)
                #print('gra:',gra)
                # _,_,acccc,f11 = com_f1(pred,batch_label)
                # print('acccc:',acccc,'f11:',f11)s


#                if loss > loss_upper_bound:
#                    print('Loss Explode!!!!!! Early Stop.')
#                    sys.exit(0)
#                 if index % 1 == 0:
#                     print("batch loss: {:.4f}, batch acc: {:.4f}".format(loss, acc))
                train_loss += loss
                train_acc += acc
                count += 1

            train_loss = train_loss/count
            train_acc = train_acc/count
            if train_loss < min_loss:
                min_loss = train_loss
            print("epoch{:d} : train_loss: {:.10f}, train_acc: {:.10f}".format(epoch, train_loss, train_acc))

            with open('DBLP/compairGCNGAT/premodel/output.txt', 'a+') as fi:
                fi.write(str(train_acc) + '\n')

            if train_acc > maxtrain_acc:
                maxtrain_acc = train_acc
                with open('DBLP/compairGCNGAT/premodel/output.txt', 'a+') as fi:
                    fi.write('training：' + '\t' + str(maxtrain_acc) + '\n')

            eva_acc, eva_pred, featts = net.test(xdata, adj_data, test_label, test_data)

            with open('DBLP/compairGCNGAT/premodel/output.txt', 'a+') as fi:
                fi.write('..............................'+str(eva_acc)+'\n')


            # with open('train_acc.txt', 'a+') as f:
            #     f.write(str(train_acc))
            #     f.write('\n')
            # with open('test_acc.txt', 'a+') as f:
            #     f.write(str(eva_acc))
            #     f.write('\n')
            # with open('train_loss.txt', 'a+') as f:
            #     f.write(str(train_loss))
            #     f.write('\n')

            if eva_acc > max_acc:
                max_acc = eva_acc
                print('present max accuracy:', eva_acc)
                print('golden label:', test_label)
                print('pred label:', eva_pred)
                net.saver.save(sess,check_file,global_step = epoch)
                print('********************* Model Saved *********************')
                featts = np.array(featts)
                np.save('DBLP/compairGCNGAT/premodel//featts.npy', featts)

                with open('DBLP/compairGCNGAT/premodel/output.txt', 'a+') as fi:
                    fi.write('testing：' + '\t' + str(eva_acc) + '\n')

                eachlabelacc = evalue.evaluate(eva_pred, test_label)
                with open('DBLP/compairGCNGAT/premodel/output.txt', 'a+') as fi:
                    fi.write('each label：' + '\t' + str(eachlabelacc) + '\n')



        # variable_names = [v.name for v in tf.trainable_variables()]
        # values = sess.run(variable_names)
        # for k, v in zip(variable_names, values):
        #     print("Variable: ", k)
        #     print("Shape: ", v.shape)
        #     print(v)





# 0.31428573  epoch504