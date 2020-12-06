
import tensorflow as tf
import numpy as np
import time
# import layers
import csv
import sys
from GCN import *



fold = 20
data_size = 4200   #same_size
node_size = 4200

node_embedding = 40
node_encoding = 128

meta_size = 3

test_size = int(data_size/fold)
train_size = data_size - test_size

#GCN
gcn_para = [512,256]

#train
batch_size = 128
epoch_num = 50
learning_rate = 1e-6
momentum = 0.9


def read_data():
    label = np.load('one-hot-label.npy')
    index = [i for i in range(data_size)]
    np.random.shuffle(index)
    label = label[index]
    return index[:train_size], label[:train_size], index[train_size:], label[train_size:]

# lastlabel = np.load('one-hot-label.npy')
# index = [i for i in range(data_size)]
# lastlabel = lastlabel[index]

train_data, train_label, test_data, test_label = read_data()

def get_data(ix, int_batch):
    if ix + int_batch >= train_size:
        ix = train_size - int_batch
        end = train_size
    else:
        end = ix + int_batch
    return train_data[ix:end], train_label[ix:end]

# In[8]:

class GCNPair(object):
    def __init__(self, session,
                 meta,
                 nodes,
                 class_size,
                 gcn_output1,
                 gcn_output2, 
                 embedding,
                 encoding):
        self.meta = meta
        self.nodes = nodes
        self.class_size = class_size
        self.gcn_output1 = gcn_output1
        self.gcn_output2 = gcn_output2
        self.embedding = embedding
        self.encoding_size = encoding
        
        self.build_placeholders()

        self.loss, self.probabilities, self.features ,self.logits= self.forward_propagation()
        self.l2 = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.01), tf.trainable_variables())


        self.pred = tf.one_hot(tf.argmax(self.probabilities,1),class_size)

        correct_prediction = tf.equal(tf.argmax(self.probabilities,1), tf.argmax(self.t, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


        print('Forward propagation finished.')
        
        self.sess = session
        #self.optimizer = tf.train.MomentumOptimizer(self.lr, self.mom) #.minimize(self.loss+self.l2)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        gradients = self.optimizer.compute_gradients(self.loss+self.l2)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        self.train_op = self.optimizer.apply_gradients(capped_gradients)
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
        
    def forward_propagation(self):
        with tf.variable_scope('weights_n'):
            A = tf.reshape(self.a, [self.meta, self.nodes*self.nodes])
            A_ = tf.transpose(A, [1, 0]) #dimention exchange
            W = tf.nn.sigmoid(tf.get_variable('W', shape=[self.meta, 1], initializer=tf.contrib.layers.xavier_initializer()))
            weighted_adj = tf.matmul(A_, W)# add all meta based matrix
            weighted_adj = tf.reshape(weighted_adj, [1, self.nodes, self.nodes])


        with tf.variable_scope('spectral_gcn'):
            gcn_out = GCN(tf.expand_dims(self.x, 0), weighted_adj, [self.gcn_output1, self.gcn_output2, self.encoding_size]).build()
        
        with tf.variable_scope('classification'):
            batch_data = tf.matmul(tf.one_hot(self.batch_index, self.nodes), gcn_out[0])
            #W = tf.get_variable(name='weights', shape=[self.encoding_size, self.class_size], initializer=tf.contrib.layers.xavier_initializer())
            W = tf.get_variable(name='weights', shape=[self.encoding_size, self.class_size], initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
            b = tf.get_variable(name='bias', shape=[1, self.class_size], initializer=tf.zeros_initializer())
            logits = tf.matmul(batch_data, W) + b

            #loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.t, logits=logits)
            #loss = tf.losses.softmax_cross_entropy(onehot_labels=self.t, logits=logits)
            loss = tf.losses.softmax_cross_entropy(onehot_labels=self.t, logits=logits)


        return loss, tf.nn.softmax(logits), gcn_out[0],logits
    
    def train(self, x, a, t, b, learning_rate = 1e-2, momentum = 0.9):
        feed_dict = {
            self.x: x,
            self.a: a,
            self.t: t, 
            self.batch_index: b,
            self.lr: learning_rate,
            self.mom: momentum
        }
        _, loss, acc, pred, prob,logits , features= self.sess.run([self.train_op, self.loss, self.accuracy, self.pred, self.probabilities,self.logits, self.features], feed_dict = feed_dict)
        
        return loss, acc, pred, prob,logits,features

    def test(self, x, a, t, b):
        feed_dict = {
            self.x : x,
            self.a : a,
            self.t : t,
            self.batch_index : b
        }
        acc, pred, features = self.sess.run([self.accuracy, self.pred, self.features], feed_dict = feed_dict)
        return acc, pred, features

    def save(self, path, step):
        self.saver.save(self.sess, path, global_step=step)
        
def com_f1(pred,label):
    MI_F1 = []
    l = len(pred)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    f1 = 0
    for i in range(l):
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
    return [pre,rec,acc,f1]
    #return acc

if __name__ == "__main__":

    #with tf.Session() as sess:
     #   net = GCNPair(session=sess, class_size=7, gcn_output1=gcn_para[0],
      #          gcn_output2=gcn_para[1], meta=13,nodes=node_size, embedding=node_embedding, encoding=node_encoding)

  
    xdata = np.load("typefeature.npy")
  
    # max_ = xdata_.max(0)
    # min_ = xdata_.min(0)
    # xdata = (xdata_ - min_) / (max_ - min_)



   
    adj_data = np.load("small_adj_data.npy")
    with tf.Session() as sess:
        net = GCNPair(session=sess, class_size=4, gcn_output1=gcn_para[0],
                gcn_output2=gcn_para[1], meta=meta_size,nodes=node_size, embedding=node_embedding, encoding=node_encoding)
        sess.run(tf.global_variables_initializer())

        min_loss = 15061162
        max_acc = -1
        loss_upper_bound = 100
        compair = 0
        for epoch in range(epoch_num):
            train_loss = 0
            train_acc = 0
            count = 0
            
            for index in range(0, train_size, batch_size):
                batch_data, batch_label = get_data(index, batch_size)
                loss, acc, pred, prob,logits,features = net.train(xdata, adj_data, batch_label, batch_data, learning_rate, momentum)
#                if loss > loss_upper_bound:
#                    print('Loss Explode!!!!!! Early Stop.')
#                    sys.exit(0)
                if index % 1 == 0:
                    print("batch loss: {:.4f}, batch acc: {:.4f}".format(loss, acc))
                train_loss += loss
                train_acc += acc
                count += 1

            train_loss = train_loss/count
            train_acc = train_acc/count
            if train_loss < min_loss:
                min_loss = train_loss
            print("epoch{:d} : train_loss: {:.4f}, train_acc: {:.4f}".format(epoch, train_loss, train_acc))

            if train_acc > compair:
                compair = train_acc
                net.saver.save(sess, 'premodel/')
                print('********************* Model Saved *********************')

            net.saver.restore(sess, 'premodel/')
            eva_acc, eva_pred, features = net.test(xdata, adj_data, test_label, test_data)


            # with open('train_acc.txt', 'a+') as f:
            #     f.write(str(train_acc))
            #     f.write('\n')
            # with open('test_acc.txt', 'a+') as f:
            #     f.write(str(eva_acc))
            #     f.write('\n')
            # with open('train_loss.txt', 'a+') as f:
            #     f.write(str(train_loss))
            #     f.write('\n')

            #if eva_acc > max_acc:
                #max_acc = eva_acc
                # print('present max accuracy:', eva_acc)
                # print('golden label:', test_label)
                # print('pred label:', eva_pred)
#                    net.saver.save(sess,"model/model")
                #print('********************* Model Saved *********************')
                #features = np.array(features)
                #np.save('features.npy', features)

        print("Train end!")
        print("The loss is {:.4f}, the acc is {:.4f}".format(min_loss, max_acc))

        # variable_names = [v.name for v in tf.trainable_variables()]
        # values = sess.run(variable_names)
        # for k, v in zip(variable_names, values):
        #     print("Variable: ", k)
        #     print("Shape: ", v.shape)
        #     print(v)

        




        ##-----------node classificaiton test--------

        #from sklearn.model_selection import train_test_split
        from sklearn.datasets import load_digits
        #from sklearn.linear_model import LogisticRegression

        #LR = LogisticRegression(C=1.0, penalty='l1', tol=0.01)
        #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
        #LR.fit(X_train, Y_train)
        #pred=LR.predict(X_test)

        #print('start knn, kmean.....')
        #xx = np.expand_dims(out[0], axis=0)[test_mask]
        #xx = features[test_data]

        #from numpy import linalg as LA


        #yy = lastlabel#2857*4
        #yy=test_label
        #print('xx: {}, yy: {}'.format(xx.shape, yy.shape))
        #from GCNexp import my_KNN, my_Kmeans  # , my_TSNE, my_Linear

        #my_KNN(xx, yy)
        #my_Kmeans(xx, yy)

        #sess.close()
