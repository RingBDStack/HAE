﻿import numpy as np
import tensorflow as tf

from models import GAT
from utils import process

checkpt_file = 'DBLP/compairGATGAT/premodel/'

batch_size = 1
nb_epochs = 1000000
patience = 100
lr = 0.001 # learning rate
l2_coef = 0.001  # weight decay
hid_units = [8] # numbers of hidden units per each attention head in each layer
n_heads = [8, 1] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = GAT

data_size = 4057
meta_size = 3


#print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))
#=================================================================================================================================


adj = np.load('DBLP/compairGATGAT/small_adj_data.npy')
features = np.load('DBLP/compairGATGAT/features.npy')

print('features.shape:', features.shape)
print('adj.shape:', adj.shape)

lastlabel = np.load('DBLP/compairGATGAT/one_hot_labels.npy')
train_data = np.load('DBLP/compairGATGAT/train_idx.npy')
test_data = np.load('DBLP/compairGATGAT/test_idx.npy')
train_label = lastlabel[train_data]
test_label = lastlabel[test_data]

train_size = train_data.shape[0]
test_size = test_data.shape[0]

print('alllabel_shape:',lastlabel.shape)
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

train_idx = np.zeros([1,train_size],dtype=int)
test_idx = np.zeros([1,test_size],dtype=int)
train_idx[0]=train_data
test_idx[0]=test_data

train_mask = sample_mask(train_idx, lastlabel.shape[0])
test_mask = sample_mask(test_idx, lastlabel.shape[0])

y_train = np.zeros(lastlabel.shape)
y_test = np.zeros(lastlabel.shape)

y_train[train_mask, :] = lastlabel[train_mask, :]#only train part has label delete others
y_test[test_mask, :] = lastlabel[test_mask, :]
print('y_train:{}, y_test:{}, train_mask:{}, test_mask:{}'.format(y_train.shape,
                                                                        y_test.shape,
                                                                        train_mask.shape,
                                                                        test_mask.shape))



nb_nodes = features.shape[0]
ft_size = features.shape[1]
nb_classes = y_train.shape[1]

print('nb_nodes:', nb_nodes)
print('ft_size:', ft_size)
print('nb_classes:', nb_classes)

#adj = adj.todense()

features = features[np.newaxis]
y_train = y_train[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

print('features:{},adj:{}, y_train:{} ,y_test:{}, train_mask:{},  test_mask:{}'.format(np.array(features).shape,
                                                                                             np.array(adj).shape,
                                                                                             y_train.shape,
                                                                                             y_test.shape,
                                                                                             train_mask.shape,
                                                                                             test_mask.shape))

#biases = process.adj_to_bias(adj, nb_nodes, nhood=1)
biases = adj
print('biases:',biases.shape)
print('biases:',biases)

with tf.Graph().as_default():

    ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
    bias_in = tf.placeholder(dtype=tf.float32, shape=(meta_size, nb_nodes, nb_nodes))
    lbl_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes))
    msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))

    attn_drop = tf.placeholder(dtype=tf.float32, shape=())
    ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
    is_train = tf.placeholder(dtype=tf.bool, shape=())

    A = tf.reshape(bias_in, [meta_size, nb_nodes * nb_nodes])
    A_ = tf.transpose(A, [1, 0])
    # W_ = tf.get_variable( 'W_', shape=[meta_size, 1], initializer=tf.contrib.layers.xavier_initializer())
    W = tf.nn.sigmoid(tf.get_variable('W', shape=[meta_size, 1], initializer=tf.contrib.layers.xavier_initializer()))
    # W = tf.nn.sigmoid(tf.get_variable('W', shape=[meta_size, 1], initializer=tf.random_normal_initializer()))
    # W_ = tf.Variable(tf.truncated_normal([meta_size,1]),name='W_')
    # W = tf.nn.sigmoid(W_)
    weighted_adj = tf.matmul(A_, W)
    weighted_adj = tf.reshape(weighted_adj, [1, nb_nodes, nb_nodes])
    WL2 = tf.nn.sigmoid(tf.get_variable('WL2', shape=[meta_size, 1], initializer=tf.contrib.layers.xavier_initializer()))
    weighted_L2adj = tf.matmul(A_, WL2)
    weighted_L2adj = tf.reshape(weighted_L2adj, [1, nb_nodes, nb_nodes])

    _, L1out = model.inference(ftr_in, nb_classes, nb_nodes, is_train,
                                  attn_drop, ffd_drop,
                                  bias_mat=weighted_adj,
                                  hid_units=hid_units, n_heads=n_heads,
                                  residual=residual, activation=nonlinearity)

    logits, L2out = model.inference(L1out, nb_classes, nb_nodes, is_train,
                                  attn_drop, ffd_drop,
                                  bias_mat=weighted_L2adj,
                                  hid_units=hid_units, n_heads=n_heads,
                                  residual=residual, activation=nonlinearity)


    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)

    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)
    f1 = model.micro_f1(log_resh, lab_resh, msk_resh)

    train_op = model.training(loss, lr, l2_coef)

    saver = tf.train.Saver()

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    vacc_mx = 0.0
    curr_step = 0

    with tf.Session() as sess:

        sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        train_f1_avg = 0
        # train_f1_avg = 0
        # compair = 0

        tr_size = features.shape[0]
        max_test_acc = -1
        max_train_acc = -1
        compair = 0

        for epoch in range(nb_epochs):

            tr_step = 0
            train_f1_avg = 0

            print(sess.run(W))

            while tr_step * batch_size < tr_size:
                _, loss_value_tr, acc_tr, f1_tr = sess.run([train_op, loss, accuracy, f1],feed_dict={
                                                               ftr_in: features[tr_step * batch_size:(tr_step + 1) * batch_size],
                                                               bias_in: biases[tr_step * meta_size:(tr_step + 1) * meta_size],
                                                               lbl_in: y_train[tr_step * batch_size:(tr_step + 1) * batch_size],
                                                               msk_in: train_mask[tr_step * batch_size:(tr_step + 1) * batch_size],
                                                               is_train: True,
                                                               attn_drop: 0.1, ffd_drop: 0.1})

                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                train_f1_avg += f1_tr

                tr_step += 1

            print('epoch %s,Training: loss = %.10f, acc = %.10f, f1 = %.10f' % (epoch, train_loss_avg / tr_step, train_acc_avg / tr_step, train_f1_avg / tr_step))

            with open('DBLP/compairGATGAT/premodel/output.txt', 'a+') as fi:
                fi.write(str(train_acc_avg / tr_step) + '\n')

            if (train_acc_avg / tr_step) > max_train_acc:
                max_train_acc = train_acc_avg / tr_step
                with open('DBLP/compairGATGAT/premodel/output.txt', 'a+') as fi:
                    fi.write('accuracy of training：' + '\t' + str(max_train_acc) + '\n')

            # if (train_acc_avg/tr_step) > compair:
            #     compair = train_acc_avg/tr_step
            #     #saver.save(sess, checkpt_file)
            #     print('...................Save Model................')

            train_loss_avg = 0
            train_acc_avg = 0

            # saver.restore(sess, checkpt_file)
            ts_size = features.shape[0]
            ts_step = 0
            ts_loss = 0.0
            ts_acc = 0.0
            ts_f1 = 0.0

            while ts_step * batch_size < ts_size:
                loss_value_ts, acc_ts, f1_ts, feas = sess.run([loss, accuracy, f1, L2out], feed_dict={
                                                                  ftr_in: features[ts_step * batch_size:(ts_step + 1) * batch_size],
                                                                  bias_in: biases[ts_step * meta_size:(ts_step + 1) * meta_size],
                                                                  lbl_in: y_test[ts_step * batch_size:(ts_step + 1) * batch_size],
                                                                  msk_in: test_mask[ts_step * batch_size:(ts_step + 1) * batch_size],
                                                                  is_train: False,
                                                                  attn_drop: 0.0, ffd_drop: 0.0})
                ts_loss += loss_value_ts
                ts_acc += acc_ts
                ts_f1 += f1_ts
                ts_step += 1

                print(feas.shape)  # (1,4200,64)
                with open('DBLP/compairGATGAT/premodel/output.txt','a+') as fi:
                    fi.write('............................'+str(ts_acc / ts_step)+'\n')

                if (ts_acc / ts_step) > max_test_acc:
                    max_test_acc = ts_acc / ts_step
                    print('Test loss:', ts_loss / ts_step, '; Test accuracy:', ts_acc / ts_step, 'Test f1:',ts_f1 / ts_step)
                    with open('DBLP/compairGATGAT/premodel/output.txt','a+') as fi:
                        fi.write('accuracy of test：'+'\t'+str(max_test_acc)+'\n')
                    saver.save(sess, checkpt_file,global_step = epoch)
                    print('...................Save Model................')
                    feas = np.array(feas)
                    np.save('DBLP/compairGATGAT/premodel/featts.npy', feas)

        sess.close()










