#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np

def evaluate(prelabel, truelabel):
    numlabel = truelabel.shape[1]
    
    label_dic = dict()
    prelabel = np.argmax(prelabel, axis=1)
    truelabel = np.argmax(truelabel, axis=1)
    for eachlabel in range(numlabel):
        pre = [i for i, x in enumerate(prelabel) if x == eachlabel]
        true = [i for i, x in enumerate(truelabel) if x == eachlabel]

        sum_ = 0
        for j in pre:
            if j in true:
                sum_+=1
        print('for label {}, true_label_number:{},pre_label_number:{}'.format(eachlabel,sum_,len(true)))

        label_dic[eachlabel] = (sum_ / len(true))

    print(label_dic)
    return label_dic
    

