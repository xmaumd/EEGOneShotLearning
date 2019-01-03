###Feature extractor
import pandas as pd
import numpy as np
import scipy.io as sio
from pywt import wavedec
from os import scandir

#Scan the files and get labels for each

summary = pd.read_csv('C:/Users/Shawn Ma/Documents/2018Fall/Thesis/CHBMIT/6/chb06_summary.csv') # 0, 3, 8, 9, 11, 16, 17
summary = summary.as_matrix()

epoch_array_list = []
label_array_list = []
for entry in scandir('C:/Users/Shawn Ma/Documents/2018Fall/Thesis/CHBMIT/6/csv'):
    sig = pd.read_csv(entry.path)
    sig = sig.as_matrix()
    sig = sig[:,1:]
    [epoch_array, label_array] = slice_n_label(sig, summary, entry.name)
    epoch_array_list.append(epoch_array)
    label_array_list.append(label_array)

sz_file_nums = [0, 3, 8, 9, 11, 16, 17]
for i in sz_file_nums:
    #Pick epoch_array's with seizure onsets, shape up the training set
    sz_ind  = np.where(label_array_list[i] != 0)[0]; sz_ind = list(sz_ind)
    #Count the num
    sample_num = 9 * len(sz_ind)
    #Sample according to ratio
    _zero = np.where(label_array_list[i] == 0)[0]
    samp_step = int(np.ceil(len(_zero) / sample_num))
    nsz_ind = _zero[range(0,len(_zero),samp_step)]; nsz_ind = list(nsz_ind)
    #Compose a training set
    ind = nsz_ind + sz_ind
    #train_set = [epoch_array_list[i][j] for j in ind]
    #train_set1 = np.reshape(train_set, list(np.shape(train_set))+[1])
    train_label = label_array_list[i][ind]
    train_label1 = np.zeros([len(train_label),2])
    for j in range(len(train_label)):
        if train_label[j] == 0:
            train_label1[j,:] = [0,1]
        else:
            train_label1[j,:] = [1,0]
    #np.save('train_set_%d' % i, train_set1)
    np.save('train_label_%d' % i, train_label)
    #np.save('train_label_soft_%d' % i, train_label1)

def slice_n_label(sig, summary, file_name):
    #Slice into epochs
    epoch_len_in_secs = 3
    epoch_len = epoch_len_in_secs * 256
    num_epoch = int(np.floor(np.shape(sig)[0] / epoch_len))
    epoch_list = []
    for i in range(num_epoch):
        epoch_list.append(sig[i*epoch_len:(i+1)*epoch_len,:])
    #Label the epochs
    loc = np.where(summary == file_name)[0]
    #Further for multiple seizure epochs
    label_list = np.zeros([num_epoch, 1])
    if summary[loc,1] != 0:
        for i in range(int(summary[loc,1])):
            label_list[int(np.floor(float(summary[loc,2+i*2])/epoch_len_in_secs)):int(np.ceil(float(summary[loc,3+i*2])/epoch_len_in_secs))] = 1
    return([epoch_list,label_list])


freq = 256

import nolds

def feature_extract(ep):

    feature_mtrx = []
    scale = 6
    up_to = 3
    for i in range(np.shape(ep)[1]):
        ##DWT
        coeff = wavedec(ep[:,i],'db4',level = scale)
        #Select decomposition bands up to scale #
        ##Features on each scale
        E = []
        FI = []
        AE = []
        CoV = []
        #LE = [] #Lyapunov Exponent
        CD = []
        DFA = []
        for j in range(up_to+1):
            length = len(coeff[j])
        #Relative Energy
            if j == 0:
                tau = (2 ** (scale - 1)) / freq
            else:
                tau = j * (2**(scale-1)) / freq
            E.append(sum([x**2 for x in coeff[j]]) * tau / length)
        #Fluctuation Index
            FI.append( sum( np.abs(coeff[j][1:] - coeff[j][0:len(coeff[j])-1]) ) / length )

        #Detrended Fluctuation Analysis
            DFA.append(nolds.dfa(coeff[j]))
        #Approximate Entropy
            #AE.append(shannon_entropy(coeff[j]))
            AE.append(ApEn(coeff[j],2,3))

        #Coefficient of Variation
            u = np.mean(coeff[j])
            v = np.std(coeff[j])
            CoV.append(v**2/u**2)

        #Lyapunov Exponent
            #LE.append(nolds.lyap_r(coeff[j],emb_dim=5))
        #Correlation Dimension
            #CD.append(nolds.corr_dim(coeff[j],2))

        feature_mtrx.append([E,FI,AE,CoV,DFA])

        #Normalization?

    ##Output
    return(feature_mtrx)