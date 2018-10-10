###Feature extractor
import pandas as pd
import numpy as np
from pywt import wavedec

file_name = 'chb02_16'
sig = pd.read_csv('C:/Users/Shawn Ma/Documents/2018Summer/CHBMIT dataset/02/csv/%s_data.csv' % file_name)
sig = sig.as_matrix()
sig = sig[:,1:]

#Slice into epochs
epoch_len_in_secs = 3
epoch_len = epoch_len_in_secs * 256
num_epoch = int(np.floor(np.shape(sig)[0] / epoch_len))
epoch_list = []
for i in range(num_epoch):
    epoch_list.append(sig[i*epoch_len:(i+1)*epoch_len,:])

#Load the summary and label epochs
summary = pd.read_csv('C:/Users/Shawn Ma/Documents/2018Summer/CHBMIT dataset/02/csv/chb02_summary.csv')
summary = summary.as_matrix()
loc = np.where(summary == file_name+'.edf')[0]
sz_start = summary[loc,2]; sz_end = summary[loc,3]
label = np.zeros([num_epoch,1])
label[int(np.floor(float(sz_start)/3)):int(np.ceil(float(sz_end)/3))] = 1

freq = 256

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
        SE = []
        #CV = []
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

        # #Shannon Entropy
            SE.append(shannon_entropy(coeff[j]))

        #Coefficient of Variation


        feature_mtrx.append([E,FI,SE])

        #Normalization?


    ##Output
    return(feature_mtrx)