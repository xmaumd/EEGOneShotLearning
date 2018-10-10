###Feature extractor
import pandas as pd
import numpy as np
from pywt import wavedec

#Traverse the files and get labels for each

summary = pd.read_csv('C:/Users/Shawn Ma/Documents/2018Summer/CHBMIT dataset/02/csv/chb02_summary.csv')
summary = summary.as_matrix()

num = 16
file_name = 'chb02_'+str(num)
sig = pd.read_csv('C:/Users/Shawn Ma/Documents/2018Summer/CHBMIT dataset/02/csv/%s_data.csv' % file_name)
sig = sig.as_matrix()
sig = sig[:,1:]


for entry in scandir('C:/Users/Shawn Ma/Documents/2018Summer/CHBMIT dataset/02/csv'):
    print(entry)

##To be refined
#Slice into epochs
def slice(sig):
    epoch_len_in_secs = 3
    epoch_len = epoch_len_in_secs * 256
    num_epoch = int(np.floor(np.shape(sig)[0] / epoch_len))
    epoch_list = []
    for i in range(num_epoch):
        epoch_list.append(sig[i*epoch_len:(i+1)*epoch_len,:])
    return(epoch_list)

#Load the summary and label epochs
def label([summary, file_name, num_epoch]):
    loc = np.where(summary == file_name+'.edf')[0]
    sz_start = summary[loc,2]; sz_end = summary[loc,3]
    label = np.zeros([num_epoch,1])
    label[int(np.floor(float(sz_start)/3)):int(np.ceil(float(sz_end)/3))] = 1
    return(label)

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