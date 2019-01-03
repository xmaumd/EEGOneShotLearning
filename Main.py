from matchingnetwork import convnet
import numpy as np

#Form pairwised data
train_set_0 = np.load('train_set_0.npy')
train_set_0 = np.reshape(train_set_0,[167,768,23])
train_label_0 = np.load('train_label_0.npy')
list_len = 100
list0 = np.random.randint(np.shape(train_set_0)[0], size=(list_len, 1))
list1 = np.random.randint(np.shape(train_set_0)[0], size=(list_len, 1))
#Form the lists for 1000 samples
# list0 = np.random.randint(0, 150, size=(765,))
# list0 = np.concatenate((list0,np.random.randint(151, 167, size=(85,))))
# list0 = np.concatenate((list0,np.random.randint(151, 167, size=(75,))))
# list0 = np.concatenate((list0,np.random.randint(0, 150, size=(75,))))
#
# list1 = np.random.randint(0, 150, size=(765,))
# list1 = np.concatenate((list1,np.random.randint(151, 167, size=(85,))))
# list1 = np.concatenate((list1,np.random.randint(0, 150, size=(75,))))
# list1 = np.concatenate((list1,np.random.randint(151, 167, size=(75,))))


pair_label = train_label_0[list0] + train_label_0[list1]
pair_label = [0 if x == 2 or x == 0 else 1 for x in pair_label]
pair_label_train = np.zeros([len(pair_label),2])
for i in range(len(pair_label)):
    if pair_label[i] == 0:
        pair_label_train[i,:] = [1,0]
    else:
        pair_label_train[i,:] = [0,1]

#Extract feature and store in matrix
a = feature_extract(train_set_0[0,:,:])
num_features = np.shape(a)[1]
feature_list = []
# for j in range(num_features):
#     feature_list.append([])

for i in range(np.shape(train_set_0)[0]):
    a = feature_extract(train_set_0[i,:,:])
    a = np.asarray(a)
    a = np.reshape(a,[23,np.shape(a)[1]*np.shape(a)[2],1])
    feature_list.append(a)
    # for j in range(num_features):
    #     feature_list[j].append(np.ravel(a[:,j,:]))

feature_list = np.asarray(feature_list)
list_shape = np.shape(feature_list)
feature_list1 = np.zeros([list_shape[1],list_shape[2],list_shape[0]])
for i in range(list_shape[1]):
    feature_list1[i,:,:] = feature_list[:,i,:].T

#Normalization
def norm_array(arr):
    #Each sample as row vector
    for i in range(np.shape(arr)[1]):
        mu = np.mean(arr[:,i])
        sigma = np.std(arr[:,i])
        arr[:,i] = (arr[:,i] - mu) / sigma
    return(arr)

for i in range(len(feature_list)):
    feature_list[i,:,:] = norm_array(feature_list[i,:,:])

# feature_list_0 = [feature_list1[i,:,:] for i in np.ravel(list0)]
# feature_list_1 = [feature_list1[i,:,:] for i in np.ravel(list1)]

# feature_list_0 = [feature_list[i,:,:,:] for i in np.ravel(list0)]
# feature_list_1 = [feature_list[i,:,:,:] for i in np.ravel(list1)]

feature_list_0 = [train_set_0[i,:,:,:] for i in np.ravel(list0)]
feature_list_1 = [train_set_0[i,:,:,:] for i in np.ravel(list1)]

feature_pair_list = [feature_list_0, feature_list_1]

model = siamese_net
model.summary()

opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.99, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
#model.compile(loss='mean_squared_error', optimizer=opt, metrics=["accuracy"])
model.fit(feature_pair_list, pair_label_train, batch_size=1, epochs=150)


#Testing strategy
loss, acc = model.evaluate(train_set1[164:166], train_label1[164:166,:])
pred = model.predict(feature_pair_list)


num = []
for i in range(len(train_label1)):
    num.append(np.argmax(pred[i]))
print(num)
print(pred)