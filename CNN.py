import numpy as np
from array import array
import scipy.io as sio
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, \
    BatchNormalization
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform
from keras.optimizers import Adam

from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from keras.utils.vis_utils import model_to_dot, plot_model
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow


def EEG_CNN(input_shape):
    """
    Function creating the EEG_LSTM model's graph.

    Arguments:
    input_shape -- shape of the input, usually (number of vectors, max_len)

    Returns:
    model -- a model instance in Keras
    """

    ### START CODE HERE ###
    # Define sentence_indices as the input of the graph, it should be of shape input_shape and dtype 'int32' (as it contains indices).
    EEG_train = Input(input_shape, dtype='float32')

    X = ZeroPadding2D((1, 1))(EEG_train)
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    X = Conv2D(30, (3, 3), strides=(1, 1), name='conv0')(X)

    # Add dropout with a probability of 0.5
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    #    X = MaxPooling2D((3,3), name='max_pool0')(X)
    #    X = Conv2D(15, (3, 3), strides = (1, 1), name = 'conv1')(X)

    X = MaxPooling2D((3, 3), name='max_pool1')(X)
    X = Conv2D(5, (3, 3), strides=(1, 1), name='conv2')(X)
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    # X = LSTM(128, return_sequences=False)(X)
    # # Add dropout with a probability of 0.5
    # X = Dropout(0.5)(X)
    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.
    X = Flatten()(X)
    X = Dense(2)(X)
    # Add a softmax activation
    X = Activation('softmax')(X)
    # X = X.index(max(X))
    # Create Model instance which converts sentence_indices into X.
    model = Model(inputs=EEG_train, outputs=X)

    ### END CODE HERE ###

    return model


vecnlabel = sio.loadmat(
    'C:\\Users\\Shawn Ma\\Documents\\2017-18 Fall\\EEG project\\Data_Processing\\image_4channel.mat')
label = vecnlabel['label_feature_1hot']
train_data = vecnlabel['train_feature']
# train_data = train_data.reshape(112,100,41,4)

print(label)
print(label.shape[0], label.shape[1])
print(train_data.shape[0], train_data.shape[1], train_data.shape[2], train_data.shape[3])

model = EEG_CNN(input_shape=(100, 41, 4))
model.summary()

opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
model.fit(train_data, label, batch_size=1, epochs=30)

loss, acc = model.evaluate(train_data, label)
pred = model.predict(train_data)

num = []
for i in range(label.shape[0]):
    num.append(np.argmax(pred[i]))
print(num)
print(pred)

# plot_model(model, to_file='model.png')
# SVG(model_to_dot(model).create(prog='dot', format='svg'))