import deepstab
import h5py
import keras
import numpy as np
import os
import sys

from time import time
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import BatchNormalization
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Bidirectional
from keras.callbacks import TensorBoard
import sklearn.metrics

class danQ:
    """A bidirectional LSTM on top of a simple embedding"""
    def __init__(self, dset, mode = "simple"):
        if mode == "simple":
            train_seqs, train_labels, test_seqs, test_labels = self.get_data(dset, mode = "simple")
            self.run(train_seqs, train_labels, test_seqs, test_labels)

    def get_data(self, dset, mode, p = 0.8):
        if mode == "simple":
            with h5py.File(dset, 'r') as f:
                seq = f['seq'].value
                labels = f['labels'].value

            lim = int(p*len(seq))
            assert(lim == int(p*len(labels)))

            return (seq[:lim],
                    labels[:lim],
                    seq[lim:],
                    labels[lim:])


    def run(self,train_seqs, train_labels, test_seqs, test_labels):
        nout = len(train_labels)

        model = Sequential()
        model.add(Convolution1D(
            input_shape = (len(train_seqs[0]), len(train_seqs[0][0])),
            nb_filter=50,
            filter_length=20,
            border_mode="valid",
            activation="relu",
            #subsample_length=1,
            kernel_initializer='random_uniform',
            bias_initializer='zeros'))
        model.add(MaxPooling1D(pool_size=25))
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(kernel_initializer='random_uniform', return_sequences=True,bias_initializer='zeros', units = 100)))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(1, kernel_initializer='normal', activation='linear')) 

        model.compile(loss='mean_squared_error', optimizer='Adam')

        earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

        model.fit(train_seqs, 
                train_labels, 
                batch_size=100, 
                epochs=50, 
                shuffle=True, 
                validation_split=0.2, 
                callbacks = [earlystopper])

        self.model = model
        self.eval = model.evaluate(test_seqs,test_labels)
        self.r2 = sklearn.metrics.r2_score(test_labels,model.predict(test_seqs))
 
