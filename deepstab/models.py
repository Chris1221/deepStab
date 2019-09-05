# Ideally this submodule would be a seperate package with different 
# dependencies so I don't have to boot up keras and tf every time 
# that I want to do anything at all. 

import deepstab
from h5py import File 
import keras
import numpy as np
import os
import math
import sys

from time import time
from keras.preprocessing import sequence
import tensorflow as tf
#from tensorflow.test import is_gpu_available
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers import BatchNormalization, GlobalMaxPooling2D, GlobalMaxPooling1D
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
        elif mode == "loovc":
            #cv = deepstab.models.loocv(dset)
            self.gpu()

            # For now, this just loads the first chromosome as
            # the test data and the rest as the training.
            #
            # In the future, I want to do the full CV to 
            # validate performance, and this implementation 
            # will work for that too. 
            try:
                #train_seqs, train_labels, test_seqs, test_labels = next(cv)
                self.run(dset)
            except StopIteration:
                print("Done! (and maybe something went wrong...?)")

    def get_data(self, dset, mode, p = 0.8):
        if mode == "simple":
            with File(dset, 'r') as f:
                seq = f['seq'].value
                labels = f['labels'].value

            lim = int(p*len(seq))
            assert(lim == int(p*len(labels)))

            return (seq[:lim],
                    labels[:lim],
                    seq[lim:],
                    labels[lim:])

    def gpu(self):
        os.environ["CUDA_VISIBLE_DEVICES"]="1"
        #assert is_gpu_available(), "GPU is not available."  
        #print("GPU: Is now on....")


    def run(self, dset, split = 0.8):
        nout = 1

        n, nvalid = self.get_n(dset, 1, split = split)
        #pdb.set_trace()

        # Which chromosomes do we want to use for
        # training, testing, etc.
        # Might want to wrap this into an iterator later.
        train = range(2,15)
        valid = range(15, 23)
        test = [1]


        # Make a dictionary of the IDs to pass to the generator
        partition = {'train': [], 'valid': [], 'test': []}

        with File(dset) as f: 
            for t in train:
                partition['train'].extend(['chr'+str(t)+'/'+s for s in f['chr'+str(t)].keys()])
            for v in valid:
                partition['valid'].extend(['chr'+str(v)+'/'+s for s in f['chr'+str(v)].keys()])
            for t in test:
                partition['test'].extend(['chr'+str(t)+'/'+s for s in f['chr'+str(t)].keys()])

        training_generator = deepstab.models.DataGenerator(partition['train'], dset)
        validation_generator = deepstab.models.DataGenerator(partition['valid'], dset)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        model = Sequential()
        model.add(Convolution1D(
            input_shape = (None, 11),
            nb_filter=50,
            filter_length=20,
            border_mode="valid",
            activation="relu",
            #subsample_length=1,
            kernel_initializer='random_uniform',
            dilation_rate=4,
            bias_initializer='zeros'))
        model.add(MaxPooling1D(pool_size=500))
        model.add(Dropout(0.3))
        #model.add(Bidirectional(LSTM(kernel_initializer='random_uniform', return_sequences=True,bias_initializer='zeros', units = 50, input_shape = (None, 11))))
        #model.add(Dropout(0.4))
        #model.add(Flatten())
        model.add(GlobalMaxPooling1D())
        model.add(Dense(1, kernel_initializer='normal', activation='linear')) 

        model.compile(loss='mean_squared_error', optimizer='Adam')

        earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

        model.fit_generator(
                generator= training_generator,
                validation_data = validation_generator,
                epochs=50, 
                callbacks = [earlystopper])

        self.model = model
        #self.eval = model.evaluate(test_seqs,test_labels)
        #self.r2 = sklearn.metrics.r2_score(test_labels,model.predict(test_seqs))

    def batch_iter(self, dset, chr, split = 0.8):
        f = File(dset, 'r')
        grp = f[f"chr_{chr}/train/"]
        n = math.floor(split*sum(["seq" in s for s in grp.keys()]))

        for i in range(n):
            yield (grp[f"chunk_{i}_seq"][()], grp[f"chunk_{i}_label"][()])

    def valid_iter(self, dset, chr, split = 0.8):
        f = File(dset, 'r')
        grp = f[f"chr_{chr}/train/"]
        n = math.floor(split*sum(["seq" in s for s in grp.keys()]))
        total = sum(["seq" in s for s in grp.keys()])

        for i in range(n, total):
            yield (grp[f"chunk_{i}_seq"][()], grp[f"chunk_{i}_label"][()])

    def get_n(self, dset, chr, split = 0.8):
        f = File(dset, 'r')
        grp = f[f"chr_{chr}/train/"]
        n = math.floor(split*sum(["seq" in s for s in grp.keys()]))
        total =  sum(["seq" in s for s in grp.keys()])
        f.close() 
        return(n, total)


class loocv:
    """This will work for actual LOOCV, but might just use it once for now."""
    def __init__(self, dset, chrs = range(1,22)):
        self.dset = dset
        self.chrs = list(chrs)
        self.current = self.chrs[0]

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current > max(self.chrs):
            raise StopIteration
        else:
            train_seqs = []
            train_labels = []

            with File(self.dset, 'r') as f:
                test_seqs = f['chr'+str(self.current)+'_seq'].value
                test_labels = f['chr'+str(self.current)+'_labels'].value
 
                for chr in self.chrs:
                    if chr != self.current:
                        if isinstance(train_seqs, list):
                            train_seqs = f['chr'+str(self.current)+'_seq'].value
                            train_labels = f['chr'+str(self.current)+'_labels'].value
                        else:
                            train_seqs = np.concatenate( (train_seqs, f['chr'+str(self.current)+'_seq'].value) )
                            train_labels = np.concatenate( (train_labels, f['chr'+str(self.current)+'_labels'].value) )
            self.current += 1

            return (train_seqs, train_labels, test_seqs, test_labels)




class DataGenerator(keras.utils.Sequence):
    'Taken from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly'
    def __init__(self, list_IDs, dset, batch_size = 32, dim = (None, 11, 32), n_channels = 1, shuffle = True):
        self.dim = dim
        self.batch_size = batch_size 
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.dset = dset

    def __len__(self):
        return int(np.floor( len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index): 
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        #X = np.empty((self.batch_size, *self.dim, self.n_channels))
        #y = np.empty((self.batch_size), dtype=int)

        with File(self.dset) as f:
            current = [list(f[id]) for id in list_IDs_temp]
            y = [f[id].attrs['label'] for id in list_IDs_temp]


            X = sequence.pad_sequences(
                    current,
                    maxlen=max([len(c) for c in current]),
                    dtype = 'str',
                    value = 'P')

            #y = [c.attrs['label'] for c in current]^i

        return (X, y)
                






