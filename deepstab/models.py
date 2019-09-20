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
import json
import pandas as pd
import warnings

from keras.layers import Input, Dense, Conv2D, Dropout, MaxPool2D, Flatten, Conv1D, MaxPool1D, Lambda, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2
from keras.initializers import constant
from keras.models import model_from_json
from keras import backend as K
from keras.constraints import max_norm
from time import time
from keras.preprocessing import sequence
import tensorflow as tf
from keras.layers import CuDNNLSTM
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
import sklearn.preprocessing
from keras import backend as K
from keras import optimizers 
from keras.utils import multi_gpu_model
from keras import regularizers

class danQ:
    """A bidirectional LSTM on top of a simple embedding"""
    def __init__(self, dset, partition, mode = "simple"):
        if mode == "simple":
            train_seqs, train_labels, test_seqs, test_labels = self.get_data(dset, mode = "simple")
            self.run(train_seqs, train_labels, test_seqs, test_labels)
        elif mode == "loovc":
            #cv = deepstab.models.loocv(dset)
            self.gpu()
            K.clear_session()

            # For now, this just loads the first chromosome as
            # the test data and the rest as the training.
            #
            # In the future, I want to do the full CV to 
            # validate performance, and this implementation 
            # will work for that too. 
            try:
                #train_seqs, train_labels, test_seqs, test_labels = next(cv)
                self.run(dset, partition)
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


    def run(self, dset, partition, split = 0.8):
        nout = 1

        #n, nvalid = self.get_n(dset, 1, split = split)
        #pdb.set_trace()

        # Which chromosomes do we want to use for
        # training, testing, etc.
        # Might want to wrap this into an iterator later.
        training_generator = deepstab.models.DataGenerator(partition.dict['train'], dset, partition.df)
        validation_generator = deepstab.models.DataGenerator(partition.dict['valid'], dset, partition.df) 

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        def coeff_determination(y_true, y_pred):
            SS_res =  K.sum(K.square( y_true-y_pred ))
            SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
            return ( 1 - SS_res/(SS_tot + K.epsilon()) )

        model = Sequential()
        model.add(Convolution1D(
            input_shape = (None, 4),
            nb_filter=50,
            filter_length=10,
            border_mode="same", 
            kernel_initializer='zeros',
            dilation_rate=1))
            #kernel_regularizer = regularizers.l2(1e-6)))
        #model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Bidirectional(CuDNNLSTM(return_sequences=True,bias_initializer='zeros', units = 5, input_shape = (None, 4))))
        #model.add(Dropout(0.4))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(1, activation='linear')) 

        adm = optimizers.Adam(lr=0.001, epsilon=None,amsgrad=False)

        model.compile(loss='mean_squared_error', metrics = [coeff_determination], optimizer=adm)

        earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

        model.fit_generator(
                generator= training_generator,
                validation_data = validation_generator,
                epochs=50, 
                callbacks = [earlystopper])

        self.model = model

        test_generator = deepstab.models.DataGenerator(partition.dict['test'], dset, partition.df, shuffle = False)
        self.r2 = self.model.evaluate_generator(test_generator)
       
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

class DeepSea:
    def __init__(self, dset, partition):
        training_generator = deepstab.models.DataGenerator(partition.dict['train'], dset, partition.df)
        validation_generator = deepstab.models.DataGenerator(partition.dict['valid'], dset, partition.df)

        os.environ["CUDA_VISIBLE_DEVICES"]="1"
        K.clear_session()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
 
        stride_length = 1
        input_height = 4
        #input_length=len(train_seqs[0])
        pool_width=10
        hidden_units=100
        layer_1_filter_size = 8
        layer_1_filters = 20
        layer_2_filter_size = 8
        layer_2_filters=40
        layer_3_filter_size = 8
        layer_3_filters=80
        #l2_sparsity = 5e-7 ### 1e-6 is also used by the authors
        #l1_sparsity = 1e-8
        l2_sparsity = 0
        l1_sparsity = 0
        norm_constraint=0.9

        deepsea=Sequential()
        deepsea.add(Conv1D(input_shape=(None, input_height), #but one channel in the one hot encoding of the genome
            filters=layer_1_filters,
            kernel_size=layer_1_filter_size,
            strides=stride_length,
            padding="same",
            kernel_regularizer=l2(l2_sparsity),
            kernel_constraint=max_norm(norm_constraint),
            activation="relu",
            kernel_initializer='random_uniform', bias_initializer='zeros'))
        deepsea.add(MaxPool1D(pool_size=pool_width))
        deepsea.add(Dropout(0.2))
        deepsea.add(Conv1D(filters=layer_2_filters,
            kernel_size=layer_2_filter_size,
            strides=stride_length,
            padding="same",
            kernel_regularizer=l2(l2_sparsity),
            kernel_constraint=max_norm(norm_constraint),
            activation="relu",
            kernel_initializer='random_uniform', bias_initializer='zeros'))
        deepsea.add(MaxPool1D(pool_size=pool_width))
        deepsea.add(Dropout(0.2))
        deepsea.add(Conv1D(filters=layer_3_filters,
            kernel_size=layer_3_filter_size,
            strides=stride_length,
            padding="same",
            kernel_regularizer=l2(l2_sparsity),
            kernel_constraint=max_norm(norm_constraint),
            activation="relu",
            kernel_initializer='random_uniform', bias_initializer='zeros'))
        deepsea.add(Dropout(0.2))
        deepsea.add(Conv1D(filters=layer_3_filters,
            kernel_size=layer_3_filter_size,
            strides=stride_length,
            padding="same",
            kernel_regularizer=l2(l2_sparsity),
            kernel_constraint=max_norm(norm_constraint),
            activation="relu",
            kernel_initializer='random_uniform', bias_initializer='zeros'))
        #deepsea.add(Dropout(0.5))
        #deepsea.add(Conv1D(filters=layer_3_filters,
        #    kernel_size=layer_3_filter_size,
        #    strides=stride_length,
        #    padding="same",
        #    kernel_regularizer=l2(l2_sparsity),
        #    kernel_constraint=max_norm(norm_constraint),
        #    activation="relu",
        #    kernel_initializer='random_uniform', bias_initializer='zeros'))
        #deepsea.add(Dropout(0.2))
        #deepsea.add(Conv1D(filters=layer_3_filters,
        #    kernel_size=layer_3_filter_size,
        #    strides=stride_length,
        #    padding="same",
        #    kernel_regularizer=l2(l2_sparsity),
        #    kernel_constraint=max_norm(norm_constraint),
        #    activation="relu",
        #    kernel_initializer='random_uniform', bias_initializer='zeros'))
        #deepsea.add(Dropout(0.5))

        #deepsea.add(Flatten())
        #deepsea.add(GlobalAveragePooling1D())
        #deepsea.add(Dense(hidden_units,  kernel_regularizer=l2(l2_sparsity),
            #kernel_constraint=max_norm(norm_constraint),
            #activity_regularizer=l1(l1_sparsity),
            #activation="relu",
            #kernel_initializer='random_uniform', bias_initializer='zeros'))
        #deepsea.add(Bidirectional(CuDNNLSTM(return_sequences=True,bias_initializer='zeros', units = 5)))
        deepsea.add(GlobalMaxPooling1D())
        #deepsea.add(Conv1D(filters=10,
        #    kernel_size=10,
        #    dilation_rate=1))
        #deepsea.add(Conv1D(filters=10,
        #    kernel_size=10,
        #    dilation_rate=2))
        #deepsea.add(Conv1D(filters=10,
        #    kernel_size=10,
        #    dilation_rate=3))
        #deepsea.add(Dropout(0.2))
        #deepsea.add(GlobalAveragePooling1D())
        #deepsea.add(Flatten())
        deepsea.add(Dense(1, activation="linear"))

        from keras import optimizers
        eps=1e-4
        opt = optimizers.Adam(epsilon=eps)
        
        def coeff_determination(y_true, y_pred):
            SS_res =  K.sum(K.square( y_true-y_pred ))
            SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
            return ( 1 - SS_res/(SS_tot + K.epsilon()) )

        try: 
            model = multi_gpu_model(model)
        except:
            pass
        
        deepsea.compile(loss='mean_squared_error', metrics=[coeff_determination], optimizer=opt)

        earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

        

        deepsea.fit_generator(
                generator= training_generator,
                validation_data = validation_generator,
                epochs=50, 
                callbacks = [earlystopper])
        
        self.model = deepsea


        test_generator = deepstab.models.DataGenerator(partition.dict['test'], dset, partition.df, shuffle = False)
        self.r2 = self.model.evaluate_generator(test_generator)
   
        

class partition:
    def __init__(self, dset, head = False, json = None):
        warnings.warn("This has been superceeded by qc.partition.", DeprecationWarning)
        self.dset = dset

        if json is not None:
            self.read(json)
        else:
            train = range(2,17)
            valid = range(17, 23)
            test = [1]


            # Make a dictionary of the IDs to pass to the generator
            partition = {'train': [], 'valid': [], 'test': []}

            with File(dset) as f: 
                for t in train:
                    partition['train'].extend(['chr'+str(t)+'/'+s for s in f['chr'+str(t)].keys() if np.log(f['chr'+str(t)+'/'+s].attrs['label']) < 10])
                for v in valid:
                    partition['valid'].extend(['chr'+str(v)+'/'+s for s in f['chr'+str(v)].keys() if np.log(f['chr'+str(v)+'/'+s].attrs['label']) < 10])
                for t in test:
                    partition['test'].extend(['chr'+str(t)+'/'+s for s in f['chr'+str(t)].keys() if np.log(f['chr'+str(t)+'/'+s].attrs['label']) < 10 ])

            if head:
                partition['train'] = partition['train'][:500]
                partition['valid'] = partition['valid'][:500]

            self.dict = partition
            self.normalise()

    def normalise(self):
        with File(self.dset) as f:
            keys = [s for s in self.dict['train']]
            df = pd.DataFrame({'Keys': keys})
            df['Labels'] = [f[s].attrs['label'] for s in df['Keys']]
            df['Labels'] = np.log(df['Labels'])

            valid_df = pd.DataFrame({
                'Keys': [s for s in self.dict['valid']], 
                'Labels': np.log([f[s].attrs['label'] for s in self.dict['valid']])
                })
            test_df = pd.DataFrame({
                'Keys': [s for s in self.dict['test']],
                'Labels': np.log([f[s].attrs['label'] for s in self.dict['test']])
                })

            # Fit the normaliser on the training data.
            normalizer = sklearn.preprocessing.StandardScaler().fit(df['Labels'].values.reshape(-1, 1))
            df['Labels'] = normalizer.transform(df['Labels'].values.reshape(-1, 1))
            valid_df['Labels'] = normalizer.transform(valid_df['Labels'].values.reshape(-1, 1))
            test_df['Labels'] = normalizer.transform(test_df['Labels'].values.reshape(-1, 1))

            self.norm = normalizer
            
            #if 'train_df' in f:
            #    del f['train_df']
            #    df.to_hdf(dset, 'train_df')
            #else:
            #    df.to_hdf(dset, 'train_df') 
           
            #if 'valid_df' in f:
               # del f['valid_df']
               # valid_df.to_hdf(dset, 'valid_df')
            #else:
            #    valid_df.to_hdf(dset, 'valid_df') 

            #if 'test_df' in f:
            #    del f['test_df']
            #    test_df.to_hdf(dset, 'test_df')
            #else:
            #    test_df.to_hdf(dset, 'test_df') 

            self.train_df = df
            self.valid_df = valid_df
            self.test_df = test_df
    



    def write(self, out):
        with open(out, 'w') as f:
            json.dump(self.dict, f)

    def read(self, input):
        with open(input, 'r') as f:
            self.dict = json.load(f)


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
    def __init__(self, list_IDs, dset, df, batch_size = 32, dim = (None, 4), n_channels = 1, shuffle = True):
        self.dim = dim
        self.batch_size = batch_size 
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        self.dset = dset
        self.df = df

    def __len__(self):
        return int(np.floor( len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index): 
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        #print(X.shape)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        #X = np.empty((self.batch_size, *self.dim, self.n_channels))
        #y = np.empty((self.batch_size), dtype=int)

        
        with File(self.dset) as f:
            X = [list(f[id][()]) for id in list_IDs_temp]
            lens = [len(x) for x in X]
            longest = min(max(lens), 250000)
            p = [0.25]*4 + [0]*3
            for i in range(self.batch_size):
                longest = max(longest, len(X[i]))
                if len(X[i]) < longest:
                    X[i] = X[i] + [np.array(p) for i in range(longest - len(X[i]))]

            X = np.array(X)
            X = X[:,:,:4]
            #import pdb; pdb.set_trace()
        #X = np.expand_dims(X, axis =2)

            y = np.array( [self.df['Labels'][self.df['Keys'] == s].values for s in list_IDs_temp] )
            #y = np.array( [ np.log(f[id].attrs['label']) for id in list_IDs_temp] )

        out = np.empty( (self.batch_size, longest, 4, self.n_channels))
        for i in range(self.batch_size):
            out[i,] = np.expand_dims(X[i], 3)



            #X = sequence.pad_sequences(
            #        current,
            #        maxlen=max([len(c) for c in current]),
            #        dtype = 'str',
            #        value = 'P')
            #X = [f[id] for id in list_IDs_temp]

            #y = [c.attrs['label'] for c in current]^i

        return (X, y)
                

def save_both(m, p1, name):
    m.model.save('models/'+name+'.h5')
    p1.write('models/'+name+'_dset.h5')



