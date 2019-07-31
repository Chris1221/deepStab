import gzip
import math
import numpy as np
import os
import h5py
import keras
import pandas as pd
from tensorflow.test import is_gpu_available
from keras.models import model_from_json
from Bio import pairwise2
import tqdm
from itertools import compress
from sklearn.preprocessing import scale

def qc(file, direction):
    if direction == 5:
        idx = 6
    elif direction == 3:
        idx = 7
    else:
        raise("Improper direction. Must be 5 or 3.")

    
    # Remove all entries with missing UTRs of interest
    print('I removed ' + str(sum(file[idx].isna())) + ' for missing UTRs') 
    file = file[~file[idx].isna()]


    #print('I removed ' + str(sum(['N' not in s for s in file[idx]])) + ' for having Ns')
    #file = file[['N' not in s for s in file[idx]]]

#    print('Shuffling the input just in case theres some weird chromosome thing going on.')
#    file = file.sample(frac=1)

    # Remove all entries with less than 100 reads
    #   This is really just a spitball.
    print('Removing ' + str(sum((file[3] + file[4]) < 250)) + ' for having low read counts.')
    file = file[file[3] + file[4] > 10]

    print('I removed ' + str(sum(file[5] > 20)) + ' for being too big.')
    file = file[file[5] < 20]
  #  file = file[[5,idx]]
    # Indexes stay the same
    # This might throw a warning but thats okay

    file = collapse_seq(file, 0.8)

    file[5] = np.log(file[5])

    return file

def overlap(seq1, seq2):
    return pairwise2.align.globalxx(seq1, seq2, score_only=True)

def collapse_seq(file, p):
    file['dup'] = False
    for gene in tqdm.tqdm(file[1].unique().tolist(), total= len(file[1].unique().tolist()), desc = 'Genes:'):
        for row1 in file[file[1] == gene].iterrows():
            for row2 in file[file[1] == gene].iterrows():
                if row1[0] != row2[0]:
                    if not file.loc[row1[0], 'dup'] or file.loc[row2[0], 'dup']:
                        if overlap(row1[1][6],row2[1][6]) > p*len(row1[1][6]):
                            file.loc[row2[0], 5] =  np.mean([row1[1][5], row2[1][5]])
                            file.loc[row1[0], 'dup'] = True
    file_out = file[~file['dup']].drop('dup', axis=1)
    return file_out

def collapse_seq_multi(file, p, utr='utr5'):
    file['dup'] = False
    for gene in tqdm.tqdm(file['gene'].unique().tolist(), total= len(file['gene'].unique().tolist()), desc = 'Genes:'):
        for row1 in file[file['gene'] == gene].iterrows():
            for row2 in file[file['gene'] == gene].iterrows():
                if row1[0] != row2[0]:
                    if not file.loc[row1[0], 'dup'] or file.loc[row2[0], 'dup']:
                        if overlap(row1[1][utr],row2[1][utr]) > p*len(row1[1][utr]):
                            #file.loc[row2[0], '] =  np.mean([row1[1][5], row2[1][5]])
                            file.loc[row1[0], 'dup'] = True
    file_out = file[~file['dup']].drop('dup', axis=1)
    return file_out


def split(_file_qc, _a):

    test = _file_qc[_file_qc[0].isin(['chr18', 'chr19', 'chr20', 'chr21'])]
    train = _file_qc[_file_qc[0].isin(['chr'+str(i) for i in range(15)])]
    validation = _file_qc[_file_qc[0].isin(['chr'+str(i) for i in range(15,18)])]

    train = train.sample(frac=1)

    test = test[[5,6]]
    train = train[[5,6]]
    validation = validation[[5,6]]

    #train = train[:int( _a * train.shape[0])]
    #validation = train[int( _a * train.shape[0]):]
    
    return train, validation, test

# Helper Function  get hotcoded sequence
# From Ron's DL tutorial
def get_hot_coded_seq(sequence):
    """Convert a 4 base letter sequence to 4-row x-cols hot coded sequence"""
    hotsequence = np.zeros((len(sequence),4))
    for i in range(len(sequence)):
        if sequence[i] == 'A':
            hotsequence[i,0] = 1
        elif sequence[i] == 'C':
            hotsequence[i,1] = 1
        elif sequence[i] == 'G':
            hotsequence[i,2] = 1
        elif sequence[i] == 'T':
            hotsequence[i,3] = 1
        elif sequence[i] == 'N':
            hotsequence[i,0] = 0.25
            hotsequence[i,1] = 0.25
            hotsequence[i,2] = 0.25
            hotsequence[i,3] = 0.25
        elif sequence[i] == 'P':
            pass
    return hotsequence

# Helper function to read in the labels and seqs and store as hot encoded np array
def read_and_pad(train, test, validation,length=1000):
    output = []

    length = 0

    for infile in [train, test, validation]:
        with open(infile, "r") as f:
            seqs = []
            labels = []
            for i,l in enumerate(f):
                l = l.rstrip()
                l = l.split("\t")
                seqs.append(l[1])
                labels.append(l[0])
        # make labels np.array
        labels = np.array(labels)
        # convert to one_hot_labels
        #hot_labels = keras.utils.to_categorical(labels, num_classes=4)
        hot_labels = labels
        # make seqs np.array

        # Have to first find the maximum length 
        #maxs=0
        #for s in seqs:
        #    maxs=max(maxs,len(s))

        # Recheck max lenght for each one
        # So they are all padded to the same length
        #length = max( length, maxs ) 

        seqs_list = [list(s) for s in seqs]
        padded = keras.preprocessing.sequence.pad_sequences(seqs_list, maxlen=length, dtype = 'str', value = 'P')
        padded_list = [''.join(p) for p in padded]

        hot_seqs = np.zeros( (len(seqs), length, 4) )
        # fill with hot encoded sequences
        for j in range(len(padded_list)):
            hotsequence = get_hot_coded_seq(padded_list[j])
            hot_seqs[j,] = hotsequence

        output.append([hot_labels, hot_seqs])
   
    train_labels = output[0][0]
    train_seqs = output[0][1]

    test_labels = output[1][0]
    test_seqs = output[1][1]

    valid_labels = output[2][0]
    valid_seqs = output[2][1]

    return train_labels.astype(float), train_seqs, test_labels.astype(float), test_seqs, valid_labels.astype(float), valid_seqs

def read_and_pad_multi(gammas, p=0.8, length=2000, shuf = False, split = 0.8, utr='utr5', both = False):
    if not both: 
        file = pd.read_csv(gammas, sep = "\t", na_values='nan') 
        gamma_key=file.filter(regex='gamma').columns.tolist() 
        file=file[~file[utr].isna()]
        file = file[[r.count('N') < 0.5*len(r) for r in file[utr].values.tolist()]]
        #file = collapse_seq_multi(file,p, utr = utr)
        file = file.sample(frac=1)

        for key in gamma_key:
            file[key] = np.log(file[key])

        file['G']=list(scale(file.filter(regex='gamma').values))

        seqs_list = [list(s) for s in file[utr].values.tolist()]
        padded = keras.preprocessing.sequence.pad_sequences(seqs_list, maxlen=length, dtype = 'str', value = 'P')
        padded_list = [''.join(p) for p in padded]

        hot_seqs = np.zeros( (len(seqs_list), length, 4) )

        for j in range(len(padded_list)):
            hotsequence = get_hot_coded_seq(padded_list[j])
            hot_seqs[j,] = hotsequence    
        
        #file['hot']= [utils.get_hot_coded_seq(seq) for seq in file['utr5'].tolist()] 
        #output = []
        gammas = file['G'].values

        glist = np.zeros( ( len(gammas), len(gammas[0])) )

        for g in range(len(glist)):
            glist[g,] = gammas[g]
        

        #train_seqs = list(compress(hot_seqs, file['chr'].isin(['chr'+str(i) for i in range(15)])))A
        if not shuf: 
            train_seqs = hot_seqs[ file['chr'].isin(['chr'+str(i) for i in range(15)])]
            valid_seqs = hot_seqs[file['chr'].isin(['chr'+str(i) for i in range(15,18)])]
            test_seqs = hot_seqs[file['chr'].isin(['chr'+str(i) for i in range(18,22)])]

            train_labels = glist[ file['chr'].isin(['chr'+str(i) for i in range(15)])]
            valid_labels = glist[file['chr'].isin(['chr'+str(i) for i in range(15,18)])]
            test_labels = glist[file['chr'].isin(['chr'+str(i) for i in range(18,22)])] 
        elif shuf: 
            train_seqs = hot_seqs[: int(split * len(hot_seqs))]
            test_seqs = hot_seqs[ int(split * len(hot_seqs)) : ]
            valid_seqs = train_seqs[int(split*len(train_seqs)) :]
            train_seqs = train_seqs [: int(split* len(train_seqs)) ] 

            train_labels = glist[: int(split * len(glist))]
            test_labels = glist[ int(split * len(glist)) : ]
            valid_labels = train_labels[int(split*len(train_labels)) :]
            train_labels = train_labels [: int(split* len(train_labels)) ] 

        return train_labels, train_seqs, test_labels, test_seqs, valid_labels, valid_seqs

    elif both: 
        file = pd.read_csv(gammas, sep = "\t", na_values='nan') 
        gamma_key=file.filter(regex='gamma').columns.tolist() 
        file=file[~file['utr5'].isna()]
        file=file[~file['utr3'].isna()]

        file = file[[r.count('N') < 0.5*len(r) for r in file['utr5'].values.tolist()]]
        file = file[[r.count('N') < 0.5*len(r) for r in file['utr3'].values.tolist()]]

        #file = collapse_seq_multi(file,p, utr = utr)
        file = file.sample(frac=1)

        for key in gamma_key:
            file[key] = np.log(file[key])

        file['G']=list(scale(file.filter(regex='gamma').values))

        # Do for 3' UTRs
        length = 5000

        seqs_3_list = [list(s) for s in file['utr3'].values.tolist()]
        padded_3 = keras.preprocessing.sequence.pad_sequences(seqs_3_list, maxlen=length, dtype = 'str', value = 'P')
        padded_3_list = [''.join(p) for p in padded_3]

        hot_seqs_3 = np.zeros( (len(seqs_3_list), length, 4) )

        for j in range(len(padded_3_list)):
            hotsequence_3 = get_hot_coded_seq(padded_3_list[j])
            hot_seqs_3[j,] = hotsequence_3   

        # Do for 5' UTRs
        length=500
        seqs_5_list = [list(s) for s in file['utr5'].values.tolist()]
        padded_5 = keras.preprocessing.sequence.pad_sequences(seqs_5_list, maxlen=length, dtype = 'str', value = 'P')
        padded_5_list = [''.join(p) for p in padded_5]

        hot_seqs_5 = np.zeros( (len(seqs_5_list), length, 4) )

        for j in range(len(padded_5_list)):
            hotsequence_5 = get_hot_coded_seq(padded_5_list[j])
            hot_seqs_5[j,] = hotsequence_5    

        
        #file['hot']= [utils.get_hot_coded_seq(seq) for seq in file['utr5'].tolist()] 
        #output = []
        gammas = file['G'].values

        glist = np.zeros( ( len(gammas), len(gammas[0])) )

        for g in range(len(glist)):
            glist[g,] = gammas[g]
        

        #train_seqs = list(compress(hot_seqs, file['chr'].isin(['chr'+str(i) for i in range(15)])))A
        if not shuf: 
            train_seqs_3 = hot_seqs_3[ file['chr'].isin(['chr'+str(i) for i in range(2,22)])]
            train_seqs_5 = hot_seqs_5[ file['chr'].isin(['chr'+str(i) for i in range(2,22)])]

            #valid_seqs_3 = hot_seqs_3[file['chr'].isin(['chr'+str(i) for i in range(15,18)])]
            #valid_seqs_5 = hot_seqs_5[file['chr'].isin(['chr'+str(i) for i in range(15,18)])]

            test_seqs_3 = hot_seqs_3[file['chr'].isin(['chr1'])]
            test_seqs_5 = hot_seqs_5[file['chr'].isin(['chr1'])]

            train_labels = glist[ file['chr'].isin(['chr'+str(i) for i in range(2, 22)])]
            #valid_labels = glist[file['chr'].isin(['chr'+str(i) for i in range(15,18)])]
            test_labels = glist[file['chr'].isin(['chr1'])]

            return train_seqs_3, train_seqs_5, train_labels, test_seqs_3, test_seqs_5, test_labels
        elif shuf: 
            train_seqs_3 = hot_seqs_3[: int(split * len(hot_seqs_3))]
            test_seqs_3 = hot_seqs_3[ int(split * len(hot_seqs_3)) : ]
            #valid_seqs_3 = train_seqs_3[int(split*len(train_seqs_3)) :]
            #train_seqs_3 = train_seqs_3[: int(split* len(train_seqs_3)) ] 

            train_seqs_5 = hot_seqs_5[: int(split * len(hot_seqs_5))]
            test_seqs_5 = hot_seqs_5[ int(split * len(hot_seqs_5)) : ]
            #valid_seqs_5 = train_seqs_5[int(split*len(train_seqs_5)) :]
            #train_seqs_5 = train_seqs_5[: int(split* len(train_seqs_5)) ] 


            train_labels = glist[: int(split * len(glist))]
            test_labels = glist[ int(split * len(glist)) : ]
            #valid_labels = train_labels[int(split*len(train_labels)) :]
            #train_labels = train_labels [: int(split* len(train_labels)) ] 

            return train_seqs_3, train_seqs_5, train_labels, test_seqs_3, test_seqs_5, test_labels

def save_model_for_deeplift(_model, _prefix):
    model_json = _model.to_json()
    with open(_prefix + ".json", "w") as json_file:
        json_file.write(model_json)
    _model.save_weights(_prefix + ".h5")

def gpu():
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    assert is_gpu_available(), "GPU is not available." 

def join_dfs(ldf, rdf):
    return pd.merge(ldf, rdf, on=['chr', 'gene', 'transcript', 'utr5', 'utr3'])

def recover_data(path):
    train = pd.read_hdf(path, 'train')
    test = pd.read_hdf(path, 'test')
    valid = pd.read_hdf(path, 'valid')

    return np.array(train[train.columns[0]].tolist()), np.array(train['hot'].tolist()), np.array(test[test.columns[0]].tolist()), np.array(test['hot'].tolist()), np.array(valid[valid.columns[0]].tolist()),  np.array(valid['hot'].tolist()) 

def recover_data_multi(path, both = False):
    if not both:
        f = h5py.File(path, 'r')

        train_seqs = f['train_seqs'].value
        test_seqs = f['test_seqs'].value
        valid_seqs = f['valid_seqs'].value

        train_labels = f['train_labels'].value
        test_labels = f['test_labels'].value
        valid_labels = f['valid_labels'].value

        return train_labels, train_seqs, test_labels, test_seqs, valid_labels, valid_seqs
    elif both:
        f = h5py.File(path, 'r')

        train_seqs_3 = f['train_seqs_3'].value
        test_seqs_3 = f['test_seqs_3'].value
        #valid_seqs_3 = f['valid_seqs_3'].value
        train_seqs_5 = f['train_seqs_5'].value
        test_seqs_5 = f['test_seqs_5'].value
        #valid_seqs_5 = f['valid_seqs_5'].value


        train_labels = f['train_labels'].value
        test_labels = f['test_labels'].value
        #valid_labels = f['valid_labels'].value

        return train_seqs_3, train_seqs_5, train_labels, test_seqs_3, test_seqs_5, test_labels #, valid_seqs_3, valid_seqs_5, valid_labels
 
def save_predictions(model, seqs, labels, path):
    pred = model.predict(seqs)
    out = pd.DataFrame({'Pred': pred.squeeze(), 'Truth': labels})
    out.to_csv(path, index=None)
