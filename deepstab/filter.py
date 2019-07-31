import click
import gzip
import pandas as pd
import numpy as np
import math
import utils
import h5py
from sklearn import preprocessing
import tables 
from itertools import compress

@click.command()
@click.option("--gammas", default = "gamma_and_utrs.txt.gz", help = "Path to the output from main.py")
@click.option("--prefix", default = "../data/input/testing", help = "Prefix to the produced files.")
@click.option("--direction", default=5, help="Are we interested in 5' UTRs or 3' UTRs?")
@click.option("--multi/--single", default=False, help="Does this file contain multiple gammas?")
@click.option("--split", default=0.8, help="Train test split (only for multi).")
@click.option("--shuf/--no-shuf", default = False)
@click.option("--overlap", default = 0.9)
@click.option("--utr5/--utr3", default = True)
@click.option("--both/--one", default = False)

def filter(gammas, multi, prefix, direction, split, shuf, overlap, utr5, both):
    if not multi:
        file = pd.read_csv(gammas, sep = "\t", header=None)
        file_qc = utils.qc(file, direction)
        train, validation, test = utils.split(file_qc, 0.8)

        train.to_csv(prefix+'-train.txt', header=False, index = False, sep = "\t")
        validation.to_csv(prefix+'-valid.txt', header=False, index=False, sep = "\t")
        test.to_csv(prefix+'-test.txt', header=False, index=False, sep = "\t")
    elif multi: 
        print('Shuf is ' + str(shuf))
        if utr5:
            utr = 'utr5'
        else:
            utr='utr3'

        print('UTR is ' + str(utr))

        if not both: 
            train_labels, train_seqs, test_labels, test_seqs, valid_labels, valid_seqs = utils.read_and_pad_multi(gammas, p = overlap, shuf=shuf, utr=utr, both = False)
            filename = prefix+'.h5'
            
            f = h5py.File(filename, 'w')
            f.create_dataset('train_seqs', data=train_seqs)
            f.create_dataset('valid_seqs', data=valid_seqs)
            f.create_dataset('test_seqs', data=test_seqs)
            f.create_dataset('train_labels', data=train_labels)
            f.create_dataset('valid_labels', data=valid_labels)
            f.create_dataset('test_labels', data=test_labels)
        elif both:
            print("Both is true")
            train_seqs_3, train_seqs_5, train_labels, test_seqs_3, test_seqs_5, test_labels = utils.read_and_pad_multi(gammas, p = overlap, shuf=shuf, utr=utr, both = True)
            filename = prefix+'.h5'
            
            f = h5py.File(filename, 'w')
            f.create_dataset('train_seqs_3', data=train_seqs_3)
            f.create_dataset('train_seqs_5', data=train_seqs_5)

            #f.create_dataset('valid_seqs_3', data=valid_seqs_3)
            #f.create_dataset('valid_seqs_5', data=valid_seqs_5)

            f.create_dataset('test_seqs_3', data=test_seqs_3)
            f.create_dataset('test_seqs_5', data=test_seqs_5)

            f.create_dataset('train_labels', data=train_labels)
            #f.create_dataset('valid_labels', data=valid_labels)
            f.create_dataset('test_labels', data=test_labels)







if __name__=='__main__':
    filter()
