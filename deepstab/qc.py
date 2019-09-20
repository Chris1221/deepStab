import pandas as pd
import numpy as np
from gffutils import FeatureDB
import tqdm
import pdb
import sklearn.preprocessing
from h5py import File
from itertools import chain
from random import shuffle
import ast




def create_df(path, db):
    """Create the partition object for the model. Includes the 
    dictionary of training, testing, and validation sets, as well as
    the dataframes which give the label mapping. Superceeds old partition
    class."""
    def find_max_genes(df, n = 5):
        df = df[~df.duplicated(['spliced', 'unspliced'])]
        df['sum'] = df['spliced'] + df['unspliced'] 
        return df.loc[df['sum'].nlargest(n=n).index.values.tolist()]
        

    df = pd.read_csv(path, '\t')
    df = df.groupby(['gene']).apply(find_max_genes)

    df = find_lengths(df, db)

    return(df)


def find_lengths(df, db_path):
    """Find the lengths of the transcritpt's exons and introns,
    so that we can properly normalise the counts."""
    db = FeatureDB(db_path)

    df['intron_length'] = 0
    df['cds_length'] = 0
    df['total_length'] = 0

    idx = pd.DataFrame([(tx.id, tx.attributes['transcript_name'][0]) for tx in db.features_of_type('transcript')])

    db = FeatureDB(db_path)


    for tx in tqdm.tqdm(df['transcript']):
        name = idx[0][idx[1] == tx].values[0] 
        tran = db[name]
        df.loc[df['transcript'] == tx, 'cds_length'] =  db.children_bp(name)
        df.loc[df['transcript'] == tx, 'total_length'] =  tran.stop - tran.start
        df.loc[df['transcript'] == tx, 'intron_length'] =  tran.stop - tran.start - db.children_bp(name)
        df.loc[df['transcript'] == tx, 'db_name'] = name 

   
    return(df)


def process(df):
    df['adj_u'] = df['unspliced'] / df['intron_length']
    df['adj_s'] = df['spliced'] / df['cds_length']
    df['adj_g'] = df['adj_u'] / df['adj_s']
    #df['adj_g'] = df['unspliced'] / df['spliced']
    df['adj_g'] = np.log(df['adj_g'])
    df = df[df['adj_g'] < 10]
    return(df)


class partition:
    """A container class for holding the relevant information for training the net. Superceeds the models.partition class"""
    def __init__(self, df=None, split_by_chr = True, prop = 0.8, output_name = 'new_gamma', path = None):

        if path is None:
            train = range(2,17)
            valid = range(17, 23)
            test = [1]
            partition = {'train': [], 'valid': [], 'test': []}

            print("Using " + output_name + " as the output.")

            if split_by_chr:
                partition['train'].extend( ['chr'+str(c) + '/' + tx for c in train for tx in df['db_name'][df['chr'] == 'chr'+str(c)]] )
                partition['valid'].extend( ['chr'+str(c) + '/' + tx for c in valid for tx in df['db_name'][df['chr'] == 'chr'+str(c)]] )
                partition['test'].extend( ['chr'+str(c) + '/' + tx for c in test for tx in df['db_name'][df['chr'] == 'chr'+str(c)]] )      
            else:
                partition['train'].extend( ['chr'+str(c) + '/' + tx for c in chain(train, valid) for tx in df['db_name'][df['chr'] == 'chr'+str(c)]] )
                partition['test'].extend( ['chr'+str(c) + '/' + tx for c in test for tx in df['db_name'][df['chr'] == 'chr'+str(c)]] )      
                shuffle(partition['train'])
                l = int(np.floor(prop*len(partition['train']) * prop))
                partition['valid'] = partition['train'][l:]
                partition['train'] = partition['train'][:l]
                

            self.dict = partition

            self.df = pd.DataFrame([(r[1]['chr'] + '/'+r[1]['db_name'], r[1][output_name]) for c in range(1,23) for r in df[df['chr'] == 'chr'+str(c)].iterrows()], columns = ['Keys', 'Labels'])

            self.df['partition'] = 'NA'
            for p in ['train', 'test', 'valid']:
                for t in tqdm.tqdm(partition[p], desc = p):
                    self.df.loc[self.df['Keys'] == t, 'partition'] = p




            #self.train_df = pd.DataFrame([(r[1]['chr'] + '/'+r[1]['db_name'], r[1][output_name]) for c in train for r in df[df['chr'] == 'chr'+str(c)].iterrows()], columns = ['Keys', 'Labels'])
            #self.valid_df = pd.DataFrame([(r[1]['chr'] + '/'+r[1]['db_name'], r[1][output_name]) for c in valid for r in df[df['chr'] == 'chr'+str(c)].iterrows()], columns = ['Keys', 'Labels'])
            #self.test_df = pd.DataFrame([(r[1]['chr'] + '/'+r[1]['db_name'], r[1][output_name]) for c in test for r in df[df['chr'] == 'chr'+str(c)].iterrows()], columns = ['Keys', 'Labels'])
         
            normalizer = sklearn.preprocessing.StandardScaler().fit(self.df['Labels'][self.df['partition'] == 'train'].values.reshape(-1, 1))
            self.df.loc[self.df['partition'] == 'train', 'Labels'] = normalizer.transform(self.df.loc[self.df['partition'] == 'train', 'Labels'].values.reshape(-1, 1))
            self.df.loc[self.df['partition'] == 'valid', 'Labels'] = normalizer.transform(self.df.loc[self.df['partition'] == 'valid', 'Labels'].values.reshape(-1, 1))
            self.df.loc[self.df['partition'] == 'test', 'Labels'] = normalizer.transform(self.df.loc[self.df['partition'] == 'test', 'Labels'].values.reshape(-1, 1))

    #        self.valid_df['Labels'] = normalizer.transform(self.valid_df['Labels'].values.reshape(-1, 1))
    #        self.test_df['Labels'] = normalizer.transform(self.test_df['Labels'].values.reshape(-1, 1))

            self.norm = normalizer

        else: 
            self.read(path)

    def write(self, path):
        self.df.to_hdf(path, 'df')
        with File(path) as f:    
           f.create_dataset('dict', data=str(self.dict))


    def read(self, path): 
        self.df = pd.read_hdf(path, 'df')
        with File(path) as f: 
            self.dict = ast.literal_eval(f['dict'][()])

   

def create_partition(df, db):
    df = create_df(df, db)
    df = process(df)
    p = partition(df)
    return(p)
