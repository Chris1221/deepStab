import logging as log
import pandas as pd
from pathlib import Path

def chunk(_gtf, _db, _chrom):
    file = pd.read_csv(_gtf, header=None, comment="#", sep = '\t') 

    gtf = '.' + str(Path(_gtf).with_suffix('.chr'+str(_chrom)+'.gtf').parts[-1])
    db = str(Path(_db).with_suffix('.chr'+str(_chrom)+'.db'))

    file = file[file[0]==_chrom]
    file.to_csv(gtf, header = False, index = False,sep='\t')
     
    return gtf, db 
