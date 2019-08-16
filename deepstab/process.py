import numpy as np
import matplotlib.pyplot as plt
import pysam 
import click
import gffutils
import os
import logging as log
import pyfaidx
import tqdm
import deepstab
import pdb

def process_bam(db_file, gtf, fasta, bam=None, bamlist=None, out='gamma_and_utr.txt.gz', chrom=99, intron_lengths=False):

    log.getLogger().setLevel(log.INFO) 

    # Deal with Chromosome request
    if chrom != 99:
        gtf, db_file = chunk.chunk(gtf, db_file, int(chrom)) 

    if os.path.isfile(db_file):
        log.info("GTF database exists, so not rewriting it.")
        db = gffutils.FeatureDB(db_file)
    else:
        log.info("GTF database does not exist, so writing it.")
        db = gffutils.create_db(gtf, db_file, merge_strategy="create_unique", disable_infer_transcripts=True, disable_infer_genes=True)
         
    if bam is not None and bamlist is None:
        bam_file = pysam.AlignmentFile(bam, "rb")
        read = deepstab.reads(bam_file)

        fa = pyfaidx.Fasta(fasta)

        log.info("Starting the main counting loop")
        # This is just the number from the grep.
        length = len(list(db.features_of_type('gene')))
        for gene in tqdm.tqdm(db.features_of_type('gene', order_by='start'), total = length):
            for transcript in db.children(gene, featuretype="transcript", order_by='start'):
                #pdb.set_trace()
                read.count_reads(db, gene, transcript, return_intron_length = intron_lengths)
                
        log.info("Adding in the UTRs and calculating gamma")
        read.add_utr_and_gamma(db, fa, return_intron_length = intron_lengths)

        input_data = deepstab.tf_input(read)
        input_data.save_by_chr('data.h5')

        log.info("Writing to file.") 
        read.write(out)
    elif bam is None and bamlist is not None:
        readlist = [] 
        fa = pyfaidx.Fasta(fasta) 

        with open(bamlist) as b:
            for bam in b:
                bam_file = pysam.AlignmentFile(bam.rstrip('\n'), "rb")
                read = reads.reads(bam_file)
                log.info("Starting the main counting loop for "+bam)
                # This is just the number from the grep.
                length = 58826
                for gene in tqdm.tqdm(db.features_of_type('gene', order_by='start'), total = length):
                    for transcript in db.children(gene, featuretype="transcript", order_by='start'):
                        pdb.set_trace()
                        read.count_reads(db, gene, transcript, return_intron_length = intron_lengths) 
                log.info("Adding in the UTRs and calculating gamma")
                read.add_utr_and_gamma(db, fa, return_intron_length = intron_lengths)
                readlist.append(read)

        tissues = reads.meta_reads(readlist)
        tissues.write(out)
    else:
        raise("Improper combination of bam and bamfile arguements.")
