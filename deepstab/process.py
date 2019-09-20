from pandas import read_csv
from pysam import AlignmentFile
from gffutils import create_db, FeatureDB
import os.path
import logging as log
from pyfaidx import Fasta
import tqdm
import deepstab

def process_bam(db_file, gtf, fasta, dset, bam=None, bamlist=None, out='gamma_and_utr.txt.gz', chrom=99, intron_lengths=False):

    log_file = dset + ".log"
    logger = log.getLogger("deepstab")
    log.basicConfig(level = log.INFO, filename=log_file) 
    logger.info(locals()) 

    # Deal with Chromosome request
    if chrom != 99:
        gtf, db_file = chunk.chunk(gtf, db_file, int(chrom)) 

    if os.path.isfile(db_file):
        logger.info("GTF database exists, so not rewriting it.")
        db = FeatureDB(db_file)
    else:
        logger.info("GTF database does not exist, so writing it.")
        db = create_db(gtf, db_file, merge_strategy="create_unique", disable_infer_transcripts=True, disable_infer_genes=True)
         
    if bam is not None and bamlist is None:
        #pdb.set_trace()
        if not os.path.exists(out):
            bam_file = AlignmentFile(bam, "rb")
            read = deepstab.reads(bam_file)

            fa = Fasta(fasta)
            #log.info("Starting the main counting loop")
            # This is just the number from the grep.
            length = len(list(db.features_of_type('gene')))
            for gene in tqdm.tqdm(db.features_of_type('gene', order_by='start'), total = length, desc='Counting reads'):
                for transcript in db.children(gene, featuretype="transcript", order_by='start'):
                    #pdb.set_trace()
                    read.count_reads(db, gene, transcript, return_intron_length = intron_lengths)
                    
            logger.info("Adding in the UTRs and calculating gamma")
            read.add_utr_and_gamma(db, dset, fa, return_intron_length = intron_lengths)
            read.write(out)
            #gdf = read.gdf
        else:
           gdf = read_csv(out, sep = "\t") 

        #input_data = deepstab.tf_input(gdf, dset)
        #input_data.save_by_chr(dset)

    elif bam is None and bamlist is not None:
        readlist = [] 
        fa = Fasta(fasta) 

        with open(bamlist) as b:
            for bam in b:
                bam_file = pysam.AlignmentFile(bam.rstrip('\n'), "rb")
                read = reads.reads(bam_file)
                logger.info("Starting the main counting loop for "+bam)
                # This is just the number from the grep.
                length = 58826
                for gene in tqdm.tqdm(db.features_of_type('gene', order_by='start'), total = length):
                    for transcript in db.children(gene, featuretype="transcript", order_by='start'):
                        read.count_reads(db, gene, transcript, return_intron_length = intron_lengths) 
                logger.info("Adding in the UTRs and calculating gamma")
                read.add_utr_and_gamma(db, fa, return_intron_length = intron_lengths)
                readlist.append(read)

        tissues = reads.meta_reads(readlist)
        tissues.write(out)
    else:
        raise("Improper combination of bam and bamfile arguements.")
