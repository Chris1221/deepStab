import numpy as np
import matplotlib.pyplot as plt
import pysam 
import click
import chunk
import reads
import gffutils
import os
import logging as log
import pyfaidx
import tqdm


# Obviously I still have the defaults set for me right now...
@click.command()
@click.option("--db_file", default = "../db/test.db", help="Path to database file, if it exists.")
@click.option("--gtf", default = "../lib/test.gtf", help="Path to the GTF file.")
@click.option("--bam", default = None, help="Path to the BAM file.")
@click.option("--bamlist", default = None, help="Optionally a path to a list of bam files for multi-task learning.")
@click.option("--fasta", default = "../lib/Homo_sapiens.GRCh38.dna_rm.primary_assembly.fa", help="Path to the FASTA file.")
@click.option("--out", default = "gamma_and_utr.txt.gz")
@click.option("--intron_lengths/--no_intron_lengths", default = False)
@click.option("--chrom", default = 99, help="OPTIONAL. If you would like to subset the data and process only one chromosome, indicate it here. This will append _chr${CHROM} to your database name.")


def main(db_file, gtf, bam, bamlist,fasta, out, chrom, intron_lengths):

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
        read = reads.reads(bam_file)

        fa = pyfaidx.Fasta(fasta)

        log.info("Starting the main counting loop")
        # This is just the number from the grep.
        length = 58826
        for gene in tqdm.tqdm(db.features_of_type('gene', order_by='start'), total = length):
            for transcript in db.children(gene, featuretype="transcript", order_by='start'):
                read.count_reads(db, gene, transcript, return_intron_length = intron_lengths)
                
        log.info("Adding in the UTRs and calculating gamma")
        read.add_utr_and_gamma(db, fa, return_intron_length = intron_lengths)

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
                        read.count_reads(db, gene, transcript, return_intron_length = intron_lengths) 
                log.info("Adding in the UTRs and calculating gamma")
                read.add_utr_and_gamma(db, fa, return_intron_length = intron_lengths)
                readlist.append(read)

        tissues = reads.meta_reads(readlist)
        tissues.write(out)
    else:
        raise("Improper combination of bam and bamfile arguements.")


            

if __name__=='__main__':
    main()



