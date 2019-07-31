import gffutils
import pysam 
import numpy as np
import pandas as pd
import pdb
import os
import pyfaidx
import functools

class count:
    def __init__(self):
       self.spliced = 0 
       self.unspliced = 0
       self.ambiguous = 0 

class lengths:
    def __init__(self):
       self.intron_lengths = [] 
       self.total_intron_length = 0

class reads:
    def __init__(self, bam):
        self.bam = bam
        self.counts = {}
        self.lengths = {}

    def count_reads(self, db, gene, transcript, return_intron_length = False):
        """Counts up spliced reads, those strictly matching exons.

        I also need to include exon-exon junctions""" 
        # Change to the IDs, even though they are less interpretable.
        # Need a unique ID.
        gene_name = gene.id
        transcript_name = transcript.id
        chrom = 'chr'+gene.chrom
        #contig = self.bam.get_reference_name(gene.bin)


        if gene_name in self.counts.keys():
            pass
        else:
            self.counts[gene_name] = {}
            self.lengths[gene_name] = {}

        self.counts[gene_name][transcript_name] = count()

        # Create lists of coordinates 
        self.exon_coordinates =      []
        self.intron_coordinates =    []
        
        for exon in db.children(transcript, featuretype = "exon"): 
            self.exon_coordinates.append((exon.start, exon.end))

        self.infer_intron()

        if return_intron_length:
            self.lengths[gene_name][transcript_name] = lengths()
            
            for intron in self.intron_coordinates:
                self.lengths[gene_name][transcript_name].intron_lengths.append( intron[1] - intron[0]) 
                self.lengths[gene_name][transcript_name].total_intron_length += intron[1] - intron[0]


        
        for feature in self.exon_coordinates + self.intron_coordinates: 
            for read in self.bam.fetch(chrom, feature[0], feature[1]):     
                self.count_individual_read(read, transcript, gene, gene_name, transcript_name)

            
    def count_individual_read(self, _read, _transcript, _gene, gene_name, transcript_name):
        e_read = (_read.reference_start, _read.reference_end)
        # Case 1:   The read maps nicely into an exon, completely
        #           -   Add a 1 to the "spliced" count.
        e_start_idx =   [e_read[0] >= e_list[0] and e_read[0] <= e_list[1] for e_list in self.exon_coordinates]
        e_end_idx =     [e_read[1] >= e_list[0] and e_read[1] <= e_list[1] for e_list in self.exon_coordinates]

        i_start_idx =   [e_read[0] >= i_list[0] and e_read[0] <= i_list[1] for i_list in self.intron_coordinates]
        i_end_idx =     [e_read[1] >= i_list[0] and e_read[1] <= i_list[1] for i_list in self.intron_coordinates]

        # At most there should be only one True in any list
        assert sum(e_start_idx) <= 1 
        assert sum(e_end_idx) <= 1
        assert sum(i_start_idx) <= 1
        assert sum(i_end_idx) <= 1

        # If it starts and stops in an exon
        if any(e_start_idx) and any(e_end_idx):
            # Then either it's contained within one exon
            #   or its exon-exon junction.
            #   Either way, it's evidence for spliced.
            #       Though I'm still not totally sure why only 1 exon would be evidence of spliced.
            self.counts[gene_name][transcript_name].spliced += 1
        elif any(e_start_idx) and any(i_end_idx): 
            # This is an intron exon boundary, so 
            #   evidence of unspliced.
            self.counts[gene_name][transcript_name].unspliced += 1
        elif any(i_start_idx) and any(i_end_idx): 
            # This read starts and stops in an exon
            self.counts[gene_name][transcript_name].unspliced += 1
        else:
            # Something else happened 
            self.counts[gene_name][transcript_name].ambiguous += 1 

    def infer_intron(self):
        """To avoid using the gffutils infer_introns, infers it from the exon coordinates instead."""
        for i in range(len(self.exon_coordinates)-1):
            last = len(self.exon_coordinates)
            if self.exon_coordinates[i+1][0] - self.exon_coordinates[i][1] > 0:
                self.intron_coordinates.append( (self.exon_coordinates[i][1], self.exon_coordinates[i+1][0] ) )

    def add_utr_and_gamma(self, db, fa, return_intron_length = False):
        self.gammas = []
        
        for gene in self.counts.keys():
            for transcript in self.counts[gene].keys():
                if self.counts[gene][transcript].unspliced > 0:
                    if self.counts[gene][transcript].spliced > 0:
                        gene_name = db[gene]["gene_name"][0]
                        chrom = 'chr'+db[gene].chrom
                        transcript_name = db[transcript]["transcript_name"][0]

                        five_prime_utrs = ''
                        three_prime_utrs = ''

                        for fpu in db.children(transcript, featuretype="five_prime_utr"):
                            five_prime_utrs += fpu.sequence(fa)

                        for tpu in db.children(transcript, featuretype="three_prime_utr"):
                            three_prime_utrs += tpu.sequence(fa)

                        if return_intron_length:
                            self.gammas.append([
                                    chrom, 
                                    gene_name,
                                    transcript_name,
                                    self.counts[gene][transcript].spliced,
                                    self.counts[gene][transcript].unspliced,
                                    self.counts[gene][transcript].spliced / self.counts[gene][transcript].unspliced,
                                    self.lengths[gene][transcript].total_intron_length, 
                                    five_prime_utrs,
                                    three_prime_utrs])
                            self.gdf = pd.DataFrame(self.gammas, 
                                    columns = ['chr', 'gene', 'transcript', 'spliced', 'unspliced', 'gamma', 'total_intron_length', 'utr5', 'utr3'])
                            self.gdf.name = os.path.splitext(os.path.basename(self.bam.filename.decode()))[0] 

                        else: 
                            self.gammas.append([
                                    chrom, 
                                    gene_name,
                                    transcript_name,
                                    self.counts[gene][transcript].spliced,
                                    self.counts[gene][transcript].unspliced,
                                    self.counts[gene][transcript].spliced / self.counts[gene][transcript].unspliced,
                                    five_prime_utrs,
                                    three_prime_utrs])
                            self.gdf = pd.DataFrame(self.gammas, 
                                    columns = ['chr', 'gene', 'transcript', 'spliced', 'unspliced', 'gamma', 'utr5', 'utr3'])
                            self.gdf.name = os.path.splitext(os.path.basename(self.bam.filename.decode()))[0] 


    def write(self, path):
        out = self.gdf
        out.to_csv(path, sep="\t",header=False,index=False, mode = 'a')

class meta_reads:
    def __init__(self, readlist):
        if type(readlist[0]) is not str:
            self.dfs = [read.gdf.drop(columns=['spliced', 'unspliced']) for read in readlist]
            self.names = [read.gdf.name for read in readlist]
            i=0
            for df in self.dfs:
                df.rename(index=str, columns={'gamma': 'gamma_'+self.names[i]}, inplace=True)
                i+=1 

            self.final_df = self.dfs[0]
            for df in self.dfs[1:]:
                self.final_df = pd.merge(self.final_df, df, on=['chr', 'gene', 'transcript', 'utr5', 'utr3']) 
        else:
            colnames = ['chr', 'gene', 'transcript', 'spliced', 'unspliced', 'gamma', 'utr5', 'utr3']
            self.dfs = [pd.read_csv(read, names = colnames, sep='\t') for read in readlist]
            self.names = [os.path.splitext(os.path.basename(read))[0] for read in readlist]
            i=0
            for df in self.dfs:
                df.rename(index=str, columns={'gamma': 'gamma_'+self.names[i]}, inplace=True)
                df.rename(index=str, columns={'spliced': 'spliced_'+self.names[i]}, inplace=True)
                df.rename(index=str, columns={'unspliced': 'unspliced_'+self.names[i]}, inplace=True)
                i+=1 
            self.final_df = self.dfs[0]
            for df in self.dfs[1:]:
                self.final_df = pd.merge(self.final_df, df, on=['chr', 'gene', 'transcript', 'utr5', 'utr3']) 

            self.final_df['spliced'] = self.final_df.filter(regex='^spliced').sum(axis=1)
            self.final_df['unspliced'] = self.final_df.filter(regex='^unspliced').sum(axis=1)
            self.final_df['combined'] = self.final_df['spliced'] + self.final_df['unspliced']

            # Here we account for the fact htat some genes have shittons of isoforms
            # So we only take the 5 most prevalent.
            # This is a temporary solution.
            # I think this is safe since head will still return even if thres 0
            self.final_df2 = pd.DataFrame(columns = self.final_df.columns)
            for gene in self.final_df['gene'].unique():
                self.final_df2 = self.final_df2.append(self.final_df[self.final_df['gene']==gene].sort_values(by='combined').head(5), ignore_index = True)

            self.final_df2 = self.final_df2[self.final_df2.columns.drop(list(self.final_df2.filter(regex='spliced')))]
            self.final_df2 = self.final_df2.drop(columns='combined')


    def write(self, out):
        self.final_df2.to_csv(out, sep="\t",header=True,index=False)
