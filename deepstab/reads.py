import numpy as np
import pandas as pd
import pdb
import os.path
from keras.preprocessing import sequence
import h5py
from tqdm import tqdm
from Bio import pairwise2

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
        #assert sum(i_start_idx) <= 1 # This can actually be true if you have side by side introns...
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

    def transcript_to_annotated_sequence(self, transcript, db, fa, f, name):
        """Given the name of a trasncsritp (from the nxt function), add up all the sequences of its constitutent portions and annotate it with a code to dednote whether it is an exon, intrn, utr, etc.

        I'm also forced to infer introns on the fly to get the annotations. Not perfect. It doesnt catch the case of introns at the end...

        I also would like to include the information about which portions are coding. I don't know how to ddo tat right now.

        :param transcript: Unique ID of the transcript
        :param db: A link to the GTF database created using GTFutils 
        :param fa: Fasta file
        :param f: File handle of the h5py file used for writing. As of 2/9/19
        :param name: Name of the dataset to save. As of 2/9/2019
        """
        tx_sequence = db[transcript].sequence(fa) 
        
        was_exon = False

        start = db[transcript].start
        end = db[transcript].end

        annotations = ['?']*(end-start+1)
        
        for feature in db.children(transcript, order_by = 'start'):
            
            ft = feature.featuretype
            fs = feature.start - start
            fe = feature.end - start
            
            if ft == "transcript": 
                code = "I"
            elif ft == "CDS":
                code = "C"
            elif ft == "five_prime_utr": 
                code = '5'
            elif ft == "three_prime_utr":
                code = '3'
            elif ft == "start_codon":
                code = "S"
            elif ft == "exon":
                code = 'E'
            elif ft == "stop_codon":
                code = "s"
             
            # To account for the interval notation in the GTF files.
            # See sanity checks.
            annotations[fs:(fe+1)] = code*(fe-fs+1)
 
        assert(len(annotations) == len(tx_sequence))
        return (''.join(annotations), tx_sequence)
 
    def add_utr_and_gamma(self, db, dset, fa, return_intron_length = False):
        self.gammas = []

        self.possible_annotations = [
                "I", # Intron
                "C", # CDS
                "5", # 5' UTR
                "3", # 3' UTR
                "S", # Start codon
                "E", # Exon
                's'] # Stop codon
        
        for gene in tqdm(self.counts.keys(), desc='Annotating'):
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

                        with h5py.File(dset) as f:
                            name = chrom+'/'+transcript
                            annotation, whole_transcript_sequence = self.transcript_to_annotated_sequence(transcript, db, fa, f, name) 
                            hot_seq = self.get_hot_coded_seq(whole_transcript_sequence)
                            hot_annot = self.get_hot_coded_annotations(annotation, self.possible_annotations)

                            seq = np.concatenate( (hot_seq, hot_annot), 1)

                            tx = f.create_dataset(name, data=seq) 
                            tx.attrs['label'] = self.counts[gene][transcript].unspliced / self.counts[gene][transcript].spliced 

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
                                    three_prime_utrs,
                                    whole_transcript_sequence,
                                    annotation])
                            self.gdf = pd.DataFrame(self.gammas, 
                                    columns = ['chr', 'gene', 'transcript', 'spliced', 'unspliced', 'gamma', 'total_intron_length', 'utr5', 'utr3', 'sequence', 'annotation'])
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
                                    three_prime_utrs,
                                    whole_transcript_sequence,
                                    annotation])
                            self.gdf = pd.DataFrame(self.gammas, 
                                    columns = ['chr', 'gene', 'transcript', 'spliced', 'unspliced', 'gamma', 'utr5', 'utr3', 'sequence', 'annotation'])
                            self.gdf.name = os.path.splitext(os.path.basename(self.bam.filename.decode()))[0] 


    def write(self, path):
        """Obviously still works but might be moving towards just converting in line"""
        out = self.gdf
        out.to_csv(path, sep="\t",header=True,index=False, mode = 'a')

    def get_hot_coded_seq(self, sequence):
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

    def get_hot_coded_annotations(self, annotations, possible_annotations): 
        codes = [c for c in range(len(possible_annotations))]
        lookup = dict(zip(possible_annotations, codes)) 
        hotann = np.zeros(( len(annotations), len(possible_annotations) ))
        for i in range(len(annotations)):
            try:
                hotann[i, lookup[annotations[i]]] = 1
            except KeyError:
                continue
        return hotann





class tf_input:
    """Create train, test, and validation data."""
    def __init__(self, gdf, dset, length=5000):   
        self.df = self.qc(gdf)
        self.df = self.df.reset_index()
        self.dset = dset

        #pdb.set_trace()

        self.batch(self.df, bsize = 32)


    def batch(self, df, bsize, chrs = [1]):
        
        df['length'] = [len(s) for s in df['sequence']]

        # The chr of interest is the 
        # TEST set. We will isolate the rest
        # for training. 
        for chr in chrs:

            training = df[df['chr'] != f"chr{chr}"].sort_values('length')
            test = df[df['chr'] == f"chr{chr}"].sort_values('length')

            train_iter = training.groupby(np.arange( len(training)) // bsize)
            test_iter = test.groupby(np.arange(len(test))//bsize)

            for i, dset in train_iter:
                seq, label = self.code_all(dset, length = min(max(dset['length']), 50000))

                with h5py.File(self.dset) as f:
                    f.create_dataset(f"chr_{chr}/train/chunk_{i}_seq", data=seq)
                    f.create_dataset(f"chr_{chr}/train/chunk_{i}_label", data=label)

            for i, dset in test_iter:
                seq, label = self.code_all(dset, length = min(max(dset['length']), 50000 ))

                with h5py.File(self.dset) as f:
                    f.create_dataset(f"chr_{chr}/test/chunk_{i}_seq", data=seq)
                    f.create_dataset(f"chr_{chr}/test/chunk_{i}_label", data=label)

    def code_all(self, df, length=5000):
        rows = len(df.index)
        self.possible_annotations = [
                "I", # Intron
                "C", # CDS
                "5", # 5' UTR
                "3", # 3' UTR
                "S", # Start codon
                "E", # Exon
                's'] # Stop codon

       # Handle the sequences
        hot_seqs = np.zeros(( rows, length, 4 ))
        hot_ann = np.zeros(( rows, length, len(self.possible_annotations) ))

        padded_seq = sequence.pad_sequences(
            [list(l) for l in df['sequence']],
            maxlen=length,
            dtype='str',
            value = 'P')
        padded_ann = sequence.pad_sequences(
            [list(l) for l in df['annotation']],
            maxlen=length,
            dtype='str',
            value = 'P')

        for j in range(rows):
            hot_seqs[j,] = self.get_hot_coded_seq(padded_seq[j])
            hot_ann[j,] = self.get_hot_coded_annotations(padded_ann[j], self.possible_annotations)

        seq = np.concatenate( (hot_seqs, hot_ann), 2)
        labels = df.gamma.tolist()
 
        return(seq, labels)  

    def save_all(self, file):
        with h5py.File(file, 'w') as f:
            f.create_dataset('seq', data=self.sequence)
            f.create_dataset('labels', data=self.labels)

    def save_by_chr(self, file):
        """ This indexing might cause a problem if I do QC and put it in a different object."""
        #pdb.set_trace()
        for chr in self.df.chr.unique():
            idx = self.df.query('chr == "'+chr+'"').index.tolist()
            with h5py.File(file) as f:
                f.create_dataset(chr+'_seq', data=[self.sequence[i] for i in idx])
                f.create_dataset(chr+'_labels', data=[self.labels[i] for i in idx])
 
    def get_hot_coded_seq(self, sequence):
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

    def get_hot_coded_annotations(self, annotations, possible_annotations): 
        codes = [c for c in range(len(possible_annotations))]
        lookup = dict(zip(possible_annotations, codes)) 
        hotann = np.zeros(( len(annotations), len(possible_annotations) ))
        for i in range(len(annotations)):
            try:
                hotann[i, lookup[annotations[i]]] = 1
            except KeyError:
                continue
        return hotann

    def qc(self, df):
        df = df[~df.sequence.isna()]
        df = df[df.spliced + df.unspliced > 10] # from the RNA seq tutorial.
        df = df[df.gamma < 20]
        # This is infeasible now for gene level sequences
        #df = self.collapse_seq(df, 0.8)
        
        #pdb.set_trace()
        df['total'] = df['spliced'] + df['unspliced']
        df = df.sort_values('total', ascending = False).groupby('gene').head(2)
        df.gamma = np.log(df.gamma)
        #pdb.set_trace()
        df.reset_index()
        return(df)

    def collapse_seq(self, df, p):
        df['dup'] = False
        for gene in tqdm(df['gene'].unique().tolist(), total= len(df['gene'].unique().tolist()), desc = 'Genes:'):
            print(gene)
            for row1 in df[df['gene'] == gene].iterrows():
                for row2 in df[df['gene'] == gene].iterrows():
                    if row1[0] != row2[0]:
                        if not df.loc[row1[0], 'dup'] or df.loc[row2[0], 'dup']:
                            if self.overlap(row1[1].sequence,row2[1].sequence) > p*len(row1[1].sequence):
                                pdb.set_trace()
                                df.loc[row2[0], 5] =  np.mean([row1[1][5], row2[1][5]])
                                df.loc[row1[0], 'dup'] = True
        df_out = df[~df['dup']].drop('dup', axis=1)
        return df_out

    def overlap(self, seq1, seq2):
        return pairwise2.align.globalxx(seq1, seq2, score_only=True)





class meta_reads:
    """ This is probably a bit outdated now, need to include the new sequence annotations"""
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
