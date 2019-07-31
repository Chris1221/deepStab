#import velocyto as vcy
import math as ma
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import loompy
import pdb


class entrySimple:
	"""More simple structure for the barebones fasta from the bedtools runs"""
	def __init__(self, _lines: str): 
		if _lines != "":
			self.geneName = _lines[1:].rstrip("\n")
			self.seq = []
			self.empty = False
		elif _lines == "":
			self.empty = True
	
	def append_seq(self, _lines):
		self.seq.append(_lines.rstrip("\n"))

class entry:
	"""Container for each entry"""
	def __init__(self, _lines: str):
		if _lines != "":
			line = _lines.split('|')
			self.geneID = line[0][1:]  
			self.stableID = line[1]
			self.geneName = line[2]
			self.geneStart = line[3]
			self.geneEnd = line[4]
			try:
				self.utrStart = line[5]
			except IndexError:
				pass

			try:
				self.utrEnd = line[6]
			except IndexError:
				pass

			self.seq = ''
			self.empty = False
		elif _lines == "": 
			self.empty = True

	def append_seq(self, _lines):
		self.seq += _lines.rstrip("\n")


class UTRsimple: 
	"""Parses the simple FASTA data with just the gene name"""
	def __init__(self, _direction, path): 
		self.direction = _direction
		self.entries = {}
		self.duplicated_genes = []

		# Open an empty one for the first run through
		current = entrySimple("")

		dat = open(path, "r")
		for lines in dat:
			if lines.startswith('>'):
				if not current.empty:
					if current.geneName in self.entries: 
						# This gene alraedy has an entry so 
						# add the sequence
						assert len(current.seq) == 1
						self.entries[current.geneName].append_seq(current.seq[0])
					else:
						self.entries[current.geneName] = current

				current = entrySimple(lines)	
			else:
				current.append_seq(lines)

	def return_seq(self, _geneID: str):
		try: 
			return self.entries[_geneID].seq
		except KeyError:
			return ''	

class UTR:
	"""Contains the biomart information on UTRs.
	
	Parses the data file and stores it in a retrievable way."""
	def __init__(self, _direction, path):	
		self.direction = _direction
		self.entries = {}
		self.duplicated_genes = []

		current = entry("")

		dat = open(path, "r")
		for lines in dat: 
			if lines.startswith('>'):	
				if not current.empty: 
					if current.geneName in self.entries:
						# this is a duplicated entry, append it
						self.entries[current.geneName] = [self.entries[current.geneName], current]
						self.duplicated_genes.append(current.geneName)
					else:
						self.entries[current.geneName] = current

					self.entries[current.geneName] = current

				current = entry(lines)
			else: 
				current.append_seq(lines)	
	
	def return_seq(self, _geneID: str):
		try: 
			return self.entries[_geneID].seq
		except KeyError:
			return ''	

def process_col8(_line):
    """The 8th column is the one with all the meta information"""
    # It should never have length 0 (or even close!) 
    # Split it into its component parts 
    assert len(_line[8]) > 0
    _meta = _line[8].split(';')

    info = {}

    # Then take each pairing and make it into a dictionary 
    for entry in _meta:
        if entry.startswith(" "):
            entry = entry[1:]

        pair = entry.split(" ") 

        if len(pair) == 1: 
            continue

        if pair[1].startswith("\""):
            pair[1] = pair[1].strip("\"")

        info[pair[0]] = pair[1]

    return info


    

class gene:
    """A container per gene for the large GTF file"""
    def __init__(self, _line):
        self.info = process_col8(_line)
        self.geneName = self.info['gene_name']
        self.transcripts = {}

class transcript:
    """A container per transcript of a gene in a dictionary"""
    def __init__(self, _line):
        self.chr =                  _line[0]
        self.start =                _line[3]
        self.stop =                 _line[4]
        self.strand =               _line[6]
        self.info =                 process_col8(_line)

        self.transcript_name =      self.info['transcript_name']
        self.geneName =             self.info['gene_name']
        self.transcript_version =   self.info['transcript_version']

        self.exons =                {}
        self.utrs =                 {}

class exon:
    """A container per exon of a transcript"""
    def __init__(self, _line):
        self.info =                 process_col8(_line)
        self.geneName =             self.info['gene_name']
        self.transcript_name =      self.info['transcript_name']
        self.exon_number =          self.info['exon_number']

class utr:
    """General container for UTRs"""
    def __init__(self, _line, _direction):
        self.direction =            _direction
        self.info =                 process_col8(_line)
        self.geneName =             self.info['gene_name'] 
        self.transcript_name =      self.info['transcript_name']



class meta:
    """Holds all the meta information in a series of genes and transcripts..."""

    def append_transcript(self, _transcript):
        """Convenience to append a transcript to a genes records."""
        try:
            self.genes[_transcript.geneName].transcripts[_transcript.transcript_name] = _transcript
        except KeyError:
            pdb.set_trace()

    def append_exon(self, _exon):
        """Add on an exon"""
        self.genes[_exon.geneName].transcripts[_exon.transcript_name].exons[_exon.exon_number] = _exon

    def append_utr(self, _utr):
        """For some reason theres no unique identifier of the UTRs"""
        self.genes[_utr.geneName].transcripts[_utr.transcript_name].utrs[_utr.direction].append(_utr) # this might not work

    def __init__(self, path):
        dat = open(path, "r")

        for lines in dat:
            # This is the meta information at the top.
            if lines.startswith("#!"):
                continue

            line = lines.rstrip("\n").split("\t")
            type = line[2]

            self.genes = {} 

            if type == "gene":
                gene_read = gene(line)
                self.genes[gene_read.geneName] = gene_read
            elif type == "transcript":  
                self.append_transcript(transcript(line))
            elif type == "exon":
                self.append_exon(exon(line))
            elif type == "five_prime_utr":
                self.append_utr(utr(line, 5))
            elif type == "three_prime_utr":
                self.append_utr(utr(line,3))
            elif type == "CDS":
                pass
            else:
                raise("Strange type")

