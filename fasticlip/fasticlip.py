#!/usr/bin/env python
# FAST-iCLIP

import os, cmath, math, sys, glob, subprocess, re, argparse, shutil, datetime, csv
from helper import *
import cfg
import numpy as np
import pandas as pd
from collections import defaultdict
from operator import itemgetter
import matplotlib as mpl
import matplotlib.pyplot as plt
from optparse import OptionParser
mpl.rcParams['savefig.dpi'] = 2 * mpl.rcParams['savefig.dpi']
mpl.rcParams['path.simplify'] = True
csv.register_dialect("textdialect",delimiter='\t')

### Checking environment variables ###
cfg.home = os.environ.get("FASTICLIP_PATH")
if not cfg.home and glob.glob("docs/"):
	cfg.home = os.getcwd()
if not cfg.home:
	print "Error: Environment variable FASTICLIP_PATH not found. Please set it to the FAST-iCLIP installation directory."
	exit()
if not glob.glob(cfg.home + "/docs/"):
	print "Error: docs folder not inside FASTICLIP_PATH. Make sure this environment variable is set correctly, or \
	that you have run ./configure inside the installation directory."
	exit()

### Parsing arguments ###
parser = argparse.ArgumentParser(description="FAST-iCLIP: a pipeline to process iCLIP data", epilog="Example: fasticlip -i rawdata/example_MMhur_R1.fastq rawdata/example_MMhur_R2.fastq --mm9 -n MMhur -o results")
parser.add_argument('-i', metavar='INPUT', nargs='+', help="Up to 4 input fastq or fastq.gz files separated by a space", required=True)
parser.add_argument('--trimmed', action='store_true', help="flag if files are already trimmed")
group = parser.add_mutually_exclusive_group()
group.add_argument('--hg19', action='store_true', help="required if your CLIP is from human")
group.add_argument('--mm9', action='store_true', help="required if your CLIP is from mouse")
parser.add_argument('--clipper', action='store_true', help="also run CLIPper on data")
parser.add_argument('-s', metavar="STAR_INDEX", help="Path to STAR index for your organism", required=True)
parser.add_argument('-n', metavar='NAME', help="Name of output directory", required=True)
parser.add_argument('-o', metavar='OUTPUT', help="Name of directory where output directory will be made", required=True)
parser.add_argument('-f', metavar='N', type=int, help="First base to keep on 5' end of each read. Default is 14.", default=14)
parser.add_argument('-a', metavar='ADAPTER', help="3' adapter to trim from the end of each read. Default is AGATCGGAAGAGCGGTTCAGCAGGAATGCCGAGACCGATCTCGTATGCCGTCTTCTGCTTG.", default='AGATCGGAAGAGCGGTTCAGCAGGAATGCCGAGACCGATCTCGTATGCCGTCTTCTGCTTG')
parser.add_argument('-tr', metavar='REPEAT_THRESHOLD_RULE', type=str, help="m,n: at least m samples must each have at least n RT stops mapped to repeat RNAs. Default is 1,4 (1 sample); 2,3 (2 samples); x,2 (x>2 samples)")
parser.add_argument('-tn', metavar='NONREPEAT_THRESHOLD_RULE', type=str, help="m,n: at least m samples must each have at least n RT stops mapped to nonrepeat RNAs. Default is 1,4 (1 sample); 2,3 (2 samples); x,2 (x>2 samples)")
parser.add_argument('-sr', metavar='STAR_RATIO', type=float, help="Maximum mismatches per base allowed for STAR genome mapping (corresponds to outFilterMismatchNoverLmax). Default is 0.08 (2 mismatches per 25 mapped bases).", default=0.08)
parser.add_argument('-bm', metavar='BOWTIE_MAPQ', type=int, help="Minimum MAPQ (Bowtie alignment to repeat/tRNA/retroviral indexes) score allowed. Default is 42.", default=42)
parser.add_argument('-q', metavar='Q', type=int, help="Minimum average quality score allowed during read filtering. Default is 25.", default=25)
parser.add_argument('-p', metavar='P', type=int, help="Percentage of bases that must have quality > q during read filtering. Default is 80.", default=80)
parser.add_argument('-l', metavar='L', type=int, help="Minimum length of read. Default is 15.", default=15)
parser.add_argument('--verbose', action='store_true', help="Print everything")

# organism
args = parser.parse_args()
if not (args.hg19 or args.mm9):
	print "Error: must include --hg19 or --mm9. Exiting."
	exit()
if args.hg19: org = 'human' 
else: org = 'mouse'

# input files
reads = args.i
for fn in reads:
	if not glob.glob(fn):
		print "Error: input file " + fn + " not accessible. Exiting."
		exit()
		
# sample name and output directory and STAR index
cfg.sampleName = args.n
cfg.outfilepath = args.o
if not glob.glob(cfg.outfilepath):
	print "Error: output directory " + cfg.outfilepath + " not accessible. Exiting."
	exit()
cfg.outfilepath = cfg.outfilepath + '/%s/'%cfg.sampleName
if not glob.glob(cfg.outfilepath): os.system("mkdir " + cfg.outfilepath)

star_index = args.s  # Location of STAR index
if not glob.glob(star_index):
	print "Error: STAR index at " + star_index + " not accessible. Exiting."
	exit()
	
# Create log and start pipeline
cfg.logFile=cfg.outfilepath + "runLog"
cfg.logOpen=open(cfg.logFile, 'w')

### Parameters ###

iCLIP3pBarcode = args.a # Barcode sequence to trim from reads.
star_ratio = args.sr  # Maximum mismatch/base ratio allowed for STAR
mapq = args.bm # Minimum MAPQ score allowed allowed for Bowtie2
q = args.q # Minimum quality score to keep during filtering.
p = args.p # Percentage of bases that must have quality > q during filtering.
l = args.l +  args.f - 1 # Minimum length of read + 5' adapter
iCLIP5pBasesToTrim=args.f # Number of reads to trim from 5' end of clip reads + 1 (this number represents the first base kept)
expand = 15 # Bases to expand around RT position after RT stops are merged.
cfg.run_clipper = True if args.clipper else False
cfg.verbose = True if args.verbose else False
CLIPPERoutNameDelim = '_' # Delimiter that for splitting gene name in the CLIPper windows file.

nsamp = len(reads)
if not args.tn:
	if nsamp == 1: threshold_nr = 4
	elif nsamp == 2: threshold_nr = 3
	else: threshold_nr = 2
	minpass_nr = nsamp
else:
	try: [minpass_nr, threshold_nr] = [int(x) for x in args.tn.split(',')]
	except: 
		print "Nonrepeat threshold rule must be in form m,n where m,n are integers"
		exit()
	minpass_nr = min(minpass_nr, nsamp) 

if not args.tr:
	if nsamp == 1: threshold_rep = 4
	elif nsamp == 2: threshold_rep = 3
	else: threshold_rep = 2
	minpass_rep = nsamp
else:
	try: [minpass_rep, threshold_rep] = [int(x) for x in args.tr.split(',')]
	except: 
		print "Repeat threshold rule must be in form m,n where m,n are integers."
		exit()
	minpass_rep = min(minpass_rep, nsamp) 

if org == 'human':
	repeat_index=cfg.home + '/docs/hg19/repeat/rep_spaced' # bt2 index for repeat RNA.
	repeatGenomeBuild=cfg.home+'/docs/hg19/repeat/repeatRNA_spaced.fa' # Sequence of repeat index.
	repeatAnnotation=cfg.home+'/docs/hg19/repeat/Hs_repeatIndex_spaced_positions.txt' # Repeat annotation file.
	retro_index=cfg.home+'/docs/hg19/retroviral/'
	start18s=3657
	end18s=5527
	start5s=6623
	end5s=6779
	start28s=7935
	end28s=12969
	rRNAend=13314
	index=cfg.home + '/docs/hg19/hg19/hg19' # bt2 index for mapping.
	index_tag='hg19' # Name of bt2 index.
	genomeFile=cfg.home+'/docs/hg19/human.hg19.genome' # Genome file for bedGraph, etc.
	genomeForCLIPper='-shg19' # Parameter for CLIPper.
	blacklistregions=cfg.home+'/docs/hg19/wgEncodeDukeMapabilityRegionsExcludable.bed' # Blacklist masker.
	repeatregions=cfg.home+'/docs/hg19/repeat_masker.bed' # Repeat masker.
	geneAnnot=glob.glob(cfg.home+'/docs/hg19/genes_types/*') # List of genes by type.
	miRNAmasker=cfg.home+'/docs/hg19/miR_sort_clean.bed' # miRNA masker file.
	fivePUTRBed=cfg.home+'/docs/hg19/5pUTRs_Ensbl_sort_clean_uniq.bed' # UTR annotation file.
	threePUTRBed=cfg.home+'/docs/hg19/3pUTRs_Ensbl_sort_clean_uniq.bed' # UTR annotation file.
	exonBed=cfg.home+'/docs/hg19/Exons_Ensbl_sort_clean_uniq.bed' # UTR annotation file.
	utrFile=cfg.home+'/docs/hg19/hg19_ensembl_UTR_annotation.txt' # UTR annotation file.
	genesFile=cfg.home+'/docs/hg19/hg19_ensembl_genes.txt' # Gene annotation file.
	sizesFile=cfg.home+'/docs/hg19/hg19.sizes' # Genome sizes file. 
	snoRNAindex=cfg.home+'/docs/hg19/snoRNA_reference/sno_coordinates_hg19_formatted.bed' # snoRNA coordinate file.
	tRNAindex=cfg.home+'/docs/hg19/trna/tRNA_hg19'  # this is now CCA tailed
	geneStartStopRepo=cfg.home+'/docs/hg19/all_genes.txt'
	geneStartStopRepoBed = cfg.home+'/docs/hg19/hg19_ensembl_genes_BED6.bed'
elif org == 'mouse':
	repeat_index=cfg.home + '/docs/mm9/repeat/rep_spaced' # bt2 index for repeat RNA.
	repeatGenomeBuild=cfg.home+'/docs/mm9/repeat/Mm_repeatRNA_spaced.fa' # Sequence of repeat index.
	repeatAnnotation=cfg.home+'/docs/mm9/repeat/Mm_repeatIndex_spaced_positions.txt' # Repeat annotation file.
	retro_index=cfg.home+'/docs/mm9/retroviral/'
	start18s=4007
	end18s=5876
	start5s=6877
	end5s=7033
	start28s=8123
	end28s=12836
	rRNAend=13401
	index=cfg.home + '/docs/mm9/mm9/mm9' # bt2 index for mapping.
	index_tag='mm9' # Name of bt2 index.
	genomeFile=cfg.home+'/docs/mm9/mm9.sizes' # Genome file for bedGraph, etc.
	genomeForCLIPper='-smm9' # Parameter for CLIPper.
	blacklistregions=cfg.home+'/docs/mm9/mm9-blacklist.bed' # Blacklist masker.
	repeatregions=cfg.home+'/docs/mm9/Mm_mm9_repeatMasker_formatted.bed' # Repeat masker.
	geneAnnot=glob.glob(cfg.home+'/docs/mm9/genes_types/*') # List of genes by type. 
	miRNAmasker=cfg.home+'/docs/mm9/mm9_miRNA.bed' # miRNA masker file.
	fivePUTRBed=cfg.home+'/docs/mm9/mm9_5pUTR.bed' # UTR annotation file.
	threePUTRBed=cfg.home+'/docs/mm9/mm9_3pUTR.bed' # UTR annotation file. 
	exonBed=cfg.home+'/docs/mm9/mm9_exons.bed' # UTR annotation file. 
	utrFile=cfg.home+'/docs/mm9/mm9_ensembl_UTR_annotation.txt' # UTR annotation file. 
	genesFile=cfg.home+'/docs/mm9/mm9_ensembl_genes.txt' # Gene annotation file. 
	sizesFile=cfg.home+'/docs/mm9/mm9.sizes' # Genome sizes file. 
	snoRNAindex=cfg.home+'/docs/mm9/snoRNA_reference/mm9_sno_coordinates_formatted_types.bed' # snoRNA coordinate file. 
	tRNAindex=cfg.home+'/docs/mm9/tRNA/tRNA_mm9'  # this is now CCA tailed
	geneStartStopRepo=cfg.home+'/docs/mm9/all_genes.txt'
	geneStartStopRepoBed = cfg.home+'/docs/mm9/mm9_ensembl_genes_BED6.bed'

### start running pipeline ###
now=datetime.datetime.now()
cfg.logOpen.write("Timestamp: %s\n"%str(now))
cfg.logOpen.write("\n###Parameters used###\n")
cfg.logOpen.write("3' barcode: %s\n'"%iCLIP3pBarcode)
cfg.logOpen.write("Minimum quality score (q): %s\n"%q)
cfg.logOpen.write("Percentage of bases with > q: %s\n"%p)
cfg.logOpen.write("5' bases to trim: %s\n'"%iCLIP5pBasesToTrim)
cfg.logOpen.write("Threshold for minimum number of RT stops (repeat): %s samples with >= %s RT stops\n"%(minpass_rep, threshold_rep))
cfg.logOpen.write("Threshold for minimum number of RT stops (nonrepeat): %s samples with >= %s RT stops\n"%(minpass_nr, threshold_nr))
cfg.logOpen.write("Bases for expansion around conserved RT stops: %s\n"%expand)
cfg.logOpen.write("\n\n\n")

def main():

	# 1. Trim and map
	
	log("\nProcessing sample {}".format(cfg.sampleName))

	if not args.trimmed: 
		log("\nRemoving duplicates")
		dup_removed_reads = remove_dup(reads, q, p)

	if not args.trimmed: 
		log("\nTrimming 5' and 3'")
		processed_reads = trim(dup_removed_reads, iCLIP3pBarcode, l, iCLIP5pBasesToTrim)
	else: processed_reads = reads

	log("\nRun mapping to indexes.")
	(rep_sam, retro_sam, trna_sam, gen_sam) = run_mapping(processed_reads, repeat_index, retro_index, tRNAindex, star_index, star_ratio)

	log("\nRun samtools.")
	rep_bed = run_samtools(rep_sam, "-q {}".format(mapq))
	retro_bed = run_samtools(retro_sam)  # do more with this later, so no flags yet
	trna_bed = run_samtools(trna_sam, "-q {}".format(mapq))
	gen_bed = run_samtools(gen_sam, "-q 255") # we're using STAR here so 255 signifies unique mapping

	# 2. Process repeat RT stops
	
	log("\nRun repeat and blacklist region masker.")
	
	log("\nRepeat RT stop isolation.")
	readsByStrand_rep=separateStrands(rep_bed)
	negativeRTstop_rep=isolate5prime(modifyNegativeStrand(readsByStrand_rep[0])) 
	positiveRTstop_rep=isolate5prime(readsByStrand_rep[1]) 

	log("Merge Repeat RT stops.")
	posMerged = cfg.outfilepath+cfg.sampleName+'_repeat_positivereads.mergedRT'
	negMerged = cfg.outfilepath+cfg.sampleName+'_repeat_negativereads.mergedRT'
	negAndPosMerged = cfg.outfilepath+cfg.sampleName+'_threshold=%s'%threshold_rep+'_repeat_allreads.mergedRT.bed'
	
	mergeRT(positiveRTstop_rep, posMerged, posMerged + '_stats', minpass_rep, threshold_rep, expand, '+')
	mergeRT(negativeRTstop_rep, negMerged, negMerged + '_stats', minpass_rep, threshold_rep, expand, '-')
	fileCat(negAndPosMerged, [posMerged, negMerged])
	fileCat(negAndPosMerged + '_stats', [posMerged + '_stats', negMerged + '_stats'])

	log("Nonrepeat RT stop isolation.")
	gen_norepeat_bed = remove_blacklist_retro(gen_bed, blacklistregions, repeatregions)
	readsByStrand = separateStrands(gen_norepeat_bed)
	negativeRTstop = isolate5prime(modifyNegativeStrand(readsByStrand[0])) 
	positiveRTstop = isolate5prime(readsByStrand[1]) 

	log("Merge Nonrepeat RT stops.")
	posMerged = cfg.outfilepath+cfg.sampleName+'_%s_positivereads.mergedRT'%index_tag
	negMerged = cfg.outfilepath+cfg.sampleName+'_%s_negativereads.mergedRT'%index_tag
	negAndPosMerged = cfg.outfilepath+cfg.sampleName+'_threshold=%s'%threshold_nr+'_%s_allreads.mergedRT.bed'%index_tag
	mergeRT(positiveRTstop, posMerged, posMerged + '_stats', minpass_nr, threshold_nr, expand, '+')
	mergeRT(negativeRTstop, negMerged, negMerged + '_stats', minpass_nr, threshold_nr, expand, '-')
	fileCat(negAndPosMerged,[posMerged,negMerged])
	fileCat(negAndPosMerged + '_stats', [posMerged + '_stats', negMerged + '_stats'])

	# 3. Process genic RT stops
	
	log("\nGetting list of snoRNAs")
	bedFile_sno = getSnoRNAreads(negAndPosMerged, snoRNAindex)
	if os.stat(bedFile_sno).st_size > 0:
		geneCounts_sno = countSnoRNAs(bedFile_sno) 
		outfilepathToSave = cfg.outfilepath + '/PlotData_ReadsPerGene_snoRNA'
		geneCounts_sno.to_csv(outfilepathToSave)
		
	log("\nFiltering out snoRNAs and miRNAs")
	sno_mirna_filtered_reads = filter_snoRNAs(negAndPosMerged, snoRNAindex, miRNAmasker)

	if not cfg.run_clipper: 
		log("\nAnnotating reads by gene")
		CLIPPERlowFDR = annotate_genes(sno_mirna_filtered_reads, geneStartStopRepoBed)
		CLIPperGeneList = make_gene_list_from_annotation(CLIPPERlowFDR) # list of genes; commented out all things making this
		# CLIPperOutBed  # cluster file; we're not making this, and this is not used anywhere else
	else:
		log("\nRunning CLIPper.")
		CLIPPERout = runCLIPPER(sno_mirna_filtered_reads, genomeForCLIPper, genomeFile)
		[CLIPPERlowFDR, CLIPperReadsPerCluster, CLIPperGeneList, CLIPperOutBed] = modCLIPPERout(sno_mirna_filtered_reads, CLIPPERout)

	log("Make bedGraphs")
	
	# pre-masking, pre-CLIPper/gene annotation bedgraph
	cleanBed = cleanBedFile(negAndPosMerged)
	bedGraphCLIPout = makeBedGraph(negAndPosMerged,genomeFile)
	CLIPPERlowFDRcenters = getBedCenterPoints(negAndPosMerged, expand, namecol=3)  # unannotated so there is no name column
	allLowFDRCentersBedGraph = makeBedGraph(CLIPPERlowFDRcenters, genomeFile)
	
	# post-masking, post-CLIPper/gene annotation bedgraph
	cleanBed = cleanBedFile(CLIPPERlowFDR)
	bedGraphCLIPout = makeBedGraph(cleanBed,genomeFile)
	CLIPPERlowFDRcenters = getBedCenterPoints(CLIPPERlowFDR, expand, namecol=9) # annotated so column 9 is the name of the gene
	allLowFDRCentersBedGraph = makeBedGraph(CLIPPERlowFDRcenters, genomeFile)	
	
	# 4. Partition reads by gene type
	
	# - all
	log("\nPartition reads by type.")
	pathToGeneLists = getLowFDRGeneTypes(CLIPperGeneList,geneAnnot)
	pathToReadLists = getLowFDRReadTypes(CLIPPERlowFDR,pathToGeneLists)
		
	# - protein coding
	geneRef=pd.DataFrame(pd.read_table(geneStartStopRepo))
	proteinCodingReads = cfg.outfilepath + 'clipGenes_proteinCoding_LowFDRreads.bed'
	proteinCodingReads_centered = getBedCenterPoints(proteinCodingReads, expand, namecol=9)
	geneCounts_pc = get_gene_counts(proteinCodingReads_centered) 
	cfg.outfilepathToSave = cfg.outfilepath + '/PlotData_ReadsPerGene_proteinCoding'
	geneCounts_pc.to_csv(cfg.outfilepathToSave)
	
	# - lncRNA
	lincRNAReads = cfg.outfilepath + 'clipGenes_lincRNA_LowFDRreads.bed'
	lincRNAReads_centered = getBedCenterPoints(lincRNAReads, expand, namecol=9)
	if os.stat(lincRNAReads_centered).st_size > 0:
		geneCounts_linc = get_gene_counts(lincRNAReads_centered)
		outfilepathToSave = cfg.outfilepath + '/PlotData_ReadsPerGene_lincRNA'
		geneCounts_linc.to_csv(outfilepathToSave)

		bf=pd.DataFrame(pd.read_table(lincRNAReads_centered,header=None))
		bf.columns=['Chr','Start','Stop','CLIPper_name','Q','Strand']
		bf['geneName']=bf['CLIPper_name'].apply(lambda x: x.split('_')[0])
		merge=pd.merge(geneRef,bf,left_on='Ensembl Gene ID',right_on='geneName')
		ncRNA_startStop=merge[['Ensembl Gene ID','Gene Start (bp)','Gene End (bp)','Start','Stop','Strand']]
		outfilepathToSave = lincRNAReads_centered.replace(".bed",".geneStartStop")
		ncRNA_startStop.to_csv(outfilepathToSave)

	# - other
	remaining = [f for f in glob.glob(cfg.outfilepath+"*_LowFDRreads.bed") if 'lincRNA' not in f and 'proteinCoding' not in f and 'snoRNA' not in f]
	countRemainingGeneTypes(remaining)

	if cfg.run_clipper:
		log("Get binding intensity around cluster centers.")
		bedGraphCLIPin=makeBedGraph(sno_mirna_filtered_reads,genomeFile)
		centerCoordinates=makeClusterCenter(CLIPperOutBed) 
		getClusterIntensity(bedGraphCLIPin,centerCoordinates)

	# 5. Process introns and UTRs
	
	log("\nIntron and UTR analysis.")
	exonreads, intronreads, fivePreads, threePreads, cdsreads = extract_regions(proteinCodingReads_centered, fivePUTRBed, threePUTRBed, exonBed)

	if os.stat(fivePreads).st_size > 0:
		geneCounts_5p=get_gene_counts(fivePreads) 
		outfilepathToSave=cfg.outfilepath+'/PlotData_ReadsPerGene_5pUTR'
		geneCounts_5p.to_csv(outfilepathToSave)

	if os.stat(threePreads).st_size > 0:
		geneCounts_3p=get_gene_counts(threePreads) 
		outfilepathToSave=cfg.outfilepath+'/PlotData_ReadsPerGene_3pUTR'
		geneCounts_3p.to_csv(outfilepathToSave)

	if os.stat(cdsreads).st_size > 0:
		geneCounts_cds=get_gene_counts(cdsreads) 
		outfilepathToSave=cfg.outfilepath+'/PlotData_ReadsPerGene_CDS'
		geneCounts_cds.to_csv(outfilepathToSave) 

	if os.stat(exonreads).st_size > 0:
		geneCounts_5p=get_gene_counts(exonreads) 
		outfilepathToSave=cfg.outfilepath+'/PlotData_ReadsPerGene_Exons'
		geneCounts_5p.to_csv(outfilepathToSave)

	if os.stat(intronreads).st_size > 0:
		geneCounts_3p=get_gene_counts(intronreads) 
		outfilepathToSave=cfg.outfilepath+'/PlotData_ReadsPerGene_Introns'
		geneCounts_3p.to_csv(outfilepathToSave)

	gene_binding_by_region()

	# 6. Analysis of gene bodies, CLIP binding sites (iCLIPro), tRNAs
	
	log("\nMake metagene.")
	proteinBedGraph = makeBedGraph(proteinCodingReads_centered, genomeFile)
	makeAvgGraph(proteinBedGraph, utrFile, genesFile, sizesFile)

	log("\nncRNA gene body analysis.")
	remaining=[f for f in glob.glob(cfg.outfilepath+"*_LowFDRreads.bed") if 'lincRNA' not in f and 'proteinCoding' not in f and 'snoRNA' not in f]
	for bedFile in remaining:
		st_stop=getGeneStartStop(bedFile,geneRef)

	log("\nRun iCLIPro.")
	iclipro = run_iclipro(gen_sam)

	log("\nRun tRNA isotype counting.")
	trna_readlist = trna_isotype_count(trna_sam, minpass_rep, threshold_rep)

	# 7. Repeat RNAs
	
	log("\nRecord repeat RNA.")
	repeat_genome_bases,repeatAnnotDF=makeRepeatAnnotation(repeatGenomeBuild,repeatAnnotation)
	repeatAnnotDF.set_index('Name',inplace=True,drop=False)
	repeatMerged=glob.glob(cfg.outfilepath+"*repeat_allreads.mergedRT.bed") # Get merged data for repeat index.
	rep=pd.read_table(repeatMerged[0],dtype=str,header=None)
	rep.columns=['Rep_index','Start','Stop','Read_name','Q','Strand']
	rep['RT_stop']=rep['Start'].astype(int)+expand
	for ix in repeatAnnotDF.index:
		end=repeatAnnotDF.loc[ix,'IndexEnd']
		repName=repeatAnnotDF.loc[ix,'Name']
		gene_hits=rep[(rep['RT_stop']<int(repeatAnnotDF.loc[ix,'IndexEnd']))&(rep['RT_stop']>int(repeatAnnotDF.loc[ix,'IndexStart']))]
		gene_hits['Repeat_End']=repeatAnnotDF.loc[ix,'IndexEnd']
		gene_hits['Repeat_Start']=repeatAnnotDF.loc[ix,'IndexStart']
		outfilepathToSave=cfg.outfilepath + '/PlotData_RepeatRNAreads_%s'%repName
		gene_hits.to_csv(outfilepathToSave)
	
	# 8. Plots
	
	log("\nMake plots.")
	import matplotlib
	import commands
	matplotlib.rcParams['savefig.dpi'] = 2 * matplotlib.rcParams['savefig.dpi']

	log("Making Figure 1")
	fig1 = plt.figure(1)
	plot_figure1(nsamp, reads, threshold_nr, index_tag)
	fig1.tight_layout()
	fig1.savefig(cfg.outfilepath+'Figure1.png',format='png',bbox_inches='tight',dpi=150,pad_inches=0.5)

	log("Making Figure 2")
	ensemblGeneAnnot = pd.DataFrame(pd.read_table(genesFile))
	ensemblGeneAnnot = ensemblGeneAnnot.set_index('name') # Make ENST the index

	fig2 = plt.figure(2)
	plt.subplot2grid((2,4),(0,0),colspan=4)
	plot_mRNAgeneBodyDist()
	plot_geneBodyPartition(ensemblGeneAnnot)
	fig2.tight_layout()
	fig2.savefig(cfg.outfilepath+'Figure2.png',format='png',bbox_inches='tight',dpi=150,pad_inches=0.5)
	fig2.savefig(cfg.outfilepath+'Figure2.pdf',format='pdf',bbox_inches='tight',dpi=150,pad_inches=0.5)

	log("Making Figure 3")
	fig3 = plt.figure(3)
	plot_repeatRNA(repeatGenomeBuild)
	fig3.tight_layout()
	fig3.savefig(cfg.outfilepath+'Figure3.png',format='png',bbox_inches='tight',dpi=150,pad_inches=0.5)
	fig3.savefig(cfg.outfilepath+'Figure3.pdf',format='pdf',bbox_inches='tight',dpi=150,pad_inches=0.5)

	log("Making Figure 4")
	fig4 = plt.figure(4)
	plot_rDNA(start18s, end18s, start5s, end5s, start28s, end28s, rRNAend)
	fig4.tight_layout()
	fig4.savefig(cfg.outfilepath+'Figure4.png',format='png',bbox_inches='tight',dpi=150,pad_inches=0.5)
	fig4.savefig(cfg.outfilepath+'Figure4.pdf',format='pdf',bbox_inches='tight',dpi=150,pad_inches=0.5)

	log("Making Figure 5")
	if os.stat(bedFile_sno).st_size > 0:
		fig5 = plt.figure(5)
		plot_snorna(bedFile_sno)
		fig5.tight_layout()
		fig5.savefig(cfg.outfilepath+'Figure5.png',format='png',bbox_inches='tight',dpi=150,pad_inches=0.5)
		fig5.savefig(cfg.outfilepath+'Figure5.pdf',format='pdf',bbox_inches='tight',dpi=150,pad_inches=0.5)
	else:
		log("No snoRNA reads; not making Figure 5")
		
	log("Making Figure 6")
	st_stopFiles = glob.glob(cfg.outfilepath+"*.geneStartStop")
	st_stopFiles = [f for f in st_stopFiles if 'rRNA' not in f]
	fig6 = plt.figure(6)
	plot_ncrnas(st_stopFiles, expand)
	fig6.tight_layout()
	fig6.savefig(cfg.outfilepath+'Figure6.png',format='png',bbox_inches='tight',dpi=150,pad_inches=0.5)
	fig6.savefig(cfg.outfilepath+'Figure6.pdf',format='pdf',bbox_inches='tight',dpi=150,pad_inches=0.5)

	clean_up()
	cfg.logOpen.close()

if __name__ == "__init__":
	main()

	
