#!/usr/bin/env python
# Copyright 2017 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from optparse import OptionParser
import collections
import gzip
import pdb
import os
import random
import sys
import numpy as np

import pandas as pd
import pyranges as pr

from baskerville import data
from collections import deque

"""
hound_data_filter.py

Compare sequences from a query species to pre-partitioned train/valid/test
splits of a target species, restricting homologous sequence leakage.
"""


################################################################################
def main():
    usage = "usage: %prog [options] <align_gff3> <target_sequences_bed> <query_fasta_file>"
    parser = OptionParser(usage)
    parser.add_option(
        "-a",
        dest="genome_label",
        default=None,
        help="Genome labels for query species"
    )
    parser.add_option(
        "-c",
        "--crop",
        dest="crop_bp",
        default=0,
        type="int",
        help="Crop bp off each end [Default: %default]",
    )
    parser.add_option(
        "--pad",
        dest="pad_bp",
        default=0,
        type="int",
        help="Pad each sequence with extra bp [Default: %default]",
    )
    parser.add_option(
        "-g",
        dest="gap_file",
        default=None,
        help="Assembly gaps BED file [Default: %default]",
    )
    parser.add_option(
        "-l",
        dest="seq_length",
        default=131072,
        type="int",
        help="Sequence length [Default: %default]",
    )
    parser.add_option(
        "-o",
        dest="out_dir",
        default="align_out",
        help="Output directory [Default: %default]",
    )
    parser.add_option(
        "-s",
        dest="sample_pct",
        default=1.0,
        type="float",
        help="Down-sample the segments",
    )
    parser.add_option(
        "--seed",
        dest="seed",
        default=44,
        type="int",
        help="Random seed [Default: %default]",
    )
    parser.add_option(
        "--snap",
        dest="snap",
        default=1,
        type="int",
        help="Snap sequences to multiple of the given value [Default: %default]",
    )
    parser.add_option(
        "--stride",
        dest="stride",
        default=1.0,
        type="float",
        help="Stride to advance train sequences [Default: seq_length]",
    )
    parser.add_option(
        "-u",
        dest="umap_bed",
        default=None,
        help="Genome unmappable segments to set to NA",
    )
    parser.add_option(
        "--umap_t",
        dest="umap_t",
        default=0.5,
        type="float",
        help="Remove sequences with more than this unmappable bin % [Default: %default]",
    )
    parser.add_option(
        "--chrom_alias",
        dest="chrom_alias_file",
        default=None,
        help="Chromosome alias file for target",
    )
    parser.add_option(
        "--q_chrom_alias",
        dest="query_chrom_alias_file",
        default=None,
        help="Chromosome alias file for target",
    )
    parser.add_option(
        "-w",
        dest="pool_width",
        default=32,
        type="int",
        help="Sum pool width [Default: %default]",
    )
    parser.add_option(
        "--query_bed",
        dest="query_bed_file",
        default=None,
        help="Pre-generated sequence bed file",
    )
    parser.add_option(
        "--tel",
        dest="telomere_crop_bp",
        default=131072,
        type="int",
        help="Crop this many bp from each chromosome [Default: %default]",
    )
    parser.add_option(
        "--min_pct",
        dest="min_pct_ident",
        default=10,
        type="int",
        help="Minimum alignment pct identity [Default: %default]",
    )
    parser.add_option(
        "--min_aln",
        dest="min_align_size",
        default=16384,
        type="int",
        help="Minimum alignment bp size [Default: %default]",
    )
    parser.add_option(
        "--break_aln",
        dest="break_align_size",
        default=None,#131072,
        type="int",
        help="Break alignments down to this target size [Default: %default]",
    )
    parser.add_option(
        "--t_olap_min",
        dest="t_olap_min",
        default=16384,
        type="int",
        help="Minimum alignment overlap with target sequence window [Default: %default]",
    )
    parser.add_option(
        "--q_olap_min",
        dest="q_olap_min",
        default=16384,
        type="int",
        help="Minimum alignment overlap with query sequence window [Default: %default]",
    )
    parser.add_option(
        "--shared_olap_min",
        dest="shared_olap_min",
        default=32768,
        type="int",
        help="Minimum total aggregated overlap between query and target windows [Default: %default]",
    )
    parser.add_option(
        "--max_chrom_size",
        dest="max_chrom_size",
        default=568000000,
        type="int",
        help="Maximum chromosome size (to avoid integer problems) [Default: %default]",
    )
    parser.add_option(
        "--keep_multi_olap",
        dest="keep_multi_olap",
        default=False,
        action="store_true",
        help="Keep query sequences that match multiple target fold labels [Default: %default]",
    )
    parser.add_option(
        "--keep_free",
        dest="keep_free",
        default=False,
        action="store_true",
        help="Keep query sequences without any target overlap [Default: %default]",
    )
    
    (options, args) = parser.parse_args()

    if len(args) != 3:
        parser.error("Must provide alignment GFF3 file, target sequence bed and query FASTA.")
    else:
        align_gff_file = args[0]
        sequences_bed_file = args[1]
        fasta_file = args[2]

    # set random options.seed
    random.seed(options.seed)
    np.random.seed(options.seed)

    # transform proportion options.strides to base pairs
    if options.stride <= 1:
        print("options.stride %.f" % options.stride, end="")
        options.stride = options.stride * options.seq_length
        print(" converted to %f" % options.stride)
    options.stride = int(np.round(options.stride))

    # check options.snap
    if options.snap is not None:
        if np.mod(options.seq_length, options.snap) != 0:
            raise ValueError("options.seq_length must be a multiple of options.snap")
        if np.mod(options.stride, options.snap) != 0:
            raise ValueError("options.stride must be a multiple of options.snap")

    # create output directory
    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    genome_out_dir = "%s/%s" % (options.out_dir, options.genome_label)
    if not os.path.isdir(genome_out_dir):
        os.mkdir(genome_out_dir)

    # calculate target length
    seq_tlength = options.seq_length - 2 * options.crop_bp

    ################################################################
    # define genomic contigs
    ################################################################

    contigs = None
    if options.query_bed_file is None :
        genome_chr_contigs = data.load_chromosomes(fasta_file)

        # crop chromosomes
        genome_chr_contigs_trim = {}

        # loop over contigs
        for chrom in genome_chr_contigs :
            contig = genome_chr_contigs[chrom][0]

            # crop chromosome
            if options.telomere_crop_bp is not None :
                genome_chr_contigs_trim[chrom] = [(min(options.telomere_crop_bp, contig[1]), min(max(contig[1] - options.telomere_crop_bp, 0), options.max_chrom_size))]
            else :
                genome_chr_contigs_trim[chrom] = [(contig[0], min(contig[1], options.max_chrom_size))]

        genome_chr_contigs = genome_chr_contigs_trim

        # filter for large enough
        genome_chr_contigs = {
            ctg : genome_chr_contigs[ctg]
            for ctg in genome_chr_contigs if genome_chr_contigs[ctg][0][1] - genome_chr_contigs[ctg][0][0] >= 2 * options.seq_length
        }

        # remove gaps
        if options.gap_file is not None:
            genome_chr_contigs = data.split_contigs(
                genome_chr_contigs, options.gap_file
            )

        # ditch the chromosomes
        contigs = []
        for chrom in genome_chr_contigs:
            contigs += [
                data.Contig(0, chrom, ctg_start, ctg_end)
                for ctg_start, ctg_end in genome_chr_contigs[chrom]
            ]

        # filter (again) for large enough
        contigs = [ctg for ctg in contigs if ctg.end - ctg.start >= 2 * options.seq_length]

        # print contigs to BED file
        ctg_bed_file = "%s/contigs.bed" % genome_out_dir
        data.write_seqs_bed(ctg_bed_file, contigs)

    # load sequence bed file for target species
    seq_df = pd.read_csv(sequences_bed_file, names=['chrom', 'start', 'end', 'label'], sep='\t')
    seq_df['start'] -= options.crop_bp
    seq_df['end'] += options.crop_bp

    # (optionally) load chrom.alias file
    if options.chrom_alias_file is not None :
        target_dict = {}
        with open(options.chrom_alias_file, 'rt') as f :
            for line in f :
                line_parts = line.strip().split('\t')
                target_dict[line_parts[0]] = line_parts[1]

        # translate target chromosomes (if needed)
        if seq_df.iloc[0]['chrom'] in target_dict :
            chroms = []
            for _, row in seq_df.iterrows() :
                chroms.append(target_dict[row['chrom']])

            seq_df['chrom'] = chroms

    ################################################################
    # divide between train/valid/test
    ################################################################

    # load alignment gff3
    gff_df = None
    with open(align_gff_file, "rt") as f :

        last_pos = f.tell()
        line = f.readline()
        while line is not None :
            if not line.strip().startswith("#") :
                f.seek(last_pos)
                break

            last_pos = f.tell()
            line = f.readline()

        gff_df = pd.read_csv(f, sep='\t', names=['chrom_t', 'source', 'alignment_type', 'start_t', 'end_t', 'feat1', 'strand', 'feat2', 'id_str'])

        gff_df['chrom_q'] = gff_df['id_str'].apply(lambda x: x.split(';Target=')[1].split(' ')[0])
        gff_df['start_q'] = gff_df['id_str'].apply(lambda x: int(x.split(';Target=')[1].split(' ')[1]))
        gff_df['end_q'] = gff_df['id_str'].apply(lambda x: int(x.split(';Target=')[1].split(' ')[2]))
        gff_df['num_ident'] = gff_df['id_str'].apply(lambda x: int(x.split('num_ident=')[1].split(';')[0]))

        gff_df['pct_identity_gap'] = gff_df['id_str'].apply(lambda x: float(x.split('pct_identity_gap=')[1].split(';')[0]))
        gff_df['pct_identity_ungap'] = gff_df['id_str'].apply(lambda x: float(x.split('pct_identity_ungap=')[1].split(';')[0]))

        gff_df['reciprocity'] = gff_df['id_str'].apply(lambda x: int(x.split('reciprocity=')[1].split(';')[0]))

        def _get_gap_str(row) :
            if ';Gap=' in row['id_str'] :
                return row['id_str'].split(';Gap=')[1]
            else :
                return 'M' + str(row['end_t'] - row['start_t'] + 1)

        gff_df['gap_str'] = gff_df.apply(_get_gap_str, axis=1)

        gff_df['alignment_size'] = gff_df['num_ident'] / (gff_df['pct_identity_gap'] / 100.)
        gff_df.loc[gff_df['alignment_size'].isnull(), 'alignment_size'] = 0.
        gff_df['alignment_size'] = gff_df['alignment_size'].astype(int)

        # filter on reciprocal alignments
        gff_df = gff_df.loc[(gff_df['reciprocity'] == 3) & (gff_df['alignment_type'] == 'match')].copy()

        # filter on alignment size and quality
        gff_df = gff_df.loc[gff_df['alignment_size'] >= options.min_align_size].copy()
        gff_df = gff_df.loc[gff_df['pct_identity_gap'] >= float(options.min_pct_ident)].copy()

        # retain small selection of columns
        gff_df = gff_df[[
            'chrom_t', 'start_t', 'end_t', 'chrom_q', 'start_q', 'end_q', 'strand', 'gap_str', 'pct_identity_gap', 'pct_identity_ungap', 'num_ident', 'alignment_size'
        ]].copy().reset_index(drop=True)

        # break up large alignments into sub-alignments
        if options.break_align_size is not None :
            gff_rows = []
            for _, row in gff_df.iterrows() :

                # add row back to dataframe
                if row['end_t'] - row['start_t'] + 1 <= options.break_align_size :
                    gff_rows.append([
                        row['chrom_t'], row['start_t'], row['end_t'], row['chrom_q'], row['start_q'], row['end_q'], row['pct_identity_gap'], row['pct_identity_ungap'], row['num_ident'], row['alignment_size']
                    ])
                else : # split row into sub-alignments
                    sub_alignments = []

                    # split alignment with same strand orientation
                    if row['strand'] == '+' :
                        sub_alignments = _break_large_alignment_sense(row, break_align_size=options.break_align_size)
                    elif row['strand'] == '-' : # split alignment with differrent strand orientation
                        sub_alignments = _break_large_alignment_antisense(row, break_align_size=options.break_align_size)

                    # add each sub-alignment as a new row in dataframe
                    for aln in sub_alignments :
                        gff_rows.append([
                            aln['chrom_t'], aln['start_t'], aln['end_t'], aln['chrom_q'], aln['start_q'], aln['end_q'], aln['pct_identity_gap'], aln['pct_identity_ungap'], aln['num_ident'], aln['alignment_size']
                        ])

            # construct new gff dataframe of alignments
            gff_df = pd.DataFrame(gff_rows, columns=['chrom_t', 'start_t', 'end_t', 'chrom_q', 'start_q', 'end_q', 'pct_identity_gap', 'pct_identity_ungap', 'num_ident', 'alignment_size'])

    ################################################################
    # define model sequences
    ################################################################

    seq_df_q = None
    if options.query_bed_file is None :
        chroms = []
        starts = []
        ends = []

        # loop over contigs and create dataframe of options.strided model sequences
        for ctg in contigs:
            seq_start = int(np.ceil(ctg.start / options.snap) * options.snap) + options.crop_bp
            seq_end = seq_start + seq_tlength

            while seq_end < ctg.end - options.crop_bp:
                # record sequence
                chroms.append(ctg.chr)
                starts.append(seq_start)
                ends.append(seq_end)

                # update
                seq_start += options.stride
                seq_end += options.stride

        seq_df_q = pd.DataFrame({'chrom' : chroms, 'start' : starts, 'end' : ends})
    else :
        # load sequence bed file for query species
        seq_df_q = pd.read_csv(options.query_bed_file, names=['chrom', 'start', 'end', 'label1'], sep='\t')

        # (optionally) load chrom.alias file
        if options.query_chrom_alias_file is not None :
            query_dict = {}
            with open(options.query_chrom_alias_file, 'rt') as f :
                for line in f :
                    line_parts = line.strip().split('\t')
                    query_dict[line_parts[0]] = line_parts[1]

            # translate query chromosomes (if needed)
            if seq_df_q.iloc[0]['chrom'] in query_dict :
                chroms = []
                for _, row in seq_df_q.iterrows() :
                    chroms.append(query_dict[row['chrom']])

                seq_df_q['chrom_orig'] = seq_df_q['chrom']
                seq_df_q['chrom'] = chroms

        # keep only chrom, start, end, chrom_orig, label1 fields
        seq_df_q = seq_df_q[['chrom', 'start', 'end', 'chrom_orig', 'label1']].copy()

    # remove crop
    seq_df_q['start'] -= options.crop_bp
    seq_df_q['end'] += options.crop_bp

    # find homologous sequence pairs

    # create dataframe row identifiers
    seq_df['target_id'] = seq_df['chrom'] + '_' + seq_df['start'].astype(str) + '_' + seq_df['end'].astype(str)
    seq_df_q['query_id'] = seq_df_q['chrom'] + '_' + seq_df_q['start'].astype(str) + '_' + seq_df_q['end'].astype(str)

    gff_df['align_id'] = gff_df['chrom_t'] + '_' + gff_df['start_t'].astype(str) + '_' + gff_df['end_t'].astype(str) + '_' + gff_df['chrom_q'] + '_' + gff_df['start_q'].astype(str) + '_' + gff_df['end_q'].astype(str)

    # create pyranges objects for query/target sequence dataframes
    seq_pr_t = pr.PyRanges(seq_df.rename(columns={'chrom' : 'Chromosome', 'start' : 'Start', 'end' : 'End'}))
    seq_pr_q = pr.PyRanges(seq_df_q.rename(columns={'chrom' : 'Chromosome', 'start' : 'Start', 'end' : 'End'}))

    # create pyranges object views for alignment gff
    gff_pr_t = pr.PyRanges(gff_df.rename(columns={'chrom_t' : 'Chromosome', 'start_t' : 'Start', 'end_t' : 'End'}))
    gff_pr_q = pr.PyRanges(gff_df.rename(columns={'chrom_q' : 'Chromosome', 'start_q' : 'Start', 'end_q' : 'End'}))

    # join pyranges intervals
    seq_gff_df_t = seq_pr_t.join(gff_pr_t, strandedness=False).df
    seq_gff_df_q = seq_pr_q.join(gff_pr_q, strandedness=False).df

    # calculate overlaps

    def _overlap(row) :
        return max(0, min(row['End'], row['End_b']) - max(row['Start'], row['Start_b']))

    # ..for target
    seq_gff_df_t['overlap'] = seq_gff_df_t.apply(_overlap, axis=1)

    # ..for query
    seq_gff_df_q['overlap'] = seq_gff_df_q.apply(_overlap, axis=1)

    # filter on minimum alignment size and overlaps
    seq_gff_df_t = seq_gff_df_t.loc[seq_gff_df_t['overlap'] >= options.t_olap_min].copy().reset_index(drop=True)
    seq_gff_df_q = seq_gff_df_q.loc[seq_gff_df_q['overlap'] >= options.q_olap_min].copy().reset_index(drop=True)

    # clean up target dataframe
    seq_gff_df_t = seq_gff_df_t.rename(columns={
        'Chromosome' : 'chrom', 'Start' : 'start', 'End' : 'end', 'Start_b' : 'start_t', 'End_b' : 'end_t', 'overlap' : 'overlap_t',
    })

    seq_gff_df_t = seq_gff_df_t[[
        'chrom', 'start', 'end', 'label', 'target_id', 'pct_identity_gap', 'pct_identity_ungap', 'num_ident', 'alignment_size', 'align_id', 'overlap_t',
    ]].copy().reset_index(drop=True)

    # clean up query dataframe
    seq_gff_df_q = seq_gff_df_q.rename(columns={
        'Chromosome' : 'chrom', 'Start' : 'start', 'End' : 'end', 'Start_b' : 'start_q', 'End_b' : 'end_q', 'overlap' : 'overlap_q',
    })

    seq_gff_df_q = seq_gff_df_q[[
        'chrom', 'start', 'end', 'query_id', 'align_id', 'overlap_q',
    ]].copy().reset_index(drop=True)

    # join query and target dataframes on alignment id
    seq_gff_df = seq_gff_df_q.join(seq_gff_df_t.set_index('align_id'), on='align_id', rsuffix='_t', how='inner').copy().reset_index(drop=True)

    # calculate shared (min) overlap per matched alignment
    seq_gff_df['min_overlap'] = np.minimum(seq_gff_df['overlap_q'].values, seq_gff_df['overlap_t'].values)

    # aggregate total shared overlap between query and target windows
    seq_gff_df_agg = seq_gff_df.groupby(['query_id', 'target_id']).agg({
        'label' : 'first',
        'pct_identity_gap' : 'first',
        'pct_identity_ungap' : 'first',
        'num_ident' : 'first',
        'alignment_size' : 'first',
        'min_overlap' : 'sum',
    }).copy().reset_index()

    # filter on minimum shared overlap through aggregated alignments
    seq_gff_df_agg = seq_gff_df_agg.loc[seq_gff_df_agg['min_overlap'] >= options.shared_olap_min].copy().reset_index(drop=True)

    # create a lookup dictionary for query rows
    overlap_dict = {}

    # loop over rows in aggregated dataframe
    for _, row in seq_gff_df_agg.iterrows() :
        if row['query_id'] not in overlap_dict :
            overlap_dict[row['query_id']] = []

        overlap_dict[row['query_id']].append(row['label'])

    # loop over query_ids
    for query_id in overlap_dict :

        # get unique, sorted overlapping labels
        overlap_dict[query_id] = sorted(list(set(overlap_dict[query_id])))

    # augment original sequence bed dataframe with overlapping labels

    labels = []

    # loop over rows in sequence dataframe
    for _, row in seq_df_q.iterrows() :
        if row['query_id'] in overlap_dict :
            labels.append(",".join(overlap_dict[row['query_id']]))
        else :
            labels.append('free')

    seq_df_q['label'] = labels

    # drop row ids
    seq_df_q = seq_df_q.drop(columns=['query_id']).copy().reset_index(drop=True)
    
    # potentially merge with existing labels (if starting from a pre-existing bed file)
    if 'label1' in seq_df_q.columns.values.tolist() :
        labels = []

        # loop over rows in sequence dataframe
        for _, row in seq_df_q.iterrows() :
            
            # start with pre-existing label
            label_merged = [row['label1']]
            
            # augment with new overlapping labels if not 'free'
            if row['label'] != 'free' :
                label_merged += row['label'].split(",")
            
            # get sorted, unique labels
            label_merged = sorted(list(set(label_merged)))
            
            labels.append(",".join(label_merged))
        
        # over-write labels
        seq_df_q['label'] = labels
    
    # potentially convert chromosomes back to alias (if starting from a pre-existing bed file)
    if 'chrom_orig' in seq_df_q.columns.values.tolist() :
        seq_df_q['chrom'] = seq_df_q['chrom_orig']
    
    # optionally drop sequence windows that match multiple unique labels
    if not options.keep_multi_olap :
        seq_df_q = seq_df_q.loc[~seq_df_q['label'].str.contains(",")].copy().reset_index(drop=True)
    
    # optionally drop sequence windows without any target overlap
    if not options.keep_free :
        seq_df_q = seq_df_q.loc[seq_df_q['label'] != 'free'].copy().reset_index(drop=True)

    # plot number of overlapping sequence rows per label

    import matplotlib.pyplot as plt

    # get number of overlapping rows per label
    unique_labels, label_counts = np.unique(seq_df_q['label'].values, return_counts=True)

    # sort according to count (descending)
    sort_index = np.argsort(label_counts)[::-1]
    unique_labels = unique_labels[sort_index][:32]
    label_counts = label_counts[sort_index][:32]

    f = plt.figure(figsize=(8, 4))

    # plot as bars
    plt.bar(np.arange(unique_labels.shape[0]), label_counts, edgecolor='black', color='deepskyblue', linewidth=1)

    plt.xticks(np.arange(unique_labels.shape[0]), [l for l in unique_labels.tolist()], fontsize=10, rotation=90)
    plt.yticks(fontsize=10)

    plt.ylabel('# of sequences', fontsize=10)

    plt.title('reference vs ' + options.genome_label, fontsize=10)

    plt.tight_layout()

    # save
    plt.savefig("%s/overlap.png" % genome_out_dir, dpi=300)
    plt.close()

    # re-plot with 'free' column

    # get number of overlapping rows per label
    unique_labels, label_counts = np.unique(seq_df_q.query("label != 'free'")['label'].values, return_counts=True)

    # sort according to count (descending)
    sort_index = np.argsort(label_counts)[::-1]
    unique_labels = unique_labels[sort_index][:32]
    label_counts = label_counts[sort_index][:32]

    f = plt.figure(figsize=(8, 4))

    # plot as bars (no 'free' column)
    plt.bar(np.arange(unique_labels.shape[0]), label_counts, edgecolor='black', color='deepskyblue', linewidth=1)

    plt.xticks(np.arange(unique_labels.shape[0]), [l for l in unique_labels.tolist()], fontsize=10, rotation=90)
    plt.yticks(fontsize=10)

    plt.ylabel('# of sequences', fontsize=10)

    plt.title('reference vs ' + options.genome_label + ' (without free rows)', fontsize=10)

    plt.tight_layout()

    # save
    plt.savefig("%s/overlap_nonfree.png" % genome_out_dir, dpi=300)
    plt.close()
    
    if options.query_bed_file is None :
        
        # shuffle
        shuffle_index = np.arange(len(seq_df_q), dtype='int32')
        np.random.shuffle(shuffle_index)
        seq_df_q['shuffle_index'] = shuffle_index
        
        # determine if there are fold labels in the bed file
        has_fold_labels = False
        for label in seq_df_q['label'].unique().tolist() :
            if 'fold' in label :
                has_fold_labels = True
                break
        
        num_labels = len(seq_df_q['label'].unique().tolist())
        
        # get label fold index
        if has_fold_labels and not options.keep_multi_olap :
            seq_df_q['label_sort'] = seq_df_q['label'].apply(lambda x: int(x.replace('fold', '')) if 'fold' in x else num_labels)
        else :
            seq_df_q['label_sort'] = seq_df_q['label']
        
        seq_df_q['label_sort'] = seq_df_q['label_sort'].astype(int)
        
        seq_df_q = seq_df_q.sort_values(by=['label_sort', 'shuffle_index'], ascending=True).copy().reset_index(drop=True)
        seq_df_q = seq_df_q.drop(columns=['label_sort', 'shuffle_index']).copy().reset_index(drop=True)

        # down-sample
        if options.sample_pct < 1.0:
            seq_df_q = seq_df_q.iloc[:int(options.sample_pct * len(seq_df_q))].copy().reset_index(drop=True)
    
    # re-crop
    seq_df_q['start'] += options.crop_bp
    seq_df_q['end'] -= options.crop_bp
    
    # create model sequence objects
    mseqs = []
    
    # loop over query sequence dataframe
    for _, row in seq_df_q.iterrows() :
        mseqs.append(data.ModelSeq(0, row['chrom'], row['start'], row['end'], row['label']))
    
    ################################################################
    # filter for sufficient mappability
    ################################################################
    if options.umap_bed is not None:
        # create padded model seqs
        mseqs_pad = []
        for mseq in mseqs :
            mseqs_pad.append(data.ModelSeq(
                mseq.genome, mseq.chr, mseq.start - options.pad_bp, mseq.end + options.pad_bp, mseq.label
            ))

        # annotate unmappable positions
        mseqs_unmap = data.annotate_unmap(
            mseqs_pad, options.umap_bed, seq_tlength + 2 * options.pad_bp, options.pool_width
        )

        # filter unmappable
        if options.pad_bp > 0 :
            mseqs_map_mask = mseqs_unmap[:, options.pad_bp // options.pool_width:-options.pad_bp // options.pool_width].mean(axis=1, dtype="float64") < options.umap_t
        else :
            mseqs_map_mask = mseqs_unmap.mean(axis=1, dtype="float64") < options.umap_t
        
        mseqs = [
            mseqs[si] for si in range(len(mseqs)) if mseqs_map_mask[si]
        ]
        mseqs_unmap = mseqs_unmap[mseqs_map_mask, :]

        # check if mseqs_unmap.npy file exists
        if os.path.isfile("%s/mseqs_unmap.npy" % genome_out_dir):

            # rename existing file
            os.rename("%s/mseqs_unmap.npy" % genome_out_dir, "%s/mseqs_unmap_orig.npy" % genome_out_dir)
        
        # write to file
        unmap_npy_file = "%s/mseqs_unmap.npy" % genome_out_dir
        np.save(unmap_npy_file, mseqs_unmap)
    
    # check if sequences.bed file exists
    if os.path.isfile("%s/sequences.bed" % genome_out_dir):
        
        # rename existing file
        os.rename("%s/sequences.bed" % genome_out_dir, "%s/sequences_orig.bed" % genome_out_dir)

    # write sequences to BED
    seqs_bed_file = "%s/sequences.bed" % genome_out_dir
    data.write_seqs_bed(seqs_bed_file, mseqs, True)

# break up large alignment into sub-alignments of max size (same strand orientation)
def _break_large_alignment_sense(row, break_align_size=131072) :

    gap = row['gap_str'].split(' ')

    # create fifo queue of alignment symbols
    gap_deque = deque(gap)

    alignments = []

    # counter for remaining aligned based
    align_remainder = row['end_t'] - row['start_t'] + 1

    # initial sub-alignment start positions
    start_t = row['start_t']
    start_q = row['start_q']

    while len(gap_deque) > 0 :

        # initialize with empty sub-alignment
        end_t = start_t
        end_q = start_q

        # list to record symbols of sub-alignment
        new_gap = []

        # loop over alignment symbols until we reach the target break size
        while len(gap_deque) > 0 and (end_t - start_t <= break_align_size or (align_remainder - (end_t - start_t)) <= break_align_size // 4) :
            align_symbol = gap_deque.popleft()

            # increment end positions depending on alignment symbol
            if align_symbol[0] == 'M' :
                end_t += int(align_symbol[1:])
                end_q += int(align_symbol[1:])
            elif align_symbol[0] == 'D' :
                end_t += int(align_symbol[1:])
            elif align_symbol[0] == 'I' :
                end_q += int(align_symbol[1:])

            new_gap.append(align_symbol)

        # re-calculate statistics for sub-alignment
        n_matches = 0
        n_mismatches = 0

        # loop over alignment symbols
        for align_symbol in new_gap :

            # increment match/mismatch counters
            if align_symbol[0] == 'M' :
                n_matches += int(align_symbol[1:])
            elif align_symbol[0] == 'D' or align_symbol[0] == 'I' :
                n_mismatches += int(align_symbol[1:])

        # re-calculate alignment size and number of identical matches (approximate)
        alignment_size = n_matches + n_mismatches
        num_ident = int((row['pct_identity_ungap'] / 100.) * n_matches)

        # create new alignment row
        alignment = {
            'chrom_t' : row['chrom_t'],
            'start_t' : start_t,
            'end_t' : end_t - 1,
            'chrom_q' : row['chrom_q'],
            'start_q' : start_q,
            'end_q' : end_q - 1,
            'strand' : row['strand'],
            'gap_str' : " ".join(new_gap),
            'pct_identity_gap' : row['pct_identity_gap'],
            'pct_identity_ungap' : row['pct_identity_ungap'],
            'num_ident' : num_ident,
            'alignment_size' : alignment_size,
            'align_id' : row['chrom_t'] + '_' + str(start_t) + '_' + str(end_t - 1) + '_' + row['chrom_q'] + '_' + str(start_q) + '_' + str(end_q - 1),
        }

        # append alignment row
        alignments.append(alignment)

        # calculate new remainder
        align_remainder -= (end_t - start_t)

        # move to new start of next sub-alignment
        start_t = end_t
        start_q = end_q
    
    return alignments

# break up large alignment into sub-alignments of max size (different strand orientation)
def _break_large_alignment_antisense(row, break_align_size=131072) :

    gap = row['gap_str'].split(' ')

    # create fifo queue of alignment symbols
    gap_deque = deque(gap)

    alignments = []

    # counter for remaining aligned based
    align_remainder = row['end_t'] - row['start_t'] + 1

    # initial sub-alignment start positions
    start_t = row['start_t']
    end_q = row['end_q']

    while len(gap_deque) > 0 :

        # initialize with empty sub-alignment
        end_t = start_t
        start_q = end_q

        # list to record symbols of sub-alignment
        new_gap = []

        # loop over alignment symbols until we reach the target break size
        while len(gap_deque) > 0 and (end_t - start_t <= break_align_size or (align_remainder - (end_t - start_t)) <= break_align_size // 4) :
            align_symbol = gap_deque.popleft()

            # increment end positions depending on alignment symbol
            if align_symbol[0] == 'M' :
                end_t += int(align_symbol[1:])
                start_q -= int(align_symbol[1:])
            elif align_symbol[0] == 'D' :
                end_t += int(align_symbol[1:])
            elif align_symbol[0] == 'I' :
                start_q -= int(align_symbol[1:])

            new_gap.append(align_symbol)

        # re-calculate statistics for sub-alignment
        n_matches = 0
        n_mismatches = 0

        # loop over alignment symbols
        for align_symbol in new_gap :

            # increment match/mismatch counters
            if align_symbol[0] == 'M' :
                n_matches += int(align_symbol[1:])
            elif align_symbol[0] == 'D' or align_symbol[0] == 'I' :
                n_mismatches += int(align_symbol[1:])

        # re-calculate alignment size and number of identical matches (approximate)
        alignment_size = n_matches + n_mismatches
        num_ident = int((row['pct_identity_ungap'] / 100.) * n_matches)

        # create new alignment row
        alignment = {
            'chrom_t' : row['chrom_t'],
            'start_t' : start_t,
            'end_t' : end_t - 1,
            'chrom_q' : row['chrom_q'],
            'start_q' : start_q + 1,
            'end_q' : end_q,
            'strand' : row['strand'],
            'gap_str' : " ".join(new_gap),
            'pct_identity_gap' : row['pct_identity_gap'],
            'pct_identity_ungap' : row['pct_identity_ungap'],
            'num_ident' : num_ident,
            'alignment_size' : alignment_size,
            'align_id' : row['chrom_t'] + '_' + str(start_t) + '_' + str(end_t - 1) + '_' + row['chrom_q'] + '_' + str(start_q + 1) + '_' + str(end_q),
        }

        # append alignment row
        alignments.append(alignment)

        # calculate new remainder
        align_remainder -= (end_t - start_t)

        # move to new start of next sub-alignment
        start_t = end_t
        end_q = start_q
    
    return alignments


################################################################################
if __name__ == "__main__":
    main()
