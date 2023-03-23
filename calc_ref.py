#!/usr/bin/env python
from plinkio import plinkfile
import numpy as np, pandas as pd
import gzip, h5py
try:
    import pickle as pickle
except:
    import cPickle as pickle
from joblib import Parallel, delayed
import argparse

## modified from https://github.com/bvilhjal/ldpred/blob/master/ldpred/plinkfiles.py
def parse_plink_snps(genotype_file, snp_indices):
    plinkf = plinkfile.PlinkFile(genotype_file)
    samples = plinkf.get_samples()
    num_individs = len(samples)
    num_snps = len(snp_indices)
    raw_snps = np.empty((num_snps, num_individs), dtype='int8')
    # If these indices are not in order then we place them in the right place while parsing SNPs.
    snp_order = np.argsort(snp_indices)
    ordered_snp_indices = list(snp_indices[snp_order])
    ordered_snp_indices.reverse()
   # print 'Iterating over file to load SNPs'
    snp_i = 0
    next_i = ordered_snp_indices.pop()
    line_i = 0
    max_i = ordered_snp_indices[0]
    while line_i <= max_i:
        if line_i < next_i:
            plinkf.next()
        elif line_i == next_i:
            line = plinkf.next()
            snp = np.array(line, dtype='int8')
            bin_counts = line.allele_counts()
            if bin_counts[-1] > 0:
                mode_v = np.argmax(bin_counts[:2])
                snp[snp == 3] = mode_v
            s_i = snp_order[snp_i]
            raw_snps[s_i] = snp
            if line_i < max_i:
                next_i = ordered_snp_indices.pop()
            snp_i += 1
        line_i += 1
    plinkf.close()
    assert snp_i == len(raw_snps), 'Failed to parse SNPs?'
    num_indivs = len(raw_snps[0])
    freqs = np.sum(raw_snps, 1, dtype='float32') / (2 * float(num_indivs))
    return raw_snps, freqs


def find_ld(i, snps1, snps2, n_snp, n_ind1, n_ind2, cut):
    left = max(i-300,0)
    right = min(i+300,n_snp)
    snps_std1 = snps1[left:right]
    snps_std2 = snps2[left:right]
    if left > 0:
        cor = np.maximum(np.dot(snps_std1, snps_std1[300])/n_ind1,
                        np.dot(snps_std2, snps_std2[300])/n_ind2)
        return np.where(cor**2 > cut) + np.array(i-300)
    else:
        cor = np.maximum(np.dot(snps_std1, snps_std1[i])/n_ind1,
                        np.dot(snps_std2, snps_std2[i]/n_ind2))
        return np.where(cor**2 > cut) 

def calc_ref_ld(i, snps, ld_boundaries):
    snps_std = snps[ld_boundaries[i][0]:ld_boundaries[i][1],:]
    cor = np.corrcoef(snps_std)
    n = np.shape(snps_std)[1]
    x2 = snps_std**2
    var = np.dot(x2, x2.T)*n/(n-1)**3 - np.dot(snps_std, snps_std.T)**2 / (n-1)**3
    sr = min(1, max(0, 1-(np.sum(var) - np.sum(np.diag(var))) / (np.sum(cor**2) - np.sum(np.diag(cor**2)))))
    return sr*cor + (1-sr)*np.identity(cor.shape[0])

def calc_ref(ref_path1, ref_path2, out, n_threads, cutoff, chrom):
    bim_file1 = ref_path1 + ".bim"
    bim1 = pd.read_table(bim_file1, header=None)
    bim1.columns = ['Chr', 'SNP_id', 'cM', 'Pos', 'A1', 'A2']

    bim_file2 = ref_path2 + ".bim"
    bim2 = pd.read_table(bim_file2, header=None)
    bim2.columns = ['Chr', 'SNP_id', 'cM', 'Pos', 'A1', 'A2']

    boundary = []; ld_boundaries = []
    
    left1 = bim1[bim1['Chr'] == chrom].index[0]
    right1 = bim1[bim1['Chr'] == chrom].index[-1]+1
    
    left2 = bim2[bim2['Chr'] == chrom].index[0]
    right2 = bim2[bim2['Chr'] == chrom].index[-1]+1
    
    common_list = set(bim1[left1:right1]['SNP_id']) & set(bim2[left2:right2]['SNP_id'])
    idx1 = bim1[bim1.SNP_id.isin(list(common_list))].index
    idx2 = bim2[bim2.SNP_id.isin(list(common_list))].index
    left = 0; right = len(common_list)
    
    snps1, freqs1 = parse_plink_snps(ref_path1, np.array( idx1 ))
    snps2, freqs2 = parse_plink_snps(ref_path2, np.array( idx2 ))
    
    # align alleles
    merge = pd.merge(bim1.iloc[idx1], bim2.iloc[idx2], on="SNP_id")
    idx_matched = np.logical_and(merge['A1_x'] == merge['A1_y'], 
               merge['A2_x'] == merge['A2_y'])
    idx_flip = np.logical_and(merge['A1_x'] == merge['A2_y'], 
               merge['A2_x'] == merge['A1_y'])
    #assert np.alltrue(np.logical_or(idx_matched, idx_flip))
    snps2[idx_flip,] = 2 - snps2[idx_flip,]
    
    n_rows1 = np.shape(snps1)[0]; n_cols1 = np.shape(snps1)[1]
    snps_std1 = (snps1 - np.mean(snps1, axis=1).reshape((n_rows1,1))) / np.std(snps1, axis=1).reshape((n_rows1,1))
    n_rows2 = np.shape(snps2)[0]; n_cols2 = np.shape(snps2)[1]
    snps_std2 = (snps2 - np.mean(snps2, axis=1).reshape((n_rows2,1))) / np.std(snps2, axis=1).reshape((n_rows2,1))
    
    snp_list = [(find_ld(i, snps1=snps_std1, snps2=snps_std2, n_snp=n_rows1,
                     n_ind1=n_cols1, n_ind2=n_cols2, cut=cutoff) + np.array(left)) for i in range(n_rows1)]
    
    max_list = [np.max(i) for i in snp_list]
    cummax_list = []
    cummax_list.append(max_list[0])
    for i in range(1,len(max_list)):
        if max_list[i] > cummax_list[i-1]:
            cummax_list.append(max_list[i])
        else:
            cummax_list.append(cummax_list[i-1])
    idx = np.where( cummax_list - np.array(range(left, right)) == 0) + np.array(1) + np.array(left)
    idx = idx[0]
    boundary.append([0+left, idx[0]])
    for i in range(np.size(idx)-1):
        boundary.append([idx[i], idx[i+1]])   
    #assert boundary[-1][1] == right, 'something wrong with chrom ' + str(chrom)
    size = [i[1] - i[0] for i in boundary]

    skip_start = False
    for i in range(0, len(boundary)):
        if skip_start is False:
            start_i = boundary[i][0]
        stop_i = boundary[i][1]
        if stop_i - start_i < 300 and i != len(boundary)-1:
            skip_start = True
            continue
        else:
            ld_boundaries.append([start_i, stop_i])
            skip_start = False
    
    ref_ld_mat1 = Parallel(n_jobs=n_threads)(delayed(calc_ref_ld)(i, snps_std1, ld_boundaries)
                                        for i in range(len(ld_boundaries)))   
    ref_ld_mat2 = Parallel(n_jobs=n_threads)(delayed(calc_ref_ld)(i, snps_std2, ld_boundaries)
                                        for i in range(len(ld_boundaries)))   
 
    f = gzip.open(out+"/chr_"+str(chrom)+".gz", 'wb')
    pickle.dump([bim1.iloc[idx1]['SNP_id'], bim1.iloc[idx1]['A1'], bim1.iloc[idx1]['A2'],
                  ld_boundaries, ref_ld_mat1, ref_ld_mat2], f, protocol=2)
    f.close()


parser = argparse.ArgumentParser(prog='calc_ref',
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                description="Version 0.0.1 Test Only")

parser.add_argument('--threads', type=int, default=1, 
                        help='Number of Threads used.')

parser.add_argument('--chrom', type=int, default=1, required=True, 
                        help='Chromosome number.')

parser.add_argument('--cutoff', type=float, default=.1, 
                        help='R2 cutoff to define approximately independent blocks.')

parser.add_argument('--ref_path1', type=str, required=True,
                        help='Path to prefix of plink files for pouplation 1 (typically EUR).')

parser.add_argument('--ref_path2', type=str, required=True,
                        help='Path to prefix of plink files for population 2 (typically non-EUR).')

parser.add_argument('--out', type=str, required=True,
                        help='Path to the output directory containing paritioned reference LD matrix.')

def main():
    args = parser.parse_args()
    calc_ref(args.ref_path1, args.ref_path2, args.out, args.threads, args.cutoff, args.chrom)

if __name__ == '__main__':
    main()

