#!/usr/bin/env python

from plinkio import plinkfile
from bisect import bisect_left
from joblib import Parallel, delayed
import gzip, cPickle

def parse_plink_snps(genotype_file, snp_indices):
	# from author of LDpred
    plinkf = plinkfile.PlinkFile(genotype_file)
    samples = plinkf.get_samples()
    num_individs = len(samples)
    num_snps = len(snp_indices)
    raw_snps = sp.empty((num_snps, num_individs), dtype='int8')
    # If these indices are not in order then we place them in the right place while parsing SNPs.
    snp_order = sp.argsort(snp_indices)
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
            snp = sp.array(line, dtype='int8')
            bin_counts = line.allele_counts()
            if bin_counts[-1] > 0:
                mode_v = sp.argmax(bin_counts[:2])
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
    freqs = sp.sum(raw_snps, 1, dtype='float32') / (2 * float(num_indivs))
    return raw_snps, freqs

def parse_ld_boundaries(min_block_wd, ss, block_path):
	ld_boundaries = []; add_on = 0; 
	for chrom in range(1,23):
	    block_path = block_path + str(chrom)+ ".bed"
	    pos = ss[ss.CHR == chrom]['BP']
	    with open(block_path) as f:
	        f.next()
	        skip_start = False
	        if chrom >= 2:
	            add_on = ld_boundaries[-1][1]
	        for line in f:
	            l = (line.strip()).split()
	            if skip_start is False:
	                start_i = bisect_left(np.array(pos), int(l[1]))
	            stop_i = bisect_left(np.array(pos), int(l[2]))
	            # block size at least including how many SNPs
	            if stop_i - start_i < min_block_wd: 
	                if stop_i == len(pos) + 1 or stop_i == len(pos): 
	                    # merge the last block to the previous one
	                    ld_boundaries[-1][1] = add_on + stop_i
	                    assert ld_boundaries[-1][1] == pos.index[-1]+1 , \
	                    'something wrong with chrom ' + str(chrom)
	                    break
	                else:
	                    skip_start = True
	                    continue
	            skip_start = False
	            ld_boundaries.append([add_on+start_i, add_on+stop_i])
	return ld_boundaries

def calc_ref_ld(i, n_jobs, ref_path, ld_boundaries):
    snp_indices = np.array( range(ld_boundaries[i][0], ld_boundaries[i][1]) )
    snps, freqs = plinkfiles.parse_plink_snps(genotype_file=ref_path, snp_indices=snp_indices)
    return np.corrcoef(snps)
