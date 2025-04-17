## SDPRX
SDPRX is a statistical method for cross-population prediction of complex traits. It integrates GWAS summary statistics and LD matrices from two populations (EUR and non-EUR) to compuate polygenic risk scores.

## Installation

You can download SDPRX by simply running

```
git clone https://github.com/eldronzhou/SDPRX.git
```

SDPRX is developed under python 2.7 but should be compatible with python 3. We recommend you to run SDPRX in the [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) so that libraries like numpy and scipy would be pre-installed. If running in the Anaconda environment, the only requirement to run SDPRX would be installing [joblib](https://joblib.readthedocs.io/en/latest/installing.html).

## Input 

### Reference LD

The reference LD matrices based on 1000 Genome Hapmap3 SNPs can be downloaded from the following link. Downgrade to pandas <= 2.0 if there is error opening those files.

| Populations | Number of SNPs | Size | Link
| --- | --- | --- | --- |
| EUR_EAS | 873,166 | 6.3G | [link](https://app.box.com/s/ck9t8fh8x3hqyhl7blvih16q4vcsnj9v) |
| EUR_AFR | 903,499 | 8.7G | [link](https://app.box.com/s/ibw4lbjxkrgopdacgfh9zcmyr4ibo6ru) |
| EUR_SAS |  | 19G | [link](https://app.box.com/s/fvxjap1wfeo7db8w2ty35ecrmo7ozrx4) |
| EUR_AMR |  | 14G | [link](https://app.box.com/s/atoqnpdm970cv740bwz4uae2yzy8zfxf) |

You can use the following command if you wish to use your LD panel to estimate LD matrix. 

```
python calc_ref.py --ref_path1 path_to_plink1_prefix_pop1 --ref_path2 path_to_plink1_prefix_pop2 --chrom chr_number --threads 3 --out ./
```

### Summary Statistics 

The EUR/nonEUR summary statistics should have at least following columns with the same name, where SNP is the marker name, A1 is the effect allele, A2 is the alternative allele, Z is the Z score for the association statistics, and N is the sample size. 

```
SNP     A1      A2      Z       N
rs737657        A       G       -2.044      252156
rs7086391       T       C       -2.257      248425
rs1983865       T       C       3.652    253135
...
```

## Running SDPRX

An example command to run the test data is 

```
python SDPRX.py --ss1 test/EUR.txt --ss2 test/EAS.txt --N1 40000 --N2 40000 --force_shared TRUE --load_ld test/ --valid test/test.bim --chr 1 --rho 0.8 --out test/res_1
```

A full list of options can be obtained by running `python SDPRX.py -h`. Below are the required options.

- ss1 (required): Path to the EUR summary statistics.
- ss2 (required): Path to the non-EUR summary statistics.
- N1 (required): Sample size of the EUR summary statistics.
- N2 (required): Sample size of the non-EUR summary statistics.
- load_ld (required): Path to the referecence LD directory.
- valid (required): Path to the bim file for the testing dataset, including the .bim suffix.
- chr (required): Chromosome.
- out (required): Path to the output file containing estimated effect sizes.
- rho (required): Trans-ethnic genetic correlation output by PopCorn between 0 and 1. Default is 0.8. 
- force_shared (required): Whether to force sharing of effect sizes between populations. Default is True.
- n_threads (optional): number of threads to use. Default is 1.

For real data analysis, it is recommended to run each SDPRX on each chromosome in parallel, and using 3 threads for each chromsome.  

## Output 

There are two output files corresponding to the adjusted effect sizes for EUR (e.g. res_22_1.txt) and non-EUR population (e.g. res_22_2.txt).
One can use [PLINK](https://www.cog-genomics.org/plink/1.9/score) to derive the PRS.

```
plink --bfile test_geno --score res_22_1.txt 1 2 3 header --out test_1 # EUR
plink --bfile test_geno --score res_22_2.txt 1 2 3 header --out test_2 # non-EUR
```
If a validation dataset is available, one can further learn a linear combination of PRS (a * PRS_1 + (1-a) * PRS_2 for a grid of a ranging from 0 to 1 by a step of 0.5) based on the best performance in the validation dataset. 

## Weights

The weights for real traits analyzed in our paper can be downloaded [here](https://app.box.com/s/9auedn4wzx563pbtplq3h106ybn3h3dq). The summary information in the format of PGS catalog can be found [here](https://app.box.com/s/1ky2bpkblg2jnv0v9le9r4tkej4sud87).

## Citation

Zhou G, Chen T, Zhao H. SDPRX: A statistical method for cross-population prediction of complex traits. Am J Hum Genet. 2023 Jan 5;110(1):13-22. doi: 10.1016/j.ajhg.2022.11.007. 
