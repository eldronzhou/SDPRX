## SDPRX
SDPRX is a statistical method for cross-population prediction of complex traits. It integrates GWAS summary statistics and LD matrices from two populations (EUR and non-EUR) to compuate polygenic risk scores.

## Installation

You can download SDPRX by simply running

```
git clone https://github.com/eldronzhou/SDPR.git
```

SDPRX is developed under python 2.7 but should be compatible with python 3. 

## Input 

### Reference LD

The reference LD matrices can be downloaded from the following link. 

| Populations | Number of SNPs | Size | Link
| --- | --- | --- | --- |
| EUR_EAS | 873,166 | 6.3G | [link](https://drive.google.com/file/d/1MGt-Ai5foThXBF1xAZMKksBRqZGsbQ1l/view?usp=sharing) |
| EUR_AFR | 903,499 | 8.7G | [link](https://drive.google.com/file/d/1cbcfCicsuARfcv231tY98PTnAoOoQS8O/view?usp=sharing) |

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
python SDPRX.py --ss1 test/EUR.txt --ss2 test/EAS.txt --N1 40000 --N2 40000 --load_ld test/ --valid test/test.bim --chr 1 --rho 0.8 --out test/res_1
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
If a validation dataset is available, one can further learn a linear combination of PRS based on the best performance in the validation dataset. 
