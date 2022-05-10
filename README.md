## SDPRX
SDPRX is a statistical method for cross-population prediction of complex traits. It integrates GWAS summary statistics and LD matrices from two populations to compuate polygenic risk scores.

## Installation

You can download SDPRX by simply running

```
git clone https://github.com/eldronzhou/SDPR.git
```

SDPRX is developed under python 2.7 but should be compatible with python 3. 

## Input 

### Reference LD

The reference LD matrices can be downloaded from the following link. 

### Summary Statistics 

The summary statistics should at least contain following columns with the same name, where SNP is the marker name, A1 is the effect allele, A2 is the alternative allele, Z is the Z score for the association statistics (can be calculated as Z = EFFECT / SE), and N is the sample size. 

```
SNP     A1      A2      Z       N
rs737657        A       G       -2.044      252156
rs7086391       T       C       -2.257      248425
rs1983865       T       C       3.652    253135
...
```

## Running SDPRX

```

```

The output has format:

```
SNP     A1      beta
rs12255619      C       -0.000124535
rs7909677       G       -0.000106013
rs10904494      C       -0.000178207
...
```

where SNP is the marker name, A1 is the effect allele, beta is the estimated posterior effect sizes.

Once having the ouput, one can use [PLINK](https://www.cog-genomics.org/plink/1.9/score) to derive the PRS.

```
plink --bfile test_geno --score res_1.txt 1 2 3 header --out test_1
plink --bfile test_geno --score res_2.txt 1 2 3 header --out test_2
```

## Help



