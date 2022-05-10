## SDPRX
A statistical method for cross-population prediction of complex traits.

## Installation


```
git clone https://github.com/eldronzhou/SDPR.git
```


## Quick start

SDPR can be run from the command line. To see the full list of options, please type

```bash
./SDPR -h
```

## Input 

### Reference LD


```

```

### Summary Statistics 

The summary statistics should at least contain following columns with the same name (order of the column is not important).

```
SNP	A1	A2	BETA	P
rs737657        A       G       -2.044  0.0409
rs7086391       T       C       -2.257  0.024
rs1983865       T       C       3.652   0.00026
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
plink --bfile test_geno --score res.txt 1 2 3 header --out test
```

## Help



