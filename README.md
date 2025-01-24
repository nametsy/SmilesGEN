# SmilesGEN：
## Introduction
We propose SmilesGEN, a model that generates drugs based on gene expression profiles. You use this model to generate and optimize drug-like molecules using the required expression profile. The following is a detailed introduction to the model:
## Model Architecture
![](https://github.com/nametsy/SmilesGEN/blob/main/framework.png)


## Environment Installation
The required packages are listed in the requirements.txt. 
You execute the following command to install the packages:

```
$ pip install -r requirements.txt
```

## File Description

- **datasets**:This file stores the training and testing data.

- **model**:The specific implementation of the model stores here.

- **results**:The training results of the model store here.

- dataset.py: Use for dataset processing.

- evaluation.py: Use to evaluate the model.

- generation.py: Use for generating molecules.

- main.py: Model parameters, pre-training, training, testing, and validation code.

- requirements.txt: The environment of the model.

- MolecularOptimization.py: Code for molecular optimization using models.

- tokenizer.py: SMILES encoding code.

- trainer.py: Training code.

- utils.py: Other tool codes.
- We collect the drug-treated and baseline (untreated) expression profiles from the L1000 dataset (https://clue.io/data/CMap2020#LINCS2020). Obtain molecular data from https://pubchem.ncbi.nlm.nih.gov/. Obtain ligands from DTC (https://drugtargetcommons.fimm.fi/). You use main.py to generate drug-like molecules.

## Experimental Reproduction

  - **Train**:

  ```
$ python main.py --pre_train_smiles_vae --train --cell_name=MCF7
  ```

  - **STEP 2**: Test

  ```
$ python main.py --use_seed --generation --cell_name=MCF7 --protein_name=AKT1
  ```

  - **STEP 3**: Evaluate

  ```
$ python main.py --use_seed --generation --cell_name=MCF7 --protein_name=AKT1
  ```

  - **STEP 4**: Molecular Optimization

  ```
$ python MolecularOptimization.py --cell_name=MCF7 --protein_name=AKT1
  ```

