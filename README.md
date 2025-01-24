# SmilesGEN：
## Introduction
We propose SmilesGEN, a model that can generate drugs based on gene expression profiles.You can use this model to generate and optimize drug like molecules using the required expression profile.
The following is a detailed introduction to the model:
## Model Architecture
![](https://github.com/nametsy/SmilesGEN/blob/main/framework.png)


## Environment Installation
The required packages can be viewed in the requirements.txt .
Execute the following command to install the package:

```
$ pip install -r requirements.txt
```

## File Description

- **datasets**:This file stores training and testing data.

- **model**:The specific implementation of the model is stored here.

- **results**:The training results of the model.

- dataset.py: Used for dataset processing.

- evaluation.py: Used to evaluate the model.

- generation.py: Used for generating molecules.

- main.py: Model parameters, pre training, training, testing, validation code.

- requirements.txt: The environment of the model.

- MolecularOptimization.py: Code for molecular optimization using models.

- tokenizer.py: SMILES encoding code.

- trainer.py: Training code.

- utils.py: Other tool codes.
- We collected the drug-treated and baseline (untreated) expression profiles from the L1000 datase(https://clue.io/data/CMap2020#LINCS2020). Obtaining molecular data from https://pubchem.ncbi.nlm.nih.gov/.   Obtaining ligands from DTC(https://drugtargetcommons.fimm.fi/).  You can use main.py to generate Drug-like molecules.

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

