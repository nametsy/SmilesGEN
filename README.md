# SmilesGEN

![](https://github.com/nametsy/SmilesGEN/blob/main/framework.png)

We propose SmilesGEN, a model that can generate drugs based on gene expression profiles.
You can use this model to generate drug like molecules using the required expression profile.
The following is a detailed introduction to the model.

## Environment Installation

Execute the following command:

```
$ pip install -r requirements.txt
```

## File Description

- **datasets**

- **model**

- **results**

- dataset.py

- evaluation.py

- generation.py

- main.py

- requirements.py

- tokenizer.py

- trainer.py

- utils.py
We collected the drug-treated and baseline (untreated) expression profiles from the L1000 datase.
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
$ python main.py --cell_name=MCF7 --protein_name=AKT1 --calculate_tanimoto
  ```

