# SmilesGEN



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

