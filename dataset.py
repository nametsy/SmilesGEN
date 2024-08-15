import json

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader, random_split



class Smiles_Dataset(Dataset):
    def __init__(self, gene_expression_file, cell_name,
                 tokenizer, gene_num, variant, rng=None):

        data = pd.read_csv(gene_expression_file + cell_name + '.csv', sep=',', )

        data = data.dropna(how='any')

        if variant:
            data['smiles'] = data['smiles'].apply(self.variant_smiles)
        self.data = data
        self.tokenizer = tokenizer
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        smi = self.data.iloc[item]['smiles']
        encoded_smi = self.tokenizer.encode(smi)
        cell = self.data.iloc[item]["cell_name"]
        gene = self.data.iloc[item].iloc[2:].to_numpy().astype('float32')
        return encoded_smi, gene, cell

    def variant_smiles(self, smi):
        mol = Chem.MolFromSmiles(smi)
        atom_indices = list(range(mol.GetNumAtoms()))
        mol = Chem.RenumberAtoms(mol, atom_indices)
        return Chem.MolToSmiles(mol, canonical=True)

class Smiles_DataLoader(DataLoader):
    def __init__(self, gene_expression_file, cell_name,
                 tokenizer, gene_num, batch_size, train_rate=0.9, variant=False):
        self.gene_expression_file = gene_expression_file
        self.cell_name = cell_name
        self.tokenizer = tokenizer
        self.gene_num = gene_num
        self.batch_size = batch_size
        self.train_rate = train_rate
        self.variant = variant

    def collate_fn(self, batch):
        smiles, genes,cell = zip(*batch)
        smi_tensor = [torch.tensor(smi).squeeze(0) for smi in smiles]
        gene_expression_tensor = torch.tensor(np.array(genes))
        smi_tensors = torch.nn.utils.rnn.pad_sequence(smi_tensor, batch_first=True)
        cell_tensors = np.array(cell)
        return smi_tensors, gene_expression_tensor,cell_tensors

    def get_dataloader(self):
        dataset = Smiles_Dataset(self.gene_expression_file, self.cell_name,
                                 self.tokenizer, self.gene_num, self.variant)
        train_size = int(len(dataset) * self.train_rate)
        test_size = len(dataset) - train_size
        train_data, test_data = random_split(
            dataset=dataset, lengths=[train_size, test_size], generator=torch.Generator().manual_seed(0), )
        train_dataloader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True,
            collate_fn=self.collate_fn, num_workers=1, )
        test_dataloader = DataLoader(
            test_data, batch_size=self.batch_size, shuffle=True,
            collate_fn=self.collate_fn, num_workers=1, )
        return train_dataloader, test_dataloader


# ===============================================================

def load_smiles_data(tokenizer, gene_expression_file_path, cell_name, gene_num,
                     gene_batch_size, train_rate, variant):
    smiles_loader = Smiles_DataLoader(
        gene_expression_file_path, cell_name, tokenizer, gene_num,
        batch_size=gene_batch_size,
        train_rate=train_rate,
        variant=variant
    )
    train_dataloader, valid_dataloader = smiles_loader.get_dataloader()
    return train_dataloader, valid_dataloader


# ===============================================================
class GeneExpressionDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.data_num = len(data)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        gene_data = torch.tensor(self.data.iloc[idx].values).float()
        return gene_data


def symbol2hsa(input_symbol):
    with open('datasets/tools/symbol2hsa.json', mode='rt', encoding='utf-8') as f:
        symbol_data = json.load(f)
        symbols = list(symbol_data.keys())
    hsas = []
    for sym in input_symbol:
        if sym in symbols:
            hsas.append(symbol_data[sym])
        else:
            hsas.append('-')
    return hsas


def common(df_tgt, gene_type):
    df_source = pd.read_csv('datasets/tools/source_genes.csv', sep=',')
    source_hsas = list(df_source.columns)
    tgt_hsas = list(df_tgt.columns)
    if not gene_type == 'gene_symbol':
        tgt_hsas = symbol2hsa(tgt_hsas)

        df_tgt = df_tgt.set_axis(tgt_hsas, axis=1)
    common_hsas = list(set(tgt_hsas) & set(source_hsas))
    common_hsas = sorted(common_hsas, key=source_hsas.index)
    df_source[common_hsas] = df_tgt[common_hsas]
    return df_source


def load_test_gene_data(test_gene_data, cell_name, protein_name, gene_type, gene_batch_size):
    data = pd.read_csv(test_gene_data + cell_name + "_" + protein_name + '.csv', sep=',')

    # data = data.iloc[:, 1:]
    data = data.iloc[0:1, 1:]
    print(data)
    data = common(data, gene_type)
    data.to_csv('test_gene_data.csv', index=False)
    print(data)

    test_data = GeneExpressionDataset(data)
    test_loader = DataLoader(test_data, batch_size=gene_batch_size, shuffle=False)
    return test_loader


if __name__ == '__main__':
    pass
