import os
import random
import torch
import numpy as np
import pandas as pd
import subprocess
import re
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from rdkit.DataStructs import FingerprintSimilarity

from rdkit.Chem import Draw

from matplotlib.image import imread
from rdkit import rdBase

rdBase.DisableLog('rdApp.warning')

from tokenizer import vocabulary
# ============================================================================
# GPU 选择
# get_gpu_memory：获取GPU内存信息,
# select_max_available_gpu：选择最大可用的GPU
# get_device：获取GPU设备
def get_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]  # noqa: E731
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(subprocess.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(re.findall(r'\d+', info)[0]) for info in memory_free_info]
    return memory_free_values


def select_max_available_gpu():
    memory_free_values = get_gpu_memory()
    available_gpus = {i: memory for i, memory in enumerate(memory_free_values)}
    if not available_gpus:
        raise ValueError('No available GPU.')
    # available_gpus 排序
    available_gpus = sorted(available_gpus.items(), key=lambda x: x[1], reverse=True)
    return available_gpus[0][0]


def get_device():
    return torch.device(
        "cuda:" + str(select_max_available_gpu()) if torch.cuda.is_available() else "cpu"
    )


# ===========================================================================
# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return rng


# ===========================================================================
def kld_loss(mu, logvar: torch.Tensor,kld_weight=1.0):
    mu = mu.double()
    logvar = logvar.double()

    kld = -0.5 *kld_weight* torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return kld


# ===========================================================================


def tanimoto_similarity(smi1, smi2):
    mols = [Chem.MolFromSmiles(smi1), Chem.MolFromSmiles(smi2)]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mols]
    sim_score = FingerprintSimilarity(fps[0], fps[1])

    return sim_score


def mean_similarity(pred_smiles, label_smiles):
    all_scores = [
        tanimoto_similarity(pred, label) for pred, label in zip(pred_smiles, label_smiles)
    ]
    mean_score = np.mean(all_scores)

    return mean_score


# ==========================================================================
def show_gene_vae_hyperparamaters(args):
    # Hyper-parameters
    params = {}
    print('\n\nGeneVAE Hyperparameter Information:')
    print('=' * 50)
    params['GENE_EXPRESSION_FILE'] = args.gene_expression_file_path
    params['CELL_NAME'] = args.cell_name
    params['GENE_EPOCHS'] = args.train_epochs
    params['GENE_LR'] = args.gene_lr
    params['GENE_NUM'] = args.gene_num
    params['GENE_HIDDEN_SIZES'] = args.gene_hidden_sizes
    params['GENE_LATENT_SIZE'] = args.latent_size
    params['GENE_BATCH_SIZE'] = args.gene_batch_size
    params['GENE_DROUPOUT'] = args.gene_dropout

    for param in params:
        string = param + ' ' * (5 - len(param))
        print(f'{string}:   {params[param]}')
    print('=' * 50)


def show_smiles_vae_hyperparamaters(args):
    # Hyper-parameters
    params = {}
    print('\n\nSmilesVAE Hyperparameter Information:')
    print('=' * 50)
    params['VALID_SMILES_FILE'] = args.valid_smiles_file
    params['SMILES_EPOCHS'] = args.smiles_epochs
    params['EMB_SIZE'] = args.emb_size
    params['HIDDEN_SIZE'] = args.hidden_size
    params['NUM_LAYERS'] = args.num_layers
    params['SMILES_LR'] = args.smiles_lr
    params['SMILES_DROUPOUT'] = args.smiles_dropout
    params['TRAIN_RATE'] = args.train_rate

    for param in params:
        string = param + ' ' * (5 - len(param))
        print(f'{string}:   {params[param]}')
    print('=' * 50)


def show_other_hyperparamaters(args):
    # Hyper-parameters
    params = {}
    print('\n\nOther Hyperparameter Information:')
    print('=' * 50)
    params['PROTEIN_NAME'] = args.protein_name
    params['SOURCE_PATH'] = args.source_path
    params['GEN_PATH'] = args.gen_path
    params['candidate_num'] = args.candidate_num

    for param in params:
        string = param + ' ' * (5 - len(param))
        print(f'{string}:   {params[param]}')
    print('=' * 50)


# =========================================================================
def show_density(args, figure_path, row_num, trained_gene_vae=None,smiles_vae=None, rng=None):
    file_name = os.path.dirname(figure_path)
    if not os.path.exists(file_name):
        os.makedirs(file_name)


    real_genes = pd.read_csv(args.gene_expression_file_path + args.cell_name + '.csv', sep=','
                             )
    target_gene = pd.read_csv("datasets/LINCS/landmark_ctl_MCF7.csv", sep=',', )
    target_gene = target_gene.iloc[:, 1:]
    target_gene = target_gene.mean(axis=0)
    target_gene = pd.DataFrame(target_gene.values)
    target_gene=target_gene.T
    print(target_gene)
    input()
    target_gene = target_gene.sample(real_genes.shape[0], random_state=42, replace=True)

    ctr_genes = pd.DataFrame(target_gene.values)
    ctr_genes = ctr_genes.dropna(how='any')


    real_genes = real_genes.dropna(how='any')

    if row_num == 1:
        random_rows = np.array([1])
        real_genes = real_genes.iloc[random_rows, :]
        real_genes1 = real_genes.iloc[:, 2:]
    else:
        if rng is None:
            rng = np.random.default_rng()
        random_rows = rng.choice(len(real_genes), row_num)
        real_genes = real_genes.iloc[random_rows, :]
        real_genes1 = real_genes.iloc[:, 2:]

    mean_real_pert_all_gene = real_genes1.mean()

    mean_real_all_gene = ctr_genes.mean()

    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['font.family'] = 'sans-serif'

    plt.subplots(figsize=(12, 10.2))
    plt.xlabel("Values of gene expression profile data", fontsize=40)
    plt.ylabel("Density", fontsize=40)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    color = ["#eb5c20","#0cb3ff"]

    mean_real_pert_all_gene.to_csv("mean_real.csv")
    ax = sns.histplot(mean_real_pert_all_gene, bins=50, binwidth=0.2,kde=True, label='Actual Profile', color=color[0],edgecolor='none',shrink=0.9)
    ax.grid(True, linestyle='--', color='#cfc5bb', alpha=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(axis='x', which='major', length=8, width=1.5)
    ax.tick_params(axis='y', which='major', length=8, width=1.5)
    y_ticks = ax.get_yticks()

    ax.set_ylim(bottom=0, top=y_ticks[-1])

    if trained_gene_vae:
        trained_gene_vae.eval()
        smiles_vae.eval()
        smi = real_genes.iloc[:, 0]

        tokenizer = vocabulary(args)
        tokenizer.build_vocab()
        lst = []
        for i in smi:
            s = tokenizer.encode(i)
            lst.append(s)


        max_len = max(len(l) for l in lst)
        padded_lst = [l + [0] * (max_len - len(l)) for l in lst]
        encoded_smi = torch.tensor(padded_lst,device="cuda:0")

        smi_output,_=smiles_vae.encode(encoded_smi)

        real_genes=real_genes.iloc[:, 2:]
        inputs = torch.tensor(real_genes.values, dtype=torch.float32).to(get_device())

        rec_gene, pert_rec_gene = trained_gene_vae.encode(inputs,smi_output)

        rec_genes=trained_gene_vae.decode(rec_gene)
        pert_rec_genes = trained_gene_vae.decode(pert_rec_gene)

        rec_genes = pd.DataFrame(rec_genes.cpu().detach().numpy())
        mean_rec_gene = rec_genes.mean()

        pert_rec_genes = pd.DataFrame(pert_rec_genes.cpu().detach().numpy())
        mean_pert_rec_gene = pert_rec_genes.mean()

        mean_pert_rec_gene.to_csv("mean_pert_rec_gene.csv")

        ax2 = sns.histplot(mean_pert_rec_gene, bins=50, binwidth=0.2,kde=True,label='Reconstructed Profile', color=color[1],edgecolor='none',shrink=0.9)
        ax2.grid(True, linestyle='--', color='#cfc5bb', alpha=0.4)
        for spine in ax2.spines.values():
            spine.set_linewidth(1.5)
        ax2.tick_params(axis='x', which='major', length=8, width=1.5)
        ax2.tick_params(axis='y', which='major', length=8, width=1.5)
        y_ticks2 = ax2.get_yticks()
        ax2.set_ylim(bottom=0, top=y_ticks2[-1])

    plt.legend(loc='upper right',
               frameon=True,
               edgecolor='#B2BBBE',
               facecolor='white',
               framealpha=0.9,
               ncol=1,
               fontsize=30,
               labelspacing=0.5,
               handletextpad=0.5,
               markerscale=1.0,
               fancybox=True,
               shadow=False,
               borderpad=0.4)

    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)



from scipy.stats import gaussian_kde


def show_all_gene_densities(args, trained_gene_vae, smiles_vae,rng=None):
    evaluate_file_dir = os.path.dirname(args.one_gene_density_figure)
    os.makedirs(evaluate_file_dir, exist_ok=True)
    show_density(args, args.one_gene_density_figure, 1, trained_gene_vae, smiles_vae,rng)


def show_smiles(smile_list, save_path):
    num_mols = len(smile_list)
    if num_mols == 1:
        cols = 1
    else:
        cols = 2
    rows = (num_mols + cols - 1) // cols

    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 5 * rows))

    for i, (smiles, ax) in enumerate(zip(smile_list, axs.flat)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            Draw.MolToFile(mol, f'{save_path}/mol_{i}.png')
            ax.imshow(imread(f'{save_path}/mol_{i}.png'))
            ax.set_axis_off()
            ax.set_title(f'Molecule {i}')
        else:
            ax.text(0.5, 0.5, 'Invalid SMILES', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(f'{save_path}/{num_mols}_mol.png')
    plt.show()


if __name__ == '__main__':
    path = "results/0/generation/res-AKT1-RNN.csv"
    dir_name, filename = os.path.split(path)
    data = pd.read_csv(path)
    smiles_list = data['smiles'].tolist()
    show_smiles(smiles_list, dir_name)
