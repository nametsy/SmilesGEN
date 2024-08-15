import torch.nn as nn
from trainer import Trainer
from utils import *
from tokenizer import vocabulary
from dataset import load_smiles_data, load_test_gene_data
from model import create_smiles_model, create_optimizer, GeneVAE
from generation import generation
from evaluation import evaluation
import argparse
import os
if __name__ == '__main__':

    file_path = os.path.join("results/MCF7/")
    a = "AKT1"
    lr_lst = [5e-4, 1e-4, 1e-4]
    # smile,SmileGen
    epoch_lst = [100, 500]

    parser = argparse.ArgumentParser(description='SmilesGEN parse')
    parser.add_argument("--use_seed", action="store_true",
                        help="Apply seed for reproduce experimental results")
    parser.add_argument("--cell_name", type=str, default="MCF7",
                        help="Cell name of LINCS files, e.g., mcf7")
    parser.add_argument("--protein_name", type=str, default=f"{a}",
                        help="10 proteins are AKT1, AKT2, AURKB, CTSK, EGFR, HDAC1, MTOR, PIK3CA, SMAD3, and TP53")
    parser.add_argument("--model", type=str, default="RNN",
                        help="Model type, e.g., RNN, Transformer, GRU, LSTM")
    parser.add_argument("--smiles_dropout", type=float, default=0.1,
                        help="Dropout rate for SmilesGEN")
    # ============================================
    parser.add_argument("--pre_train_smiles_vae", action="store_true",
                        help="Pre-train SmilesVAE")
    parser.add_argument("--test_smiles_vae", action="store_true",
                        help="Test SmilesVAE")
    parser.add_argument("--smiles_epochs", type=int, default=epoch_lst[0],
                        help="Number of training epochs for SmilesVAE")
    parser.add_argument("--emb_size", type=int, default=128,
                        help="Embedding size for SmilesVAE")
    parser.add_argument("--hidden_size", type=int, default=192,
                        help="Hidden layer size for SmilesVAE")
    parser.add_argument("--num_layers", type=int, default=3,
                        help="Number of training layers for SmilesVAE")
    parser.add_argument('--latent_size', type=int, default=64,
                        help='Latent vector dimension of SmilesVAE', )  # MCF7: 64
    parser.add_argument("--pre_train_smiles_lr", type=float, default=lr_lst[0],
                        help="Learning rate for Pre-Train SmilesVAE")
    parser.add_argument('--bidirectional', type=bool, default='True',
                        help='Apply bidirectional RNN')
    parser.add_argument('--temperature', type=float, default=1,
                        help='Temperature of the SMILES VAE')
    parser.add_argument('--train_rate', type=float, default=0.9,
                        help='Split training and validating subsets by training rate')
    parser.add_argument('--max_len', type=int, default=100,
                        help='Maximum length of SMILES strings')
    parser.add_argument('--saved_pre_smiles_vae', type=str, default=f'{file_path}/saved_model/saved_pre_smiles_vae',
                        help='Save the pre-trained SmilesVAE')
    parser.add_argument('--pre_train_valid_smiles_file', type=str,
                        default=f'{file_path}/pre_train/pre_train_predicted_valid_smiles',
                        help='Save the valid SMILES into file', )
    parser.add_argument('--pre_train_final_smiles_file', type=str,
                        default=f'{file_path}/pre_train/pre_train_final_smiles',
                        help='Save the valid SMILES into file', )
    parser.add_argument('--smiles_vae_pre_train_results', type=str,
                        default=f'{file_path}/pre_train/smiles_vae_pre_train_results',
                        help='Path to save the results of pre-trained SmilesVAE')
    parser.add_argument('--variant', action='store_true',
                        help='Apply variant smiles')

    # ===========================
    parser.add_argument('--train', action='store_true',
                        help='Train GeneVAE')
    parser.add_argument('--test_gene_vae', action='store_true',
                        help='Validate GeneVAE')
    parser.add_argument('--generation', action='store_true',
                        help='Validate GeneVAE')
    parser.add_argument('--train_epochs', type=int, default=epoch_lst[1],
                        help='GeneVAE training epochs')
    parser.add_argument('--gene_num', type=int, default=978,
                        help='Number of gene values')  # MCF7: 978
    parser.add_argument('--gene_hidden_sizes', type=int, default=[512, 256, 192],
                        help='Hidden layer sizes of GeneVAE')  # MCF7: [512, 256, 128, 100]
    parser.add_argument('--gene_lr', type=float, default=lr_lst[2],
                        help='Learning rate of GeneVAE')  # MCF7: 1e-4
    parser.add_argument('--gene_batch_size', type=int, default=64,
                        help='Batch size for training GeneVAE')  # 64
    parser.add_argument('--gene_dropout', type=float, default=0.2,
                        help='Dropout probability')
    parser.add_argument('--gene_expression_file_path', type=str, default='datasets/LINCS/',
                        help='Path of the training gene expression profile dataset for the VAE')
    parser.add_argument('--test_gene_data', type=str, default='datasets/test_protein/',
                        help='Path of the gene expression profile dataset for test proteins or test disease')
    parser.add_argument('--saved_gene_vae', type=str, default=f'{file_path}/saved_model/saved_gene_vae',
                        help='Save the trained GeneVAE')
    parser.add_argument('--gene_vae_train_results', type=str,
                        default=f'{file_path}/train_results/gene_vae_train_results.csv',
                        help='Path to save the results of trained GeneVAE')
    parser.add_argument('--one_gene_density_figure', type=str,
                        default=f'{file_path}/evaluate/one_gene_density_figure.pdf',
                        help='Path to save the density figures of gene data')
    parser.add_argument('--all_gene_density_figure', type=str,
                        default=f'{file_path}/evaluate/all_gene_density_figure.pdf',
                        help='Path to save the density figures of gene data')
    parser.add_argument('--smiles_lr', type=float, default=lr_lst[1],
                        help='Learning rate of Train SmilesVAE')
    parser.add_argument('--smiles_vae_train_results', type=str,
                        default=f'{file_path}/train_results/smiles_vae_train_results',
                        help='Path to save the results of trained SmilesVAE')
    parser.add_argument('--valid_smiles_file', type=str, default=f'{file_path}/train_results/predicted_valid_smiles',
                        help='Save the valid SMILES into file')
    parser.add_argument('--final_smiles_file', type=str, default=f'{file_path}/train_results/final_smiles',
                        help='Save the valid SMILES into file')
    parser.add_argument('--saved_smiles_vae', type=str, default=f'{file_path}/saved_model/saved_smiles_vae',
                        help='Save the trained SmilesVAE', )

    # ===========================
    parser.add_argument('--calculate_tanimoto', action='store_true',
                        help='Calculate tanimoto similarity for the source ligand and generated SMILES')
    parser.add_argument('--candidate_num', type=int, default=50,
                        help='Number of candidate SMILES strings')
    parser.add_argument('--gene_type', type=str, default='gene_symbol',
                        help='Gene types')
    parser.add_argument('--source_path', type=str, default='datasets/ligands/source_',
                        help='Load the source SMILES strings of known ligands')
    parser.add_argument('--gen_path', type=str, default=f'{file_path}/generation/',
                        help='Save the generated SMILES strings')

    parser.add_argument('--mol_figure_path', type=str, default=f'{file_path}/evaluate/mol_img/',
                        help='Save the image of model')

    args = parser.parse_args()

    if args.use_seed:
        rng = set_seed(888)
    tokenizer = vocabulary(args)
    tokenizer.build_vocab()

    train_dataLoder, valid_dataLoder = load_smiles_data(
        tokenizer,
        args.gene_expression_file_path, args.cell_name,
        args.gene_num, args.gene_batch_size,
        args.train_rate, args.variant)

    device = get_device()

    smiles_vae = create_smiles_model(
        args.model, args.emb_size,
        args.hidden_size, args.num_layers,
        args.latent_size, args.bidirectional,
        tokenizer, device, dropout=args.smiles_dropout)

    gene_vae = GeneVAE(
        args.gene_num, args.gene_hidden_sizes,
        args.latent_size, args.gene_num,
        nn.ReLU(), args.gene_dropout).to(device)
    # ========================================================= #
    #                1.SmilesNET                                #
    # ========================================================= #
    if args.pre_train_smiles_vae:
        show_smiles_vae_hyperparamaters(args)
        print("Pretrain SmilesNET...")

        smile_vae_optimizer = create_optimizer(smiles_vae, args.pre_train_smiles_lr)
        trainer = Trainer(
            args.model, smiles_vae, gene_vae, smile_vae_optimizer, None, device)
        # Pre-Train SmilesVAE
        # 预训练,传入参数:训练和验证,词汇处理器,结果,轮次,温度,最大长度,预训练valid文件,预训练final文件,保存预训练模型
        trainer.pre_train_smiles_vae(
            train_dataLoder, valid_dataLoder, tokenizer,
            args.smiles_vae_pre_train_results,
            args.smiles_epochs, args.temperature, args.max_len,
            args.pre_train_valid_smiles_file, args.pre_train_final_smiles_file, args.saved_pre_smiles_vae)
    # ========================================================= #
    #                2. SmilesGEN                               #
    # ========================================================= #
    if args.train:
        show_smiles_vae_hyperparamaters(args)
        show_gene_vae_hyperparamaters(args)

        print("Train SmilesGEN...")

        encoder_params = list(smiles_vae.encoder.parameters())
        decoder_params = list(smiles_vae.decoder.parameters())
        for param in encoder_params:
            param.requires_grad = False
        smile_vae_optimizer = torch.optim.Adam(decoder_params, lr=args.smiles_lr)
        gene_vae_optimizer = create_optimizer(gene_vae, args.gene_lr)

        smiles_vae.load_model(args.saved_pre_smiles_vae + "_" + args.model + ".pkl")

        trainer = Trainer(args.model,
                          smiles_vae, gene_vae,
                          smile_vae_optimizer, gene_vae_optimizer, device)
        trainer.train(
            train_dataLoder, valid_dataLoder,
            tokenizer, args.train_epochs, args.temperature,
            args.latent_size, args.emb_size,
            args.max_len,
            args.valid_smiles_file, args.smiles_vae_train_results,
            args.saved_smiles_vae, args.saved_gene_vae, args.cell_name)
    # ========================================================= #

    # ========================================================= #
    #                3. Generation                              #
    # ========================================================= #
    if args.generation:
        print("Generate SMILES...")

        smiles_vae.load_model(args.saved_smiles_vae + "_" + args.model + ".pkl")
        gene_vae.load_model(args.saved_gene_vae + '_' + args.cell_name + "_" + args.model + '.pkl')

        test_gene_loader = load_test_gene_data(
            args.test_gene_data, args.cell_name, args.protein_name, args.gene_type, args.gene_batch_size)

        generation(args.model,
                   smiles_vae, gene_vae,
                   test_gene_loader, args.latent_size, args.candidate_num,
                   args.max_len, args.gen_path, args.protein_name,
                   tokenizer, device)
    # ========================================================= #
    #                4. Tanimoto                                #
    # ========================================================= #
    if args.calculate_tanimoto:
        print("Evaluate Tanimoto Similarity.")
        evaluation(
            args.model,
            args.gene_expression_file_path, args.cell_name,
            args.gene_num,
            args.source_path, args.protein_name,
            args.gen_path, args.candidate_num,
            args.mol_figure_path)
