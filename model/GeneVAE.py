import torch
import torch.nn as nn

from utils import kld_loss


# ============================================================================
# Create a VAE encoder
class GeneEncoder(nn.Module):

    def __init__(self, input_size, hidden_sizes, latent_size, activation_fn, dropout):

        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.latent_size = latent_size
        self.activation_fn = activation_fn
        self.dropout = [dropout] * len(self.hidden_sizes)

        num_units = [self.input_size] + self.hidden_sizes

        dense_layers = []
        for index in range(1, len(num_units)):
            dense_layers.append(nn.Linear(num_units[index - 1], num_units[index]))
            dense_layers.append(self.activation_fn)

            if self.dropout[index - 1] > 0.0:
                dense_layers.append(nn.Dropout(p=self.dropout[index - 1]))

        self.encoding = nn.Sequential(*dense_layers)

        self.encoding_to_mu = nn.Linear(self.hidden_sizes[-1], self.latent_size)
        self.encoding_to_logvar = nn.Linear(self.hidden_sizes[-1], self.latent_size)

    def forward(self, inputs):

        projection = self.encoding(inputs)

        mu = self.encoding_to_mu(projection)
        logvar = self.encoding_to_logvar(projection)
        return mu, logvar



# ============================================================================
# Create a VAE decoder
class GeneDecoder(nn.Module):

    def __init__(self, latent_size, hidden_sizes, output_size, activation_fn, dropout):

        super().__init__()

        self.latent_size = latent_size

        self.hidden_sizes = hidden_sizes[::-1]
        self.output_size = output_size
        self.activation_fn = activation_fn
        self.dropout = [dropout] * len(self.hidden_sizes)

        num_units = [self.latent_size] + self.hidden_sizes + [self.output_size]

        dense_layers = []
        for index in range(1, len(num_units) - 1):
            dense_layers.append(nn.Linear(num_units[index - 1], num_units[index]))
            dense_layers.append(self.activation_fn)

            if self.dropout[index - 1] > 0.0:
                dense_layers.append(nn.Dropout(p=self.dropout[index - 1]))
        dense_layers.append(nn.Linear(num_units[-2], num_units[-1]))

        self.decoding = nn.Sequential(*dense_layers)

    def forward(self, latent_z):

        outputs = self.decoding(latent_z)

        return outputs


# ============================================================================
class GeneVAE(nn.Module):

    def __init__(
            self, input_size, hidden_sizes, latent_size, output_size, activation_fn, dropout
    ):

        super().__init__()

        self.encoder = GeneEncoder(
            input_size, hidden_sizes, latent_size, activation_fn, dropout
        )
        self.decoder = GeneDecoder(
            latent_size, hidden_sizes, output_size, activation_fn, dropout
        )
        self.reconstruction_loss = nn.MSELoss(reduction='sum')
        self.kld_loss = kld_loss

    def reparameterize(self, mu, logvar):

        return torch.randn_like(mu).mul_(torch.exp(0.5 * logvar)).add_(mu)

    def encode(self, inputs,smiles_mu=None,smiles_logvar=None):
        self.mu, self.logvar = self.encoder(inputs)
        pert_latent_z = self.reparameterize(self.mu,self.logvar)  

        if smiles_mu != None:
            self.base_mu = self.mu-smiles_mu 
            self.base_logvar = self.logvar + smiles_logvar
            base_latent_z = self.reparameterize(self.base_mu, self.base_logvar)
            return base_latent_z
        else:
            return pert_latent_z

    def decode(self, latent_gene):

        return self.decoder(latent_gene)

    def forward(self, inputs):

        latent_gene = self.encode(inputs)
        outputs = self.decode(latent_gene)

        return latent_gene, outputs

    def joint_loss(self, outputs, targets, isCtl=1,alpha=0.5, beta=1):

        rec_loss = self.reconstruction_loss(outputs, targets)
        rec_loss = rec_loss.double().to(outputs.device)
        if isCtl:
            kld_loss = self.kld_loss(self.base_mu, self.base_logvar, 1.0)
        else:
            kld_loss = self.kld_loss(self.mu,self.logvar,1.0)

        joint_loss = alpha * rec_loss + (1 - alpha) * beta * kld_loss

        return joint_loss, rec_loss, kld_loss

    def load_model(self, path):
        weights = torch.load(path, map_location=torch.device('cuda:0'))
        self.load_state_dict(weights)

    def save_model(self, path):
        torch.save(self.state_dict(), path)
