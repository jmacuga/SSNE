import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.fc_1 = nn.Linear(input_dim, 1024)
        # self.bn1 = nn.BatchNorm1d(2048)
        self.fc_2 = nn.Linear(1024, 256)
        # self.bn2 = nn.BatchNorm1d(1024)
        # self.fc_3 = nn.Linear(1024, 256)
        # self.bn3 = nn.BatchNorm1d(256)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        # self.dropout = nn.Dropout(0.1)
        self.fc_mean = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)

        self.training = True

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        # x = self.bn1(x)
        x = self.LeakyReLU(x)
        x = self.fc_2(x)
        # x = self.bn2(x)
        x = self.LeakyReLU(x)
        # x = self.fc_3(x)
        # x = self.bn3(x)
        # x = self.LeakyReLU(x)
        # x = self.dropout(x)

        mean = self.fc_mean(x)
        log_var = self.fc_var(x)  # encoder produces mean and log of variance
        #             (i.e., parateters of simple tractable normal distribution "q"

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc_1 = nn.Linear(latent_dim, 256)
        # self.bn1 = nn.BatchNorm1d(256)

        self.fc_2 = nn.Linear(256, 1024)
        # self.bn2 = nn.BatchNorm1d(1024)

        # self.fc_3 = nn.Linear(1024, 2048)
        # self.bn3 = nn.BatchNorm1d(2048)

        self.fc_5 = nn.Linear(1024, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.fc_1(x)
        # h = self.bn1(h)
        h = self.LeakyReLU(h)

        h = self.fc_2(h)
        # h = self.bn2(h)
        h = self.LeakyReLU(h)

        # h = self.fc_3(h)
        # h = self.bn3(h)
        # h = self.LeakyReLU(h)

        x_hat = torch.sigmoid(self.fc_5(h))
        x_hat = x_hat.view([-1, 3, 32, 32])
        return x_hat


class VAE(nn.Module):
    def __init__(self, x_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(
            input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim
        )
        self.decoder = Decoder(
            latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim
        )

    def reparameterization(self, mean, var):
        eps = torch.randn_like(mean)
        z = eps * var + mean
        return z

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(
            mean, torch.exp(0.5 * log_var)
        )  # takes exponential function (log var -> var)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var


def vae_loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD
