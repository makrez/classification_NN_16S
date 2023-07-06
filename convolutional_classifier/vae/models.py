import torch
import torch.nn as nn

class ConvVAE(nn.Module):
    def __init__(self, input_length, latent_dim=10):
        super(ConvVAE, self).__init__()
        self.input_length = input_length
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(32, 12, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(12, 8, kernel_size=10, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(8, 4, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Flatten(),
            nn.Linear(4*1559, 1200),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(1200, 600),
            nn.ReLU()
        )

        # The encoder output is passed through two linear layers to produce the mean and log variance of the latent variables
        self.fc_mu = nn.Linear(600, latent_dim)
        self.fc_logvar = nn.Linear(600, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 600),
            nn.ReLU(),
            nn.Linear(600, 1200),
            nn.ReLU(),
            nn.Linear(1200, 4*1559),
            nn.ReLU(),
            # Unflatten to get back to Conv operations
            # The output size must be adjusted according to the last convolutional layer size
            nn.Unflatten(1, (4, 1559)),
            nn.ConvTranspose1d(4, 8, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(8, 12, kernel_size=10, stride=1, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(12, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2, padding=3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(16, 6, kernel_size=3, stride=1, padding=4),
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + eps*std
        else:
            return mu

    def forward(self, x):
        print("Input shape:", x.shape)

        # Encoder
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            print(f"Shape after encoder layer {i}:", x.shape)

        h = x
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)

        z = self.reparameterize(mu, logvar)

        # Decoder
        for i, layer in enumerate(self.decoder):
            z = layer(z)
            print(f"Shape after decoder layer {i}:", z.shape)

        print("Final output shape:", z.shape)

        return z, mu, logvar

