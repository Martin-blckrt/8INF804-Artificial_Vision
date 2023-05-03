import torch.nn as nn
import torch
from torchviz import make_dot

class Generator(nn.Module):
    def __init__(self, latent_dim=100, image_size=64):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=(4, 4), stride=(1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, image_size, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(image_size),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(image_size, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


model = Generator()
X = torch.randn(1, 100, 1, 1)
y = model(X)
make_dot(y.mean(), params=dict(model.named_parameters())).render("gan_torchviz", format="png")
