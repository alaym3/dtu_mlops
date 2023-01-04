"""
Adapted from
https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb

A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model import Decoder, Encoder, Model
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import hydra
from torch.optim import Adam

# add logger
import logging
log = logging.getLogger(__name__)


# hyperparameters with hydra
@hydra.main(config_name="config.yaml")

def main(cfg):
    # initial settings
    torch.manual_seed(cfg.seed)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # set hyperparameters from selected experiment
    # hyperparameters = config.experiment


    # Data loading
    mnist_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = MNIST(cfg.dataset_path, transform=mnist_transform, train=True, download=True)
    test_dataset  = MNIST(cfg.dataset_path, transform=mnist_transform, train=False, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=cfg.batch_size, shuffle=False)

    encoder = Encoder(input_dim=cfg.x_dim, hidden_dim=cfg.hidden_dim, latent_dim=cfg.latent_dim)
    decoder = Decoder(latent_dim=cfg.latent_dim, hidden_dim = cfg.hidden_dim, output_dim = cfg.x_dim)

    model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)


    BCE_loss = nn.BCELoss()

    def loss_function(x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

        return reproduction_loss + KLD

    optimizer = Adam(model.parameters(), lr=cfg.lr)


    log.info("Start training VAE...")
    model.train()
    for epoch in range(cfg.epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(cfg.batch_size, cfg.x_dim)
            x = x.to(DEVICE)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            overall_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        log.info("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*cfg.batch_size))    
    log.info("Finish!!")

    # save weights
    torch.save(model, f"{os.getcwd()}/trained_model.pt")

    # Generate reconstructions
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            x = x.view(cfg.batch_size, cfg.x_dim)
            x = x.to(DEVICE)      
            x_hat, _, _ = model(x)       
            break

    save_image(x.view(cfg.batch_size, 1, 28, 28), 'orig_data.png')
    save_image(x_hat.view(cfg.batch_size, 1, 28, 28), 'reconstructions.png')

    # Generate samples
    with torch.no_grad():
        noise = torch.randn(cfg.batch_size, cfg.latent_dim).to(DEVICE)
        generated_images = decoder(noise)
        
    save_image(generated_images.view(cfg.batch_size, 1, 28, 28), 'generated_sample.png')

if __name__ == "__main__":
    main()
