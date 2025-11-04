import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
import time
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import utils
import os
import vq


# device = torch.device("cuda:0" if True and torch.cuda.is_available() else "cpu")
# the task is to train a pair of encoder and decoder, and save their parameters.
def learning(args):
    # load data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gnn_embeds = torch.load('../../src/lgn-'+ args.dataset +'-' + str(args.rec_dim) + '.pth.tar')  # change the file name?
    user_gnn_embeds = gnn_embeds['embedding_user.weight']  # requires_grad = False
    item_gnn_embeds = gnn_embeds['embedding_item.weight']  # requires_grad = False
    item_num, gnn_dim = item_gnn_embeds.shape
    user_num, _ = user_gnn_embeds.shape

    user_test_size = int(user_num * 0.1)
    item_test_size = int(item_num * 0.1)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    user_indices = torch.randperm(user_num)
    item_indices = torch.randperm(item_num)

    # tsne = TSNE(n_components=2)
    # embedded_data = tsne.fit_transform(item_gnn_embeds[item_indices[:item_test_size]].cpu())
    # plt.scatter(embedded_data[:, 0], embedded_data[:, 1])
    # plt.title('sigma-vae')
    # plt.savefig('item_embeds.png')
    # sys.exit()
    
    # create dataset
    user_train_loader, user_valid_loader = get_train_valid_loader(user_gnn_embeds[user_indices[user_test_size:]])
    item_train_loader, item_valid_loader = get_train_valid_loader(item_gnn_embeds[item_indices[item_test_size:]])
    
    # initialzation
    user_vae = SigmaVae(gnn_dim, args.latent_dim, 2, user_num).to(device)
    item_vae = SigmaVae(gnn_dim, args.latent_dim, 2, item_num).to(device)
    # user_vae = VanillaVae(gnn_dim, args.latent_dim, 2, user_num).to(device)
    # item_vae = VanillaVae(gnn_dim, args.latent_dim, 2, item_num).to(device)


    # training VAE models
    modeling(item_vae, item_train_loader, item_valid_loader, device, args.dataset, 'item_gnn')
    print(f'The training of VAE on {args.dataset} for item_gnn_embeds is done.')
    # modeling(user_vae, user_train_loader, user_valid_loader, device, args.dataset, 'user_gnn')
    # print(f'The training of VAE on {args.dataset} for user_gnn_embeds is done.')

    _, temp2 = item_vae.encode(item_gnn_embeds[item_indices[:item_test_size]])
    temp = torch.cat([torch.tensor(temp2[0]), torch.tensor(temp2[1])], dim=0)
    temp = np.array(temp.cpu())
    tsne = TSNE(n_components=2)
    embedded_data = tsne.fit_transform(temp.cpu())
    plt.scatter(embedded_data[:, 0], embedded_data[:, 1])
    plt.title('sigma-vae')
    plt.savefig('sigmavae.png')
    sys.exit()

    # reconstruction
    # user_test_loader = DataLoader(VaeDataset(user_gnn_embeds[user_indices[:user_test_size]]), batch_size=user_test_size)
    item_test_loader = DataLoader(VaeDataset(item_gnn_embeds[item_indices[:item_test_size]]), batch_size=item_test_size)
    for i, x in enumerate(item_test_loader):
        x_hat, _, _ = item_vae.valid(x.to(device))
        dis = torch.nn.functional.mse_loss(x_hat, x)
        print('reconstruction metric =', dis.item())
    sys.exit()


def modeling(model, train_loader, valid_loader, device, dataset, name, n_epochs=1000):
    # components
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)
    criterion = nn.MSELoss()
    early_stopper = utils.EarlyStopping(patience=1000)
    train_loss_list = []
    vaild_loss_list = []
    valid_bar = 100
    tic = time.time()
    for e in tqdm(range(n_epochs), desc=f'VAE training on [Dataset: {dataset}; Target: {name}]'):
        model.train()
        train_loss = 0
        for i, x in enumerate(train_loader):
            current_batch_size = x.shape[0]
            optimizer.zero_grad()
            x_hat, mus, encoded_embeds = model(x.to(device))
            loss = criterion(x_hat, x)
            # loss = configure_loss(x.to(device), x_hat, mus, encoded_embeds, criterion)  # reconstruction + mu + twins
            loss.backward()
            optimizer.step()
            record_loss = torch.nn.functional.mse_loss(x_hat, x)
            train_loss += record_loss.item()
        # train_loss /= len(train_loader.dataset)  # average loss in each epoch
        train_loss /= (i+1)  # average loss in each epoch

        model.eval()
        valid_loss = 0
        for i, x in enumerate(valid_loader):
            current_batch_size = x.shape[0]
            x_hat, mus, encoded_embeds = model.valid(x.to(device))
            loss = criterion(x_hat, x)
            # loss = configure_loss(x.to(device), x_hat, mus, encoded_embeds, criterion)  
            record_loss = torch.nn.functional.mse_loss(x_hat, x)   
            valid_loss += record_loss.item()
        # valid_loss /= len(valid_loader.dataset)
        valid_loss /= (i+1)
        
        train_loss_list.append(train_loss)
        vaild_loss_list.append(valid_loss)

        # save valided model
        if valid_loss < valid_bar:
            directory_path = '../../ckpt/'+dataset+'/'
            valid_bar = valid_loss
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            torch.save(model.state_dict(), directory_path + name+'_vae.pth')  # overwrite
        
        # early stop
        early_stopper(valid_loss)
        if early_stopper.should_stop():
            print("Early stopping")
            break
        toc = time.time()
        if (e+1) % 10 == 0:
            tqdm.write(f"Epoch {e+1}/{n_epochs} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, elapsed {(toc - tic):.2f}s")
    
    loss_logs = pd.DataFrame({'train': np.array(train_loss_list), 'valid': np.array(vaild_loss_list)})
    loss_logs.to_csv(f'VAE_{dataset}_{name}_loss.csv')
    
    # clear GPU usage
    del model
    del optimizer
    torch.cuda.empty_cache()

def configure_loss(x, x_hat, mus, encoded_embeds, criterion, beta=0.5, gama=0.5, lamda=0.005):
    batch, dim = encoded_embeds[0].shape
    
    # VAE Reconstruction, https://www.cs.toronto.edu/~bonner/courses/2022s/csc2547/papers/generative/background/vae_tutorial_doersch.pdf
    reconstruction_loss = criterion(x_hat, x)  # torch.norm.default = l2_norm
    
    # Larger Variance, https://arxiv.org/pdf/2412.08635
    avg_mu = torch.mean(torch.stack(mus), dim=0)  # [m, batch, dim] -> [batch, dim]
    mu_loss = torch.norm(avg_mu) / batch
    
    # Barlow Twins, https://arxiv.org/pdf/2103.03230
    z_norm = []
    for embed in encoded_embeds:  # mus or encoded embeds?
        temp = (embed - embed.mean(0)) / embed.std(0)
        z_norm.append(temp)
    c = torch.mm(z_norm[0].T, z_norm[1]) / dim  # [dim*dim]
    c_diff = (c - torch.eye(dim, device=x.device)).pow(2)
    diagonal = torch.diagonal(c_diff)
    c_diff = c_diff * lamda
    c_diff += torch.diag(diagonal - torch.diag(c_diff))
    twins_loss = c_diff.sum() / dim
    
    loss = reconstruction_loss + beta*mu_loss + gama*twins_loss
    return loss
    
class SigmaVae(nn.Module):
    def __init__(self, input_dim, dim, enc_num, embeds_num, mask_ratio=0.1, sigma=0.1, seed=42):
        super(SigmaVae, self).__init__()
        self.enc_num = enc_num
        self.dim = dim
        # epsilon-generation
        torch.manual_seed(seed)
        self.epsilon = torch.randn(1, self.dim)
        self.sigma = torch.normal(0, sigma, size=(1, ))
        
        # encoders
        self.encoders = nn.ModuleList()
        for n in range(enc_num):
            encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(256, dim),
                )
            self.encoders.append(encoder)
            
        self.decoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 128), 
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.ReLU(), 
            nn.Linear(128, input_dim),
            )
        
        # mask & position
        self.pos = nn.Embedding(1, input_dim)
        self.pos.weight.data.uniform_(-1.0 / embeds_num, 1.0 / embeds_num)  # init
        self.mask_ratio = mask_ratio

    def forward(self, x):  # shape = [batch, emb]
        # masking + position (train only)
        mask = x[0, :].bernoulli_(self.mask_ratio).bool()
        x = torch.masked_fill(x, mask, 0)
        x += self.pos.weight

        b, e =x.shape
        # encode    
        encoded_embeds = []
        mus = []
        for m in range(self.enc_num):
            mu = self.encoders[m](x)  # shape = [batch, dim]
            ze = mu + self.sigma.to(x.device) * self.epsilon.to(x.device)  # z = u + sigma * epsilon
            mus.append(mu)
            encoded_embeds.append(ze)
        zq = torch.mean(torch.stack(encoded_embeds), dim=0) # average pooling

        # decode
        x_hat = self.decoder(zq)
        return x_hat, mus, encoded_embeds
    
    @torch.no_grad()
    def valid(self, x):  # shape = [batch, emb]
        b, e =x.shape
        # encode    
        encoded_embeds = []
        mus = []
        for m in range(self.enc_num):
            mu = self.encoders[m](x)  # shape = [batch, dim]
            ze = mu + self.sigma.to(x.device) * self.epsilon.to(x.device)  # z = u + sigma * e
            mus.append(mu)
            encoded_embeds.append(ze)
        zq = torch.mean(torch.stack(encoded_embeds), dim=0) # [batch, dim]

        # decode
        x_hat = self.decoder(zq)
        return x_hat, mus, encoded_embeds
    
    def encode(self, x):
        b, e =x.shape
        # encode    
        encoded_embeds = []
        mus = []
        for m in range(self.enc_num):
            mu = self.encoders[m](x)  # shape = [batch, dim]
            ze = mu + self.sigma.to(x.device) * self.epsilon.to(x.device)  # z = u + sigma * e
            mus.append(mu)
            encoded_embeds.append(ze)  # left and right
        return mus, encoded_embeds
    
    def decode(self, zq):
        return self.decoder(zq)


class VanillaVae(nn.Module):
    def __init__(self, input_dim, dim, enc_num, embeds_num, seed=42):
        super(VanillaVae, self).__init__()
        self.enc_num = enc_num
        self.dim = dim
        # epsilon-generation
        torch.manual_seed(seed)
        # self.epsilon = torch.randn(1, self.dim)
        
        # encoders
        self.encoders = nn.ModuleList()
        for n in range(enc_num):
            encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.Dropout(0.5),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.Linear(256, dim),
                )
            self.encoders.append(encoder)
            
        self.decoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 128), 
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.ReLU(), 
            nn.Linear(128, input_dim),
            )
        

    def forward(self, x):  # shape = [batch, emb]
        # encode    
        encoded_embeds = []
        mus = []
        for m in range(self.enc_num):
            mu = self.encoders[m](x)  # shape = [batch, dim]
            ze = mu # + self.epsilon.to(x.device)  # z = u + sigma * epsilon
            mus.append(mu)
            encoded_embeds.append(ze)
        zq = torch.mean(torch.stack(encoded_embeds), dim=0) # average pooling

        # decode
        x_hat = self.decoder(zq)
        return x_hat, mus, encoded_embeds
    
    @torch.no_grad()
    def valid(self, x):  # shape = [batch, emb]
        # encode    
        encoded_embeds = []
        mus = []
        for m in range(self.enc_num):
            mu = self.encoders[m](x)  # shape = [batch, dim]
            ze = mu # + self.epsilon.to(x.device)  # z = u + sigma * e
            mus.append(mu)
            encoded_embeds.append(ze)
        zq = torch.mean(torch.stack(encoded_embeds), dim=0) # [batch, dim]

        # decode
        x_hat = self.decoder(zq)
        return x_hat, mus, encoded_embeds
    
    def encode(self, x):
        # encode    
        encoded_embeds = []
        mus = []
        for m in range(self.enc_num):
            mu = self.encoders[m](x)  # shape = [batch, dim]
            ze = mu # + self.epsilon.to(x.device)  # z = u + sigma * e
            mus.append(mu)
            encoded_embeds.append(ze)  # left and right
        return mus, encoded_embeds
    
    def decode(self, zq):
        return self.decoder(zq)
        

def get_train_valid_loader(embeds, valid_size=0.2, shuffle=True):
    num, dim = embeds.shape
    indices = list(range(num))
    split = int(np.floor(valid_size * num))
    if shuffle:
        np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    train_loader = DataLoader(VaeDataset(embeds), sampler=train_sampler, batch_size=512)
    valid_loader = DataLoader(VaeDataset(embeds), sampler=valid_sampler, batch_size=256)
    return train_loader, valid_loader


class VaeDataset(Dataset):
    def __init__(self, embeds):
        super().__init__()
        self.embeds = embeds

    def __len__(self):
        return len(self.embeds)

    def __getitem__(self, index: int):
        return self.embeds[index]