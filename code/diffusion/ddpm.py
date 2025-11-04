import torch
import torch.nn as nn
import numpy as np
import sys


#  ------------------ native conditional DDPM-1D ------------------ 
class DDPM():
    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)  # get betas schedule, "linear"
        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas).to(device)
        product = 1
        
        # alpha_bars: a lambda that takes an argument t from 0 to 1 and produces the cumulative product of (1-beta) up to that part of the diffusion process.
        for i, alpha in enumerate(alphas):  
            product *= alpha
            alpha_bars[i] = product
        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        
    def sample_forward(self, x, t, eps=None):  # add noise
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1).to(x.device)
        if eps is None:  # eps is the Gaussion noise
            eps = torch.randn_like(x)
        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x
        return res
    
    def sample_backward(self, embed_shape, net, condition, device, simple_var=True):  # for external usage
        x = torch.randn(embed_shape).to(device)
        net = net.to(device)
        net.eval()
        for t in range(100 - 1, -1, -1):  # self.n_steps
            x = self.sample_backward_step(x, t, net, condition, simple_var)
        return x
    
    def sample_backward_step(self, x_t, t, net, condition, simple_var=True):
        n = x_t.shape[0]
        t_tensor = torch.tensor([t] * n, dtype=torch.long).to(x_t.device)
        eps = net(x_t, condition, t_tensor)

        if t == 0:
            noise = 0
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (1 - self.alpha_bars[t - 1]) / (1 - self.alpha_bars[t]) * self.betas[t]
            noise = torch.randn_like(x_t)
            noise *= torch.sqrt(var)

        mean = (x_t - (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) * eps) / torch.sqrt(self.alphas[t])
        x_t = mean + noise

        return x_t


class PositionalEncoding(nn.Module):

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, d_model)
        nn.init.xavier_normal_(self.embedding.weight)
        self.embedding.requires_grad_(False)

    def forward(self, t):
        return self.embedding(t)
    
    
class MLP(nn.Module):
    def __init__(self, dim, c_dim, n_steps, hidden_dim=512, block_num=2):  # input_dim = 2048
        super(MLP, self).__init__()
        self.blocks = nn.ModuleList()
        for m in range(block_num):
            block = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(hidden_dim, dim),
                )
            self.blocks.append(block)

        # a linear transformation layer to map from c_dim to dim dimensions
        self.c_remap = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(c_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(hidden_dim, dim),
            )
        self.pe = PositionalEncoding(n_steps, dim)
        

    def forward(self, x, c, t, coefficient=2):  # x: input = [batch, tokens, dim], c: condition = [batch, c_dim]
        temp = x + coefficient*self.c_remap(c).unsqueeze(1) + self.pe(t).unsqueeze(1)  # .unsqueeze(1) is to transpose shapes from [batch, dim] to [batch, 1, dim]
        for i, block in enumerate(self.blocks):
            temp = block(temp) + temp  # residual connection
        return temp
    