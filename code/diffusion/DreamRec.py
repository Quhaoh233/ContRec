# code source: https://github.com/YangZhengyi98/DreamRec
# paper link: https://proceedings.neurips.cc/paper_files/paper/2023/file/4c5e2bcbf21bdf40d75fddad0bd43dc9-Paper-Conference.pdf
# Thanks for their great work!


import numpy as np
import pandas as pd
import math
import random
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import os
import logging
import copy
import time as Time
from collections import Counter
import sys


# utility functions
def extract_axis_1(data):
    res = []
    for i in range(data.shape[0]):
        res.append(data[i, -1, :])
    res = torch.stack(res, dim=0).unsqueeze(1)
    return res

def pad_history(itemlist,length,pad_item):
    if len(itemlist)>=length:
        return itemlist[-length:]
    if len(itemlist)<length:
        temp = [pad_item] * (length-len(itemlist))
        itemlist.extend(temp)
        return itemlist

def calculate_hit(sorted_list,topk,true_items,hit_purchase,ndcg_purchase):
    for i in range(len(topk)):
        rec_list = sorted_list[:, -topk[i]:]
        # print(rec_list)
        # print(true_items)
        # print('...........')
        # break
        for j in range(len(true_items)):
            if true_items[j] in rec_list[j]:
                rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])
                # total_reward[i] += rewards[j]
                # if rewards[j] == r_click:
                #     hit_click[i] += 1.0
                #     ndcg_click[i] += 1.0 / np.log2(rank + 1)
                # else:
                hit_purchase[i] += 1.0
                ndcg_purchase[i] += 1.0 / np.log2(rank + 1)


# schedules
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def linear_beta_schedule(timesteps, beta_start, beta_end):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
    return betas

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


# diffusion
class diffusion():
    def __init__(self, timesteps, beta_start, beta_end, w, device, beta_sche):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.w = w
        self.device = device
        
        if beta_sche == 'linear':
            self.betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=self.beta_start, beta_end=self.beta_end)
        elif beta_sche == 'exp':
            self.betas = exp_beta_schedule(timesteps=self.timesteps)
        elif beta_sche =='cosine':
            self.betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif beta_sche =='sqrt':
            self.betas = torch.tensor(betas_for_alpha_bar(self.timesteps, lambda t: 1-np.sqrt(t + 0.0001),)).float()

        # define alphas 
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
    
    def q_sample(self, x_start, t, noise=None):
        # print(self.betas)
        if noise is None:
            noise = torch.randn_like(x_start)
            # noise = torch.randn_like(x_start) / 100
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    # h = conditioning, t = timestep
    def p_losses(self, denoise_model, x_start, h, t, noise=None, loss_type="l2"):
        # 
        if noise is None:
            noise = torch.randn_like(x_start) 
            # noise = torch.randn_like(x_start) / 100
        
        # 
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_x = denoise_model(x_noisy, h, t)
        
        # 
        if loss_type == 'l1':
            loss = F.l1_loss(x_start, predicted_x)
        elif loss_type == 'l2':
            loss = F.mse_loss(x_start, predicted_x)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(x_start, predicted_x)
        else:
            raise NotImplementedError()

        return loss, predicted_x

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )
    
    @torch.no_grad()
    def p_sample(self, model_forward, model_forward_uncon, x, h, t, t_index):

        x_start = (1 + self.w) * model_forward(x, h, t) - self.w * model_forward_uncon(x, t)
        x_t = x 
        model_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)

            return model_mean + torch.sqrt(posterior_variance_t) * noise 
        
    @torch.no_grad()
    def sample(self, model_forward, model_forward_uncon, h):
        x = torch.randn_like(h)
        # x = torch.randn_like(h) / 100

        for n in reversed(range(0, self.timesteps)):
            x = self.p_sample(model_forward, model_forward_uncon, x, h, torch.full((h.shape[0], ), n, device=self.device, dtype=torch.long), n)

        return x


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_units, num_heads, dropout_rate):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        
        self.linear_q = nn.Linear(hidden_size, num_units)
        self.linear_k = nn.Linear(hidden_size, num_units)
        self.linear_v = nn.Linear(hidden_size, num_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, queries, keys):
        """
        :param queries: A 3d tensor with shape of [N, T_q, C_q]
        :param keys: A 3d tensor with shape of [N, T_k, C_k]
        
        :return: A 3d tensor with shape of (N, T_q, C)
        
        """
        Q = self.linear_q(queries)  # (N, T_q, C)
        K = self.linear_k(keys)  # (N, T_k, C)
        V = self.linear_v(keys)  # (N, T_k, C)
        
        # Split and Concat
        split_size = self.hidden_size // self.num_heads
        Q_ = torch.cat(torch.split(Q, split_size, dim=2), dim=0)  # (h*N, T_q, C/h)
        K_ = torch.cat(torch.split(K, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)
        V_ = torch.cat(torch.split(V, split_size, dim=2), dim=0)  # (h*N, T_k, C/h)
        
        # Multiplication
        matmul_output = torch.bmm(Q_, K_.transpose(1, 2)) / self.hidden_size ** 0.5  # (h*N, T_q, T_k)
        
        # Key Masking
        key_mask = torch.sign(torch.abs(keys.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_k)
        key_mask_reshaped = key_mask.unsqueeze(1).repeat(1, queries.shape[1], 1)  # (h*N, T_q, T_k)
        key_paddings = torch.ones_like(matmul_output) * (-2 ** 32 + 1)
        matmul_output_m1 = torch.where(torch.eq(key_mask_reshaped, 0), key_paddings, matmul_output)  # (h*N, T_q, T_k)
        
        # Causality - Future Blinding
        diag_vals = torch.ones_like(matmul_output[0, :, :])   # (T_q, T_k)
        tril = torch.tril(diag_vals)  # (T_q, T_k)
        causality_mask = tril.unsqueeze(0).repeat(matmul_output.shape[0], 1, 1)  # (h*N, T_q, T_k)
        causality_paddings = torch.ones_like(causality_mask) * (-2 ** 32 + 1)
        matmul_output_m2 = torch.where(torch.eq(causality_mask, 0), causality_paddings, matmul_output_m1)  # (h*N, T_q, T_k)
        
        # Activation
        matmul_output_sm = self.softmax(matmul_output_m2)  # (h*N, T_q, T_k)
        
        # Query Masking
        query_mask = torch.sign(torch.abs(queries.sum(dim=-1))).repeat(self.num_heads, 1)  # (h*N, T_q)
        query_mask = query_mask.unsqueeze(-1).repeat(1, 1, keys.shape[1])  # (h*N, T_q, T_k)
        matmul_output_qm = matmul_output_sm * query_mask
        
        # Dropout
        matmul_output_dropout = self.dropout(matmul_output_qm)
        
        # Weighted Sum
        output_ws = torch.bmm(matmul_output_dropout, V_)  # ( h*N, T_q, C/h)
        
        # Restore Shape
        output = torch.cat(torch.split(output_ws, output_ws.shape[0] // self.num_heads, dim=0), dim=2)  # (N, T_q, C)
        
        # Residual Connection
        output_res = output + queries
        
        return output_res


# network
class Tenc(nn.Module):
    def __init__(self, hidden_size, item_embeds, state_size, dropout, diffuser_type, device, num_heads=1):
        super(Tenc, self).__init__()
        self.state_size = state_size
        self.dropout = nn.Dropout(dropout)
        self.diffuser_type = diffuser_type
        self.device = device
        # item_embeddings, shape = [item_num+1, hidden_size]
        self.item_embeddings = item_embeds
        self.item_num, self.hidden_size = item_embeds.shape
        # none embeddings
        self.none_embedding = nn.Embedding(
            num_embeddings=1,
            embedding_dim=self.hidden_size,
        )
        nn.init.normal_(self.none_embedding.weight, 0, 1)
        # positional embeddings
        self.positional_embeddings = nn.Embedding(
            num_embeddings=state_size,
            embedding_dim=self.hidden_size
        )
        # emb_dropout is added
        self.emb_dropout = nn.Dropout(dropout)
        self.ln_1 = nn.LayerNorm(self.hidden_size)
        self.ln_2 = nn.LayerNorm(self.hidden_size)
        self.ln_3 = nn.LayerNorm(self.hidden_size)
        self.mh_attn = MultiHeadAttention(self.hidden_size, self.hidden_size, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(self.hidden_size, self.hidden_size, dropout)
        # self.s_fc = nn.Linear(hidden_size, item_num)
        # self.ac_func = nn.ReLU()

        # self.step_embeddings = nn.Embedding(
        #     num_embeddings=50,
        #     embedding_dim=hidden_size
        # )

        self.step_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )

        self.emb_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.hidden_size*2)
        )

        self.diff_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
        )


        if self.diffuser_type =='mlp1':
            self.diffuser = nn.Sequential(
                nn.Linear(self.hidden_size*3, self.hidden_size)
        )
        elif self.diffuser_type =='mlp2':
            self.diffuser = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size*2),
            nn.GELU(),
            nn.Linear(self.hidden_size*2, self.hidden_size)
        )


    def forward(self, x, h, step):

        t = self.step_mlp(step)

        try:
            if self.diffuser_type == 'mlp1':
                res = self.diffuser(torch.cat((x, h, t), dim=1))
            elif self.diffuser_type == 'mlp2':
                res = self.diffuser(torch.cat((x, h, t), dim=1))
        except:
            print(x.shape, h.shape, t.shape)   
        return res

    def forward_uncon(self, x, step):
        h = self.none_embedding(torch.tensor([0]).to(self.device))
        h = torch.cat([h.view(1, h.shape[1])]*x.shape[0], dim=0)

        t = self.step_mlp(step)

        if self.diffuser_type == 'mlp1':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
        elif self.diffuser_type == 'mlp2':
            res = self.diffuser(torch.cat((x, h, t), dim=1))
            
        return res

        # return x

    def cacu_x(self, x):
        x = copy.deepcopy(self.item_embeddings[x])
        
        return x

    def cacu_h(self, states, len_states, p):
        #hidden
        inputs_emb = states + self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out)
        h = state_hidden.squeeze()

        # mask?
        B, D = h.shape[0], h.shape[1]
        mask1d = (torch.sign(torch.rand(B) - p) + 1) / 2
        maske1d = mask1d.view(B, 1)
        mask = torch.cat([maske1d] * D, dim=1)
        mask = mask.to(self.device)

        # print(h.device, self.none_embedding(torch.tensor([0]).to(self.device)).device, mask.device)
        h = h * mask + self.none_embedding(torch.tensor([0]).to(self.device)) * (1-mask)
        return h  
    
    # states = condition, item_embeddings = GNN_embed
    def predict(self, states, len_states, diff):
        #hidden
        inputs_emb = states + self.positional_embeddings(torch.arange(self.state_size).to(self.device))
        seq = self.emb_dropout(inputs_emb)
        seq_normalized = self.ln_1(seq)
        mh_attn_out = self.mh_attn(seq_normalized, seq)
        ff_out = self.feed_forward(self.ln_2(mh_attn_out))
        ff_out = self.ln_3(ff_out)
        state_hidden = extract_axis_1(ff_out)
        h = state_hidden.squeeze()
        if len(h.shape) < 2:
            h = h.unsqueeze(0)
        x = diff.sample(self.forward, self.forward_uncon, h)
        test_item_emb = copy.deepcopy(self.item_embeddings)
        scores = torch.matmul(x, test_item_emb.transpose(0, 1))

        return x, scores



# def evaluate(model, test_data, diff, device):
#     eval_data=pd.read_pickle(os.path.join(data_directory, test_data))

#     batch_size = 100
#     evaluated=0
#     total_clicks=1.0
#     total_purchase = 0.0
#     total_reward = [0, 0, 0, 0]
#     hit_clicks=[0,0,0,0]
#     ndcg_clicks=[0,0,0,0]
#     hit_purchase=[0,0,0,0]
#     ndcg_purchase=[0,0,0,0]

#     seq, len_seq, target = list(eval_data['seq'].values), list(eval_data['len_seq'].values), list(eval_data['next'].values)


#     num_total = len(seq)

#     for i in range(num_total // batch_size):
#         seq_b, len_seq_b, target_b = seq[i * batch_size: (i + 1)* batch_size], len_seq[i * batch_size: (i + 1)* batch_size], target[i * batch_size: (i + 1)* batch_size]
#         states = np.array(seq_b)
#         states = torch.LongTensor(states)
#         states = states.to(device)

#         prediction = model.predict(states, np.array(len_seq_b), diff)
#         _, topK = prediction.topk(100, dim=1, largest=True, sorted=True)
#         topK = topK.cpu().detach().numpy()
#         sorted_list2=np.flip(topK,axis=1)
#         sorted_list2 = sorted_list2
#         calculate_hit(sorted_list2,topk,target_b,hit_purchase,ndcg_purchase)

#         total_purchase+=batch_size
 

#     hr_list = []
#     ndcg_list = []
#     print('{:<10s} {:<10s} {:<10s} {:<10s} {:<10s} {:<10s}'.format('HR@'+str(topk[0]), 'NDCG@'+str(topk[0]), 'HR@'+str(topk[1]), 'NDCG@'+str(topk[1]), 'HR@'+str(topk[2]), 'NDCG@'+str(topk[2])))
#     for i in range(len(topk)):
#         hr_purchase=hit_purchase[i]/total_purchase
#         ng_purchase=ndcg_purchase[i]/total_purchase

#         hr_list.append(hr_purchase)
#         ndcg_list.append(ng_purchase[0,0])

#         if i == 1:
#             hr_20 = hr_purchase

#     print('{:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}'.format(hr_list[0], (ndcg_list[0]), hr_list[1], (ndcg_list[1]), hr_list[2], (ndcg_list[2])))

#     return hr_20


# if __name__ == '__main__':

#     # args = parse_args()
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

#     data_directory = './data/' + args.data
#     data_statis = pd.read_pickle(
#         os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing seq_size and item_num
#     seq_size = data_statis['seq_size'][0]  # the length of history to define the seq
#     item_num = data_statis['item_num'][0]  # total number of items
#     topk=[10, 20, 50]

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     timesteps = args.timesteps


#     model = Tenc(args.hidden_factor,item_num, seq_size, args.dropout_rate, args.diffuser_type, device)
#     diff = diffusion(args.timesteps, args.beta_start, args.beta_end, args.w)

#     if args.optimizer == 'adam':
#         optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
#     elif args.optimizer =='adamw':
#         optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
#     elif args.optimizer =='adagrad':
#         optimizer = torch.optim.Adagrad(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
#     elif args.optimizer =='rmsprop':
#         optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)

#     # scheduler = lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1, total_iters=20)
    
#     model.to(device)
#     # optimizer.to(device)

#     train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))

#     total_step=0
#     hr_max = 0
#     best_epoch = 0

#     num_rows=train_data.shape[0]
#     num_batches=int(num_rows/args.batch_size)
#     for i in range(args.epoch):
#         start_time = Time.time()
#         for j in range(num_batches):
#             batch = train_data.sample(n=args.batch_size).to_dict()
#             seq = list(batch['seq'].values())
#             len_seq = list(batch['len_seq'].values())
#             target=list(batch['next'].values())

#             optimizer.zero_grad()
#             seq = torch.LongTensor(seq)
#             len_seq = torch.LongTensor(len_seq)
#             target = torch.LongTensor(target)

#             seq = seq.to(device)
#             target = target.to(device)
#             len_seq = len_seq.to(device)


#             x_start = model.cacu_x(target)

#             h = model.cacu_h(seq, len_seq, args.p)

#             n = torch.randint(0, args.timesteps, (args.batch_size, ), device=device).long()
#             loss, predicted_x = diff.p_losses(model, x_start, h, n, loss_type='l2')

#             loss.backward()
#             optimizer.step()


#         # scheduler.step()
#         if args.report_epoch:
#             if i % 1 == 0:
#                 print("Epoch {:03d}; ".format(i) + 'Train loss: {:.4f}; '.format(loss) + "Time cost: " + Time.strftime(
#                         "%H: %M: %S", Time.gmtime(Time.time()-start_time)))

#             if (i + 1) % 10 == 0:
                
#                 eval_start = Time.time()
#                 print('-------------------------- VAL PHRASE --------------------------')
#                 _ = evaluate(model, 'val_data.df', diff, device)
#                 print('-------------------------- TEST PHRASE -------------------------')
#                 _ = evaluate(model, 'test_data.df', diff, device)
#                 print("Evalution cost: " + Time.strftime("%H: %M: %S", Time.gmtime(Time.time()-eval_start)))
#                 print('----------------------------------------------------------------')




                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

