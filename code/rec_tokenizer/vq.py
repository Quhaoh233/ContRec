# This code is modified based on TokenRec. code source: https://github.com/Quhaoh233/TokenRec, paper link: https://arxiv.org/abs/2406.10450
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import time
import sys
from kmeans_pytorch import kmeans
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# VQ-VAE
class VQ(nn.Module):
    def __init__(self, input_dim, dim, n_embedding, m_book):
        super(VQ, self).__init__()
        self.m_book = m_book
        self.encoders = nn.ModuleList()
        self.codebooks = nn.ModuleList()
        for m in range(m_book):
            codebook = nn.Embedding(n_embedding, dim)
            codebook.weight.data.uniform_(-1.0 / n_embedding, 1.0 / n_embedding)
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
            self.codebooks.append(codebook)
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
        b, e = x.shape
        patch = int(e/self.m_book)
        
        # encode    
        res_list = []
        ce_list = []
        for m in range(self.m_book):
            # quantization
            ze = self.encoders[m](x)
            embedding = self.codebooks[m].weight    
            N, C = ze.shape  # ze: [batch, dim]
            K, _ = embedding.shape  # embedding [n_codewords, dim]
            ze_broadcast = ze.reshape(N, 1, C)
            embedding_broadcast = embedding.reshape(1, K, C)
            distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
            nearest_neighbor = torch.argmin(distance, 1)
            ce = self.codebooks[m](nearest_neighbor)
            ce_list.append(ce)
            res_list.append(ze)
            ce = ze + (ce - ze).detach()  # straight through loss
                        
        zq = torch.sum(torch.stack(ce_list, dim=0), dim=0)
        decoder_input = zq

        # decode
        x_hat = self.decoder(decoder_input)
        return x_hat, res_list, ce_list
    
    def valid(self, x):  # shape = [batch, emb]
        # encode    
        res_list = []
        ce_list = []
        for m in range(self.m_book):
            ze = self.encoders[m](x)
            embedding = self.codebooks[m].weight.data   
            N, C = ze.shape  # ze: [batch, dim]
            K, _ = embedding.shape  # embedding [n_codewords, dim]
            ze_broadcast = ze.reshape(N, 1, C)
            embedding_broadcast = embedding.reshape(1, K, C)
            distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
            nearest_neighbor = torch.argmin(distance, 1)
            ce = self.codebooks[m](nearest_neighbor)
            ce_list.append(ce)
            res_list.append(ze)
            
        zq = torch.sum(torch.stack(ce_list, dim=0), dim=0)
        decoder_input = zq

        # decode
        x_hat = self.decoder(decoder_input)
        return x_hat, res_list, ce_list
    
    def encode(self, x):
        nearest_neighbor_list = []
        res_list = []
        ce_list = []
        for m in range(self.m_book):
            ze = self.encoders[m](x)
            embedding = self.codebooks[m].weight.data  
            N, C = ze.shape  # ze: [batch, dim]
            K, _ = embedding.shape  # embedding [n_codewords, dim]
            ze_broadcast = ze.reshape(N, 1, C)
            embedding_broadcast = embedding.reshape(1, K, C)
            distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
            nearest_neighbor = torch.argmin(distance, 1)
            ce = self.codebooks[m](nearest_neighbor)
            ce_list.append(ce)
            res_list.append(ze)
            nearest_neighbor_list.append(nearest_neighbor)
            
        codeword_idx = torch.stack(nearest_neighbor_list, dim=0).transpose(0, 1)  # shape = [batch_size, n_codebook]
        return codeword_idx


# parallel structure
class MQ(nn.Module):
    def __init__(self, input_dim, dim, n_embedding, m_book, mask_ratio=0.2):
        super(MQ, self).__init__()
        self.m_book = m_book
        self.encoders = nn.ModuleList()
        self.codebooks = nn.ModuleList()
        for m in range(m_book):
            codebook = nn.Embedding(n_embedding, dim)
            codebook.weight.data.uniform_(-1.0 / n_embedding, 1.0 / n_embedding)
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
            self.codebooks.append(codebook)
            self.encoders.append(encoder)
        
        self.pos = nn.Embedding(1, input_dim)
        self.pos.weight.data.uniform_(-1.0 / n_embedding, 1.0 / n_embedding)
        self.mask_ratio = mask_ratio
            
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
        b, e = x.shape
        patch = int(e/self.m_book)
        
        # encode    
        res_list = []
        ce_list = []
        for m in range(self.m_book):
            # mask & position
            mask = x[0, :].bernoulli_(self.mask_ratio).bool()
            mask[:m*patch] = 0  # release the specific masked part
            mask[(m+1)*patch-1:] = 0
            x = torch.masked_fill(x, mask, 0)
            x += self.pos.weight
            
            # quantization
            ze = self.encoders[m](x)
            embedding = self.codebooks[m].weight    
            N, C = ze.shape  # ze: [batch, dim]
            K, _ = embedding.shape  # embedding [n_codewords, dim]
            ze_broadcast = ze.reshape(N, 1, C)
            embedding_broadcast = embedding.reshape(1, K, C)
            distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
            nearest_neighbor = torch.argmin(distance, 1)
            ce = self.codebooks[m](nearest_neighbor)
            ce_list.append(ce)
            res_list.append(ze)
            ce = ze + (ce - ze).detach()  # straight through loss
                        
        zq = torch.sum(torch.stack(ce_list, dim=0), dim=0)
        decoder_input = zq

        # decode
        x_hat = self.decoder(decoder_input)
        return x_hat, res_list, ce_list
    
    def valid(self, x):  # shape = [batch, emb]
        x += self.pos.weight.data
        
        # encode    
        res_list = []
        ce_list = []
        for m in range(self.m_book):
            ze = self.encoders[m](x)
            embedding = self.codebooks[m].weight.data   
            N, C = ze.shape  # ze: [batch, dim]
            K, _ = embedding.shape  # embedding [n_codewords, dim]
            ze_broadcast = ze.reshape(N, 1, C)
            embedding_broadcast = embedding.reshape(1, K, C)
            distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
            nearest_neighbor = torch.argmin(distance, 1)
            ce = self.codebooks[m](nearest_neighbor)
            ce_list.append(ce)
            res_list.append(ze)
            
        zq = torch.sum(torch.stack(ce_list, dim=0), dim=0)
        decoder_input = zq

        # decode
        x_hat = self.decoder(decoder_input)
        return x_hat, res_list, ce_list
    
    def encode(self, x):
        x += self.pos.weight.data
        nearest_neighbor_list = []
        res_list = []
        ce_list = []
        for m in range(self.m_book):
            ze = self.encoders[m](x)
            embedding = self.codebooks[m].weight.data  
            N, C = ze.shape  # ze: [batch, dim]
            K, _ = embedding.shape  # embedding [n_codewords, dim]
            ze_broadcast = ze.reshape(N, 1, C)
            embedding_broadcast = embedding.reshape(1, K, C)
            distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
            nearest_neighbor = torch.argmin(distance, 1)
            ce = self.codebooks[m](nearest_neighbor)
            ce_list.append(ce)
            res_list.append(ze)
            nearest_neighbor_list.append(nearest_neighbor)
            
        codeword_idx = torch.stack(nearest_neighbor_list, dim=0).transpose(0, 1)  # shape = [batch_size, n_codebook]
        return codeword_idx
    
# serial structure
class ResidualVQVAE(nn.Module):
    def __init__(self, input_dim, dim, n_embedding, m_book):
        super(ResidualVQVAE, self).__init__()
        self.encoder = nn.Sequential(
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

        self.m_book = m_book
        self.codebooks = nn.ModuleList()
        for m in range(m_book):
            codebook = nn.Embedding(n_embedding, dim)
            codebook.weight.data.uniform_(-1.0 / n_embedding, 1.0 / n_embedding)
            self.codebooks.append(codebook)
            
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

    def forward(self, x):
        # encode
        res_list = []
        ce_list = []
        for m in range(self.m_book):
            if m == 0:
                ze = self.encoder(x)
                embedding = self.codebooks[0].weight
                N, C = ze.shape  # ze: [batch, dim]
                K, _ = embedding.shape  # embedding [n_codewords, dim]
                ze_broadcast = ze.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebooks[0](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(ze)
            else:
                res = res_list[m-1] - ce_list[m-1]
                embedding = self.codebooks[m].weight  # It should be learnable!
                N, C = res.shape  # ze: [batch, dim]
                K, _ = embedding.shape  # embedding [n_codewords, dim]
                res_broadcast = res.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - res_broadcast)**2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebooks[m](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(res)
        zq = torch.sum(torch.stack(ce_list, dim=0), dim=0)
        decoder_input = ze + (zq - ze).detach()

        # decode
        x_hat = self.decoder(decoder_input)
        return x_hat, res_list, ce_list

    def valid(self, x):
        # encode
        res_list = []
        ce_list = []
        for m in range(self.m_book):
            if m == 0:
                ze = self.encoder(x)
                embedding = self.codebooks[0].weight.data
                N, C = ze.shape  # ze: [batch, dim]
                K, _ = embedding.shape  # embedding [n_codewords, dim]
                ze_broadcast = ze.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebooks[0](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(ze)
            else:
                res = res_list[m-1] - ce_list[m-1]
                embedding = self.codebooks[m].weight.data  # It should be learnable!
                N, C = res.shape  # ze: [batch, dim]
                K, _ = embedding.shape  # embedding [n_codewords, dim]
                res_broadcast = res.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - res_broadcast)**2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebooks[m](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(res)
        zq = torch.sum(torch.stack(ce_list, dim=0), dim=0)
        decoder_input = ze + (zq - ze).detach()

        # decode
        x_hat = self.decoder(decoder_input)
        return x_hat, res_list, ce_list

    def encode(self, x):
        nearest_neighbor_list = []
        res_list = []
        ce_list = []
        for m in range(self.m_book):
            if m == 0:
                ze = self.encoder(x)
                embedding = self.codebooks[0].weight
                N, C = ze.shape  # ze: [batch, dim]
                K, _ = embedding.shape  # embedding [n_codewords, dim]
                ze_broadcast = ze.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - ze_broadcast)**2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebooks[0](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(ze)
                nearest_neighbor_list.append(nearest_neighbor)
            else:
                res = res_list[m-1] - ce_list[m-1]
                embedding = self.codebooks[m].weight
                N, C = res.shape  # ze: [batch, dim]
                K, _ = embedding.shape  # embedding [n_codewords, dim]
                res_broadcast = res.reshape(N, 1, C)
                embedding_broadcast = embedding.reshape(1, K, C)
                distance = torch.sum((embedding_broadcast - res_broadcast)**2, 2)
                nearest_neighbor = torch.argmin(distance, 1)
                ce = self.codebooks[m](nearest_neighbor)
                ce_list.append(ce)
                res_list.append(res)
                nearest_neighbor_list.append(nearest_neighbor)
        zq = torch.sum(torch.stack(ce_list, dim=0), dim=0)
        codeword_idx = torch.stack(nearest_neighbor_list, dim=0).transpose(0, 1)  # shape = [batch_size, n_codebook]
        return codeword_idx

# dataset
class VQDataset(Dataset):
    def __init__(self, embs):
        super().__init__()
        self.embs = embs

    def __len__(self):
        return len(self.embs)

    def __getitem__(self, index: int):
        emb = self.embs[index]
        return emb

def user_codebook_to_str(vq_id):
	user_head = ['a', 'b', 'c', 'd', 'e']
	id_num = vq_id.shape[0]
	codebook_num = vq_id.shape[1]
	sample = []
	for i in range(id_num):
		temp = ['user_']
		for j in range(codebook_num):
			token = "".join(["<", user_head[j], '-', str(vq_id[i, j].item()), '>'])
			temp.append(token)
		temp = "".join(temp)
		sample.append(temp)
	sample = " ".join(sample)
	return sample

def item_codebook_to_str(vq_id):
	id_num = vq_id.shape[0]
	codebook_num = vq_id.shape[1]
	sample = []
	for i in range(id_num):
		temp = ['item_']
		for j in range(codebook_num):
			token = "".join(["<", str(j), '-', str(vq_id[i, j].item()), '>'])
			temp.append(token)
		temp = "".join(temp)
		sample.append(temp)
	sample = " ".join(sample)
	return sample

class RecDataset(Dataset):
    def __init__(self, data, user_emb, user_vq, item_emb, item_vq, max_item_num=1e10):
        super().__init__()
        self.data = data
        self.max_item_num = max_item_num
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.user_vq = user_vq
        self.item_vq = item_vq

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        # orginal idx
        sample = self.data[index]
        sample = sample.split()  # split the list
        user = sample[0]
        item_list = sample[1:]
        if len(item_list) > self.max_item_num:
            item_list = item_list[:self.max_item_num]
        
        user_id = int(user)
        item_idx = [int(x) for x in item_list]

        # codebook idx
        user_emb = self.user_emb[user_id].unsqueeze(0)  # [1, n_emb]
        item_emb = self.item_emb[item_idx]
        with torch.no_grad():
            user_vq_id = self.user_vq.encode(user_emb)  # [1, 3]
            item_vq_idx = self.item_vq.encode(item_emb)  # [n_item, 3]

        user_cb_id = user_codebook_to_str(user_vq_id)
        item_cb_id = item_codebook_to_str(item_vq_idx)
        return user_id, user_cb_id, item_cb_id
    
# trainer
def vqtrainer(model, model_name, device, co_emb, n_embedding, kmean_epoch=50, m_book=2, valid_ratio=0.2, batch_size=512, lr=1e-3, n_epochs=1000, l_w_embedding=1, l_w_commitment=0.25):
    # random sampling
    idx = torch.rand_like(co_emb[:, 0])
    sh = torch.quantile(idx, valid_ratio)
    valid_emb = []
    train_emb = []
    train_loss_list = []
    vaild_loss_list = []
    for n in range(co_emb.shape[0]):
        if idx[n] <= sh:
            valid_emb.append(co_emb[n])
        else:
            train_emb.append(co_emb[n])
    valid_emb = torch.stack(valid_emb, dim=0)
    train_emb = torch.stack(train_emb, dim=0)
    train_dataset = VQDataset(train_emb)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_dataset = VQDataset(valid_emb)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_emb.shape[0], shuffle=True)
    model.to(device)
    # print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.001)
    mse_loss = nn.MSELoss()
    
    tic = time.time()
    valid_loss = 10
    early_stop = 20
    counter = 0
    for e in range(n_epochs):
        total_loss = 0
        model.train()
        for i, x in enumerate(train_dataloader):
            current_batch_size = x.shape[0]
            x = x.to(device)
            x_hat, res_dict, ce_dict = model(x)
            # # codebook kmeans initialization
            # if e == 0 and i == 0:
            # # if e in [0, 10, 20] and i == 0:
            #     for m in range(m_book):
            #         ids, centers = kmeans(res_dict[m], n_embedding, distance='euclidean', device=device)
            #         model.codebooks[m].weight.data = centers.to(device)
            l_reconstruct = mse_loss(x_hat, x) # reconstruction_loss / data_variance?
            loss = l_reconstruct
            for m in range(m_book):
                l_embedding = mse_loss(res_dict[m].detach(), ce_dict[m])
                # l_commitment = mse_loss(res_dict[m], ce_dict[m].detach())
                loss = loss + l_w_embedding * l_embedding # + l_w_commitment * l_commitment

            mse_l = mse_loss(x_hat, x)   # !
            optimizer.zero_grad()
            mse_l.backward()  # !
            optimizer.step()
            record_loss = torch.nn.functional.mse_loss(x_hat, x)
            total_loss += record_loss.item() * current_batch_size
        total_loss /= len(train_dataloader.dataset)
        
        # eval and print
        if e % 1 == 0:
            model.eval()
            epoch_loss = 0
            for x in valid_dataloader:
                current_batch_size = x.shape[0]
                x = x.to(device)
                x_hat, res_dict, ce_dict = model.valid(x)
                l_reconstruct = mse_loss(x_hat, x)
                loss = l_reconstruct
                for m in range(m_book):
                    l_embedding = mse_loss(res_dict[m], ce_dict[m])
                    l_commitment = mse_loss(res_dict[m], ce_dict[m])
                    loss += l_w_embedding * l_embedding + l_w_commitment * l_commitment
                record_loss = torch.nn.functional.mse_loss(x_hat, x)
                epoch_loss += record_loss.item() * current_batch_size
            epoch_loss /= len(valid_dataloader.dataset)

            if epoch_loss < valid_loss:
                # print('Pass the validation, Save the MQ model.')
                valid_loss = epoch_loss
                torch.save(model.state_dict(), '../ckpt/tokenizer/' + model_name + '.pth')
                counter = 0
            else:
                counter += 1
            toc = time.time()
            train_loss_list.append(total_loss)
            vaild_loss_list.append(valid_loss)
            print(f'VQ: epoch {e} train_loss: {total_loss} valid_loss: {valid_loss} elapsed {(toc - tic):.2f}s')
         
        # # early stop
        # if counter >= early_stop:
        #     print("Early Stop.")
        #     break
    loss_logs = pd.DataFrame({'train':train_loss_list, 'valid': vaild_loss_list})
    loss_logs.to_csv(f'logs/{model_name}_losses.csv')

# learning
def learning(args):
    # hype-params
    lgn_dim = args.rec_dim
    codebook_dim = args.latent_dim  # LLM token dimension
    device = torch.device("cuda:0" if True and torch.cuda.is_available() else "cpu")
    data_name = args.dataset
    lgn_name = 'lgn-'+ data_name + '-' + str(lgn_dim)
    vq_name = args.discrete_tokenizer_model + '-' + lgn_name
    print('Process: VQ is working:', vq_name)

    # read collaborative embeddings
    LightGCN = torch.load(f'../src/{data_name}/{lgn_name}.pth.tar')
    user_emb = LightGCN['embedding_user.weight']  # requires_grad = False
    item_emb = LightGCN['embedding_item.weight']  # requires_grad = False
    print('total number of items:', item_emb.shape[0])
    print('total number of users:', user_emb.shape[0])

    # read textual embeddings


    # codebook initial
    if args.discrete_tokenizer_model == 'VQ':
        user_vq = VQ(input_dim=user_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token, m_book=args.n_book)
        item_vq = VQ(input_dim=item_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token, m_book=args.n_book)          
    if args.discrete_tokenizer_model == 'RQ':
        user_vq = ResidualVQVAE(input_dim=user_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token, m_book=args.n_book)
        item_vq = ResidualVQVAE(input_dim=item_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token, m_book=args.n_book)
    elif args.discrete_tokenizer_model == 'MQ':
        user_vq = MQ(input_dim=user_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token, m_book=args.n_book)
        item_vq = MQ(input_dim=item_emb.shape[1], dim=codebook_dim, n_embedding=args.n_token, m_book=args.n_book)   
    else:
        NotImplementedError

    # ---------------------- training -------------------------------
    user_test_size = int(user_emb.shape[0] * 0.1)
    item_test_size = int(item_emb.shape[0] * 0.1)
    torch.manual_seed(2025)
    torch.cuda.manual_seed(2025)
    user_indices = torch.randperm(user_emb.shape[0])
    item_indices = torch.randperm(item_emb.shape[0])


    # item vq
    item_vq_name = 'item-' + vq_name
    vqtrainer(item_vq, item_vq_name, device, item_emb[item_indices[item_test_size:]], n_embedding=args.n_token, m_book=args.n_book)
    item_vq.load_state_dict(torch.load(f'../ckpt/tokenizer/{item_vq_name}.pth'))
    item_vq.to(device)

    # # plot
    # codewords = torch.cat([item_vq.codebooks[0].weight.data, item_vq.codebooks[1].weight.data, item_vq.codebooks[2].weight.data], dim=0)
    # tsne = TSNE(n_components=2)
    # embedded_data = tsne.fit_transform(codewords.cpu())
    # plt.scatter(embedded_data[:, 0], embedded_data[:, 1])
    # plt.title('VQ-VAE')
    # plt.tight_layout()
    # plt.savefig('rqvae.png')
    # sys.exit()

    # user vq
    user_vq_name = 'user-' + vq_name
    vqtrainer(user_vq, user_vq_name, device, user_emb[user_indices[user_test_size:]], n_embedding=args.n_token, m_book=args.n_book)
    user_vq.load_state_dict(torch.load(f'../ckpt/tokenizer/{user_vq_name}.pth'))
    user_vq.to(device)

    # -------------------- reconstruction ---------------------------
    user_test_loader = DataLoader(VQDataset(user_emb[user_indices[:user_test_size]]), batch_size=user_test_size)
    item_test_loader = DataLoader(VQDataset(item_emb[item_indices[:item_test_size]]), batch_size=item_test_size)
    item_vq.to(device)
    for i, x in enumerate(item_test_loader):
        x_hat, _, _ = item_vq.valid(x.to(device))
        dis  = torch.nn.functional.mse_loss(x_hat, x)
        print('reconstruction metric =', dis.item())
        sys.exit()
    
    # -------------------- output -------------------------
    file_path = f'../data/{data_name}/train.txt'
    with open(file_path, 'r') as f:
        data = f.readlines()
    
    item_vq.eval()
    user_vq.eval()
    train_rec_dataset = RecDataset(data, user_emb, user_vq, item_emb, item_vq)
    train_rec_loader = DataLoader(train_rec_dataset, batch_size=256, shuffle=False)

    item_list = []
    user_list = []
    for i, sample in enumerate(train_rec_loader):
        user_id, user_cb_id, item_cb_id = sample
        user_list += user_cb_id
        item_list += item_cb_id

    data = {'user_cb_id': user_list, "item_cb_id": item_list}
    df = pd.DataFrame(data)
    df.to_csv(f'../data/{data_name}/codebook_train.txt', index=None)


    file_path = f'../data/{data_name}/test.txt'
    with open(file_path, 'r') as f:
        data = f.readlines()
    test_rec_dataset = RecDataset(data, user_emb, user_vq, item_emb, item_vq)
    test_rec_loader = DataLoader(test_rec_dataset, batch_size=256, shuffle=False)

    item_list = []
    user_list = []
    for i, sample in enumerate(test_rec_loader):
        user_id, user_cb_id, item_cb_id = sample
        user_list += user_cb_id
        item_list += item_cb_id

    data = {'user_cb_id': user_list, "item_cb_id": item_list}
    df = pd.DataFrame(data)
    df.to_csv(f'../data/{data_name}/codebook_test.txt', index=None)
