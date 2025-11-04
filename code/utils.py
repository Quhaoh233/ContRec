import pandas as pd
import numpy as np
import json
import gzip
import torch
import pickle
import math
import torch.nn as nn
from tqdm import tqdm
import sys
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def train_prompt(remap_user, remap_items, item_list, data_name, k, user_token, with_title=True):
    token_format = ''
    user = ''
    for i in range(k):
        dif_token = f'<collaborative_{i}>'
        token_format += dif_token
        user_dif_token = f'<user_collaborative_{i}>'
        user += user_dif_token
        
    indicators = ['<diff>', '<\diff>']
    
    remap_ids = [int(idx) for idx in remap_items]
    titles = np.array(item_list['title'])[remap_ids].astype(str)  # to string
    
    if with_title:
        items = ''
        for title in titles:
            if len(title) > 20:  # truncate the title to prevent from exceed input length!
                title = title[:20]
            items += ' ' + title + ' ' + indicators[0] + token_format + indicators[1] + ','  # example: [Apple Watch <diff><semantic_left><semantic_right><collaborative_left><collaborative_right><\diff>]
    else:
        per_item = ' item_' + indicators[0] + token_format + indicators[1] + ','  # should we input the title here?
        items = per_item * len(remap_items)
    
    example_input = 'Given the following purchase history: [item side information]. I wonder what the user will like. Can you help me decide?'
    example_response = 'The interaction history shows that the user might like [item_title].'
    
    role_play = f'You are the expert in {data_name} products, entrusted with the responsibility of recommending the perfect products to our users.'
    few_shot = f'\nHere is an example format for recommendations: ### Input: {example_input} \n ### Response: {example_response}'
    prefix = '\nNow, please provide your recommendations based on the following content:\n'
    suffix = " Taking into account the user's preferences and purchase history, I would suggest that"
    
    if user_token:
        user = indicators[0] + user + indicators[1]
    else: user = ''
    prompts = dict()
    prompts[0] = f'### Input: Given the following purchase history:{items}. Predict the user{user} preferences. ### Response:'
    prompts[1] = f'### Input: I find the purchase history list:{items}. I wonder what the user{user} will like. Can you help me decide? ### Response:'
    prompts[2] = f'### Input: Considering the user{user} has interacted with{items}. What are the user preferences? ### Response:'
    prompts[3] = f'### Input: According to what items the user{user} has purchased:{items}. Can you describe the user preferences? ### Response:'
    prompts[4] = f"### Input: By analyzing the user{user}'s purchase of{items}, what are the expected preferences of the user? ### Response:"
    prompts[5] = f"### Input: Given the user{user}'s previous interactions with the{items}, what are the user preferences? ### Response:"
    prompts[6] = f"### Input: Taking into account the user{user}'s engagement with the{items}, what are the user potential interests? ### Response:"
    prompts[7] = f"### Input: In light of the user{user}'s interactions with the{items}, what might the user be interested in? ### Response:"
    prompts[8] = f"### Input: Considering the user{user}'s past interactions with the{items}, what are the user likely preferences? ### Response:"
    prompts[9] = f"### Input: With the user{user}'s history of engagement with the{items}, what would the user be inclined to like? ### Response:"
    idx = int(np.random.randint(len(prompts), size=1))
    
    structured_prompt = role_play + few_shot + prefix + prompts[idx] + suffix
    return structured_prompt


def structure_response(target, item_list):
    title = str(np.array(item_list['title'])[int(target)])
    category = str(np.array(item_list['category'])[int(target)])
    response =f"The user might lean towards ({category}) products based on an analysis of their purchase history, like {title}."
    return response


def structure_response_cf(targets, item_list):
    categories = ", ".join([str(np.array(item_list['category'])[int(target)]) for target in targets])
    response =f"The user might lean towards ({categories}) products based on an analysis of their purchase history."

    return response


def sentence_encoding(data_name):  # encoding sentences into embeddings
    data_dir = '../data/' + data_name
    item_list = pd.read_csv(data_dir+'/item_list.txt', header=0, index_col=None, sep=' ')
    metas = getDF(data_dir+'/meta_'+data_name.title()+'.json.gz')
    
    template = 'Here is the detailed information about the item. Title: {}. Description: {}. Brand: {}. Categories: {}.'  # structure item side information
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    messages = []
    for org_id, remap_id in tqdm(zip(item_list['org_id'], item_list['remap_id']), total=len(item_list), desc='Encoding semantic information of items ...'):
        desc = metas.loc[metas['asin'] == org_id]['description'].iloc[0]
        title = metas.loc[metas['asin'] == org_id]['title'].iloc[0]
        brand = metas.loc[metas['asin'] == org_id]['brand'].iloc[0]
        categories = metas.loc[metas['asin'] == org_id]['categories'].iloc[0]
        messages.append(template.format(title, desc, brand, categories))

    embeddings = torch.Tensor(model.encode(messages))
    torch.save(embeddings, '../src/'+data_name+'/sentence_embeddings.pt')
    
    
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

    def should_stop(self):
        return self.early_stop


class ResidualBlock(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x) + x  # Residual connection
    

class MLPEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, k, mask_ratio=0.1):
        super(MLPEncoder, self).__init__()
        self.blocks = nn.ModuleList()
        for _ in range(k):
            block = nn.Sequential(
                # you can stacking residualblocks here
                ResidualBlock([
                    nn.ReLU(),
                    nn.Dropout(p=0.2),
                    nn.Linear(in_dim, in_dim)
                ]),
                # output block
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(in_dim, out_dim),     
                nn.ReLU()      
            )
            self.blocks.append(block)
            
        # mask & position
        self.pos = nn.Embedding(1, in_dim)
        self.pos.weight.data.uniform_(-1.0 / in_dim, 1.0 / in_dim)  # init
        self.mask_ratio = mask_ratio

    def forward(self, x):
        mask = x[0, :].bernoulli_(self.mask_ratio).bool()
        x = torch.masked_fill(x, mask, 0)
        x += self.pos.weight
        return torch.stack([block(x) for block in self.blocks])


class MLPDecoder(nn.Module):
    def __init__(self, length_in, length_out, dim_in, dim_out):
        super(MLPDecoder, self).__init__()

        self.dim_block = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(dim_in, dim_out),     
            nn.ReLU()      
        )
        
        self.length_block = nn.Sequential(
            # you can stacking residualblocks here
            ResidualBlock([
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(length_in, length_in)
            ]),
            # output block
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(length_in, length_out),     
            nn.ReLU()      
        )
        
    def forward(self, x):  # x.shape = [batch, len, dim]
        x = self.dim_block(x)
        x = x.transpose(1, 2)  # process on the len dimension
        x = self.length_block(x).transpose(1, 2)
        return x


# ------------------------ metrics for sequential recommendation ----------------------------------
def get_metrics(targets, results, k):
    targets = [int(target) for target in targets]
    hits, batch = hit_at_k(targets, results, k)
    ndcg, _ = ndcg_at_k(targets, results, k)
    metrics = [hits, ndcg.item()]
    return metrics, batch


def hit_at_k(labels, results, k):
    '''
    labels.shape = [batch]
    results.shape = [batch, item_num]
    '''
    hit = 0.0
    batch = results.shape[0]
    for i in range(batch):
        res = results[i, :k]
        label = labels[i]
        if label in res:
            hit += 1
    return hit, batch


def ndcg_at_k(labels, results, k):
    """
    Since we apply leave-one-out, each user only have one ground truth item, so the idcg would be 1.0
    """
    ndcg = 0.0
    batch = results.shape[0]
    for i in range(batch):
        res = results[i, :k]
        label = labels[i]
        one_ndcg = 0.0
        rel = torch.where(res == label, 1, 0)
        for j in range(len(rel)):
            one_ndcg += rel[j] / math.log(j+2,2)
        ndcg += one_ndcg
    return ndcg, batch


def similarity_score(predicts, item_emb, item_id):
	'''
	predicts.shape = [batch, emb]
	item_emb.shape = [item_num, emb]
	items.shape = [batch, num]
	'''
	score = []
	cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
	batch = predicts.shape[0]
	for i in range(batch):
		items = item_id[i].split(" ")
		items = [int(item) for item in items]
		temp = cos(predicts[i, :].unsqueeze(0), item_emb)
		temp[items] = 0  # remove the impact of interacted items
		score.append(temp)
	score = torch.stack(score, dim=0)
	return score


def MSE_distance(predicts, item_emb):
    '''
    predicts.shape = [batch, emb]
    item_emb.shape = [item_num, emb]
    '''
    score = []
    batch, dim = predicts.shape
    item_num, _ = item_emb.shape
    for i in range(batch):
		# temp = predicts[i, :].unsqueeze(0).expand(item_num, -1)  # [item_num, emb]
		# dis = (temp - item_emb).pow(2).sum(1).sqrt()
        dis = torch.cdist(predicts[i, :].unsqueeze(0), item_emb)  # Calculate the pairwise Euclidean distances between the vector and vectors_set
        score.append(dis)
    score = torch.stack(score, dim=0).squeeze()
    return score


def calculate_text_score(text1, text2, pi=0.1, sh=0.9):
    vectorizer = CountVectorizer()
    corpus = text1 + text2
    vectors = vectorizer.fit_transform(corpus)
    corpus1 = vectors[:len(text1)]
    corpus2 = vectors[len(text1):]
    similarity = cosine_similarity(corpus1, corpus2)  # similarity matrix
    text_score = np.where(similarity > sh, 0, pi)
    return text_score


# ------------------- Evaluation Solution 2 ----------------------------
def calculate_hit(sorted_list,topk,true_items,hit_purchase,ndcg_purchase):
    for i in range(len(topk)):  # i = @K wise
        rec_list = sorted_list[:, -topk[i]:]
        for j in range(len(true_items)):
            if true_items[j] in rec_list[j]:  # j = batch wise
                rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])
                hit_purchase[i] += 1.0
                ndcg_purchase[i] += 1.0 / np.log2(rank + 1)

    return hit_purchase, ndcg_purchase


# ------------------- metrics for collaborative filtering -------------------------
# code source: https://github.com/gusye1234/LightGCN-PyTorch
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data: groundTrue items for each user in one batch, e.g., [[3, 7, 2], [1, 0], ...], shape list(test_batch, .)
    pred_data: the sorted version based on predicted scores for all items
    r: results of prediction, a list of booleans indicating, for each item in predictTopK, whether it is in the user's ground-truth items.
    k: top-k
    topks: a list of ks, e.g., [10, 20, 30]
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


# calculate r
def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):  # groundTrue items for each user in one batch, e.g., [[3, 7, 2], [1, 0], ...]
        groundTrue = test_data[i]  # list of ground truth
        predictTopK = pred_data[i]  # the sorted version for all items? it seems that here is no "topk" operation.
        pred = list(map(lambda x: x in groundTrue, predictTopK))  # producesa list of booleans indicating, for each item in predictTopK, whether it is in the user's ground-truth items.
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')


def eval_cf(sorted_items, groundTrue, topks):
    sorted_items = [[str(element.item()) for element in row] for row in sorted_items]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in topks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}