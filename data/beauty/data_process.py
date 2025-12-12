import pandas as pd 
import numpy as np 
import sys 
import json
import copy
from tqdm import tqdm
import gzip
import ast


# configurations
dataset = 'beauty'
path = 'Beauty'



# read 5-core data
rates = []
time = []
user = []
item = []
with gzip.open(f"reviews_{path}_5.json.gz", "rt", encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        rates.append(obj['overall'])
        time.append(obj['unixReviewTime'])
        user.append(obj['reviewerID'])
        item.append(obj['asin'])
            # review = obj['reviewText']
            # summary = obj['summary']
df = pd.DataFrame({'rates': rates, 'time': time, 'user': user, 'item': item})
print(len(df['rates']))

# drop the duplicate and negative reviews
df = df[df['rates']> 3]
df.drop_duplicates(inplace=True)

# Drop users whose interactions are less than 5
user_list = df['user'].unique().tolist()
item_list = df['item'].unique().tolist()
for i, u in tqdm(enumerate(user_list), total=len(user_list), desc="Drop users whose interactions are less than 5"):
    interactions = df[df['user'] == u]
    if len(interactions) < 5:
        df.drop(df[df['user'] == u].index)
df = df.reset_index(drop=True)

# remapping users and items
user_list = df['user'].unique().tolist()
item_list = df['item'].unique().tolist()
remap_user_list = np.array(np.arange(len(user_list)))
remap_item_list = np.array(np.arange(len(item_list)))
print('numbers of users =', len(user_list))
print('numbers of items =', len(item_list))
print('numbers of interactions =', len(df['user']))

# output user_list and item_list
df_user_list = pd.DataFrame({'org_id': user_list, 'remap_id': remap_user_list})
category_list = copy.deepcopy(item_list)
title_list = copy.deepcopy(item_list)
desc_list = copy.deepcopy(item_list)

g = gzip.open(f'meta_{path}.json.gz', 'rb')
for l in g:
    line = l
    obj = line.decode('utf-8')
    obj = ast.literal_eval(obj)
    try:
        category = obj['categories']
        title = obj['title']
        item_id = obj['asin']
        desc = obj['description']
        if item_id in item_list:
            idx = item_list.index(item_id)
            category_list[idx] = category
            title_list[idx] = title
            desc_list[idx] = desc

    except:
        pass
df_item_list = pd.DataFrame({'org_id': item_list, 'remap_id': remap_item_list, 'title': title_list, 'category': category_list, 'desc': desc_list})

# with gzip.open(f"meta_{path}.json.gz", mode="rt") as f:
#     for line in f:
#         obj = json.loads(line)
#         category = ", ".join(obj['category']) + ", " + obj['brand'].strip('  ')
#         category.strip('\n').strip('</span></span></span>')
#         title = obj['title'].strip('\n').strip('</span></span></span>')
#         item_id = obj['asin']
#         if item_id in item_list:
#             idx = item_list.index(item_id)
#             category_list[idx] = category
#             title_list[idx] = title
# df_item_list = pd.DataFrame({'org_id': item_list, 'remap_id': remap_item_list, 'title': title_list, 'category': category_list})

# output train.txt and test.txt
train = []
test = []
for i, u in tqdm(enumerate(user_list), total=len(user_list), desc='create train.txt and test.txt'):
    user_interactions = df[df['user'] == u]
    items = np.array(user_interactions['item'])
    times = np.array(user_interactions['time'])
    indices = np.argsort(times)
    items = items[indices]
    row = [str(i)]
    if len(items) >= 5:  # drop users whose interactions are less than 5
        for v in items:
            remap_item_id = item_list.index(v)
            row.append(str(remap_item_id))
        train_row = " ".join(row[:-1])
        test_row = " ".join(row)
        train.append(train_row)
        test.append(test_row)


# output files
# train
with open(f"train.txt", "w") as f:
    for line in train:
        print(line, file=f)

# test
with open(f"test.txt", "w") as f:
    for line in test:
        print(line, file=f)

df_user_list.to_csv(f"user_list.txt", sep=" ", encoding="utf-8", index=None)
df_item_list.to_csv(f"item_list.txt", sep=" ", encoding="utf-8", index=None)

print('The data processing has been completed.')