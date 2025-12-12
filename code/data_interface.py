import pytorch_lightning as pl
import torch.utils.data as data
import utils
import pandas as pd
import random
import sys


class DataInterface(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.task = args.task
        self.data_name = args.dataset
        self.batch_size = args.batch_size
        self.k = args.k
        self.user_token = args.user_token
        self.shuffle = args.shuffle
        self.load_meta_data()

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        if self.task == 'leave-one-out':
            if stage == 'fit':
                self.train = LeaveOneOut(self.train_data, self.item_list, self.data_name, self.k, user_token=self.user_token, shuffle=self.shuffle, mode='train')  # , mode='train'
                self.valid = LeaveOneOut(self.train_data, self.item_list, self.data_name, self.k, user_token=self.user_token, shuffle=self.shuffle)
            if stage == 'test':
                self.test = LeaveOneOut(self.test_data, self.item_list, self.data_name, self.k, user_token=self.user_token, shuffle=self.shuffle)
            if stage == 'predict':
                self.predict = LeaveOneOut(self.test_data, self.item_list, self.data_name, self.k, user_token=self.user_token, shuffle=self.shuffle)
                
        elif self.task == 'random-select':
            if stage == 'fit':
                self.train = RandomSelect(self.train_data, self.item_list, self.data_name, self.k, user_token=self.user_token)  # , mode='train'
                self.valid = RandomSelect(self.valid_data, self.item_list, self.data_name, self.k, user_token=self.user_token)
            if stage == 'test':
                self.test = RandomSelect(self.test_data, self.item_list, self.data_name, self.k, user_token=self.user_token)
            if stage == 'predict':
                self.predict = RandomSelect(self.test_data, self.item_list, self.data_name, self.k, user_token=self.user_token)
        else:
            NotImplementedError


    def train_dataloader(self):
        return data.DataLoader(self.train, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return data.DataLoader(self.valid, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return data.DataLoader(self.test, batch_size=self.batch_size)
    
    def load_meta_data(self):
        data_dir = '../data/' + self.data_name
        self.train_data = pd.read_csv(data_dir + '/train.txt', header=None, index_col=None)
        self.test_data = pd.read_csv(data_dir + '/test.txt', header=None, index_col=None)
        self.item_list = pd.read_csv(data_dir + '/item_list.txt', header=0, index_col=None, sep=' ')
        
        if self.task == 'random-select':
            df = self.test_data
            self.train_data = df.sample(frac=0.7, random_state=42) # Sample the training set (70%)
            remaining = self.test_data.drop(self.train_data).reset_index(drop=True)
            self.valid_data = remaining.sample(frac=1/3, random_state=42)  # Sample the validation set (10% of total, which is 1/3 of remaining)
            self.test_data = remaining.drop(self.valid_data)
            
            # reset index for all splits
            self.train_data = self.train_data.reset_index(drop=True)
            self.valid_data = self.valid_data.reset_index(drop=True)
            self.test_data = self.test_data.reset_index(drop=True)
        

# Setting 1: Sequential Recommendation, Leave-one-out policy
class LeaveOneOut(data.Dataset):
    def __init__(self, data, item_list, data_name, k, user_token, shuffle, mode=None, max_len=20):
        super().__init__()
        if mode == 'train':  # for data augmentation
            self.data = data_construction(data, max_len)
        else:
            self.data = data
        self.max_len = max_len
        self.mode = mode
        self.item_list = item_list
        self.data_name = data_name
        self.k = k
        self.user_token = user_token
        self.shuffle = shuffle
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        sample = self.data.iloc[index, 0]  # format = string
        sample = sample.strip('\n').split(' ')  # split the list
        
        # 
        user = sample[0]
        items = sample[1:-1]
        interacted = " ".join(sample[1:-1])
        target = sample[-1]
            
        # truncation
        if len(items) > self.max_len:
            items = items[-self.max_len:]
            
        # shuffle
        if self.shuffle:
            random.shuffle(items)
        join_items = " ".join(items)
        
        prompt = utils.train_prompt(user, items, self.item_list, self.data_name, self.k, self.user_token)
        answer = utils.structure_response(target, self.item_list)  # the answer prompt to item brand or category
        return {'input': prompt, 'answer': answer, 'user': user, 'items': join_items, 'target': target, 'item_num':len(items), 'interacted': interacted}

    
def data_construction(data, max_len, min_len=3, augmentation=True, max_augment_num=10):  # data = pd.Dataframe()
    output_data = []
    for index, _ in data.iterrows():
        row = data.iloc[index, 0]
        sample = row.strip('\n').split(' ')  # user + items
        user = sample[0]
        items = sample[1:-1]  # leave-one-out for validation
        num = len(items)  # min = 3, max = 20

        # augmentation
        if augmentation:
            if num > max_len:
                augment_num = 0
                for i in range(num-max_len):
                    if i == 0:
                        temp_items = items[-max_len:]
                    else:
                        temp_items = items[-(max_len+i):-i]
                    current_sample = " ".join([user] + temp_items)
                    output_data.append(current_sample)
                    augment_num += 1
                    if augment_num >= max_augment_num:
                        break
            else:
                augment_num = 0
                for i in range(num, min_len-1, -1):
                    temp_items = items[-i:]
                    current_sample = " ".join([user] + temp_items)
                    output_data.append(current_sample)
                    augment_num += 1
                    if augment_num >= max_augment_num:
                        break       
                    
        else:
            current_sample = " ".join(sample[:-1])
            output_data.append(current_sample)
            
    output_data = pd.DataFrame(output_data)
    return output_data


# Setting 2: Collaborative Filtering, randomly select 7:1:2 interactions of each user for training:validation:testing
class RandomSelect(data.Dataset):
    def __init__(self, data, item_list, data_name, k, user_token, max_len=20, ratio=0.8):
        super().__init__()
        self.data = data
        self.max_len = max_len
        self.item_list = item_list
        self.data_name = data_name
        self.k = k
        self.user_token = user_token
        self.ratio = ratio
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        sample = self.data.iloc[index, 0]  # format = string
        sample = sample.strip('\n').split(' ')  # split the list
        
        user = sample[0]
        items = sample[1:]
        interacted = " ".join(items)
        if len(items) > self.max_len:  # truncation
            items = items[-self.max_len:]
        split = int(self.ratio * len(items))
        
        # randomly select
        random.shuffle(items)
        input_list = " ".join(items[:split])
        label_list = " ".join(items[split:])
        target = items[-1]
        
        prompt = utils.train_prompt(user, items[:split], self.item_list, self.data_name, self.k, self.user_token)
        answer = utils.structure_response(target, self.item_list)  # answer based on item brand or category, -1 is to select the last item
        # answer = utils.structure_response_cf(items[split:], self.item_list)  # answer based on item brand or category
        return {'input': prompt, 'answer': answer, 'user': user, 'items': input_list, 'target': target, 'item_num':len(items), 'interacted': interacted, 'label_list': label_list}
