# base environments
import torch, sys, random
import pandas as pd
import numpy as np
import re
import copy
import pytorch_lightning as pl
# from fairscale.nn import checkpoint_wrapper, auto_wrap, wrap  # !!
from optims import LinearWarmupCosineLRScheduler
import torch.nn.functional as F
import warnings

# local environments
import diffusion.ddpm as ddpm
import diffusion.DreamRec as DreamRec
import utils
warnings.filterwarnings("ignore")

# llm environments
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
)


class ModelInterface(pl.LightningModule):
    def __init__(self, **kargs):
        super().__init__()
        self.save_hyperparameters()
        # the init will be impliment on cpu, which means self.device='cpu'. Then in training process, lightning will put the tensors and models into cuda automatically.
        self.gpu = 'cuda:' + self.hparams.cuda  # manually set cuda to avoid device bug
        self.load_llm()
        self.load_representation()
        self.load_diffusion()
        self.load_MLPEncoder()
        self.diffusion_token_ids, self.user_diffusion_token_ids = self.load_new_tokens()

        
    def forward(self, batch):
        # ----------------------- llm --------------------
        input_pairs = [[prompt, answer] for prompt, answer in zip(batch['input'], batch['answer'])]
        batch_size = len(input_pairs)
        input_encoding = self.tokenizer(input_pairs, return_tensors='pt', max_length=self.hparams.max_input_length, padding="max_length", truncation=True, return_token_type_ids=True)
        input_ids, attention_mask, token_type_ids = input_encoding.input_ids, input_encoding.attention_mask, input_encoding.token_type_ids
        input_embeds = self.model.get_input_embeddings()(input_ids.to(self.gpu))  # batch, max_length, dim
    
        # insert semantic and collaborative tokens into corresponding positions
        input_embeds = self.insert_diffusion_tokens(batch['user'], batch['items'], input_ids, input_embeds, batch_size)

        # mask the pad tokens
        target_ids = copy.deepcopy(input_ids)
        target_ids = target_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)
        target_ids = target_ids.masked_fill(token_type_ids == 0, -100)

        # device = input_embeds.device
        outputs = self.model(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask.to(self.gpu),
                    return_dict=True,
                    labels=target_ids.to(self.gpu),
                    use_cache=False,
                )
        lm_loss = outputs.loss
        
        
        # link the outputs to diffusion
        # ----------------------- diffusion -------------------------
        responses = [answer for answer in batch['answer']]
        responses_ids = self.tokenizer(responses, return_tensors='pt', max_length=self.hparams.max_gen_length, padding="max_length", truncation=True, return_token_type_ids=True).input_ids
        responses_embeds = self.model.get_input_embeddings()(responses_ids.to(self.gpu))  # [batch, gen_length, dim]
        
        targets = batch['target']  # the target here is an item randomly selected from the labels (20%)
        x = self.from_remap_ids_to_encoded_embeds(targets)  # [batch, 2, 2048] / [batch, 512]  # the target is changed! use contrastive learning objective instead!
        if len(x.shape) < 2:
            x = x.unsqueeze(0)

        # create conditioning
        condition = self.output_decoder(responses_embeds.float())
        current_batch = x.shape[0]
        
        # diffusion
        targets = [int(tgt) for tgt in targets]  # label_item_id, 4682
        x_start = self.diffusion_net.cacu_x(targets)
        h = self.diffusion_net.cacu_h(condition, self.hparams.seq_size, self.hparams.dropout)
        n = torch.randint(0, self.hparams.timesteps, (current_batch, ), device=self.gpu).long()
        difloss, predicted_x = self.ddpm.p_losses(self.diffusion_net, x_start, h, n, loss_type='l2')
            
        # disperse loss, a regularization term, no need to train on the negative samples!
        disploss = self.disp_loss(predicted_x)
        
        return lm_loss, difloss, disploss
    
    def configure_loss(self, out, labels=None):
        lm_loss, difloss, disploss = out
        return 0.1 * lm_loss + difloss + 0.5 * disploss
    
    def training_step(self, batch, batch_idx):
        torch.autograd.set_detect_anomaly(True)  # use to check the anomaly in gradient backward
        if self.scheduler:
            self.scheduler.step(self.trainer.global_step, self.current_epoch, self.trainer.max_steps)
        out = self(batch)
        loss = self.configure_loss(out)
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log('lr', self.scheduler.optimizer.param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log('global_step_num', self.trainer.global_step, on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        return loss

    # validation functions
    def generate(self, batch, temperature=0, do_sample=False, num_beams=1, min_gen_length=1, repetition_penalty=1.0, length_penalty=1.0, num_return_sequences=1):
        max_gen_length = self.hparams.max_gen_length
        input_pairs = [prompt for prompt, answer in zip(batch['input'], batch['answer'])]
        batch_size = len(input_pairs)
        input_encoding = self.tokenizer(input_pairs, return_tensors='pt', max_length=self.hparams.max_input_length, padding="max_length", truncation=True, return_token_type_ids=True)
        input_ids, attention_mask, token_type_ids = input_encoding.input_ids, input_encoding.attention_mask, input_encoding.token_type_ids
        input_embeds = self.model.get_input_embeddings()(input_ids.to(self.gpu))  # ids to embeds
        input_embeds = self.insert_diffusion_tokens_valid(batch['user'], batch['items'], input_ids, input_embeds, batch_size)  # insert semantic and collaborative tokens into corresponding positions
        
        outputs = self.model.generate(
            inputs_embeds=input_embeds.to(self.gpu),
            attention_mask=attention_mask.to(self.gpu),
            temperature=temperature,
            do_sample=do_sample,
            num_beams=num_beams,
            max_new_tokens=max_gen_length,
            min_new_tokens=min_gen_length,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
            output_scores=True,
            return_dict_in_generate=True,
            return_legacy_cache=True,
            )
        
        sequence_ids = outputs['sequences']
        outputs = self.tokenizer.batch_decode(sequence_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)  # list, reasoning texts
        
        # ddpm backward
        sequence_ids_list = {"input_ids": [seq.tolist() for seq in sequence_ids]}
        padded_sequence_ids = self.tokenizer.pad(sequence_ids_list, padding='max_length', max_length=max_gen_length, return_tensors="pt")['input_ids']
        gen_embeds = self.model.get_input_embeddings()(padded_sequence_ids.to(self.gpu))  # [batch, len, dim]
        condition = self.output_decoder(gen_embeds.float())
        
        with torch.no_grad():
            predictions, scores = self.diffusion_net.predict(condition, np.array([self.hparams.seq_size for _ in range(batch_size)]), self.ddpm)  # predictions.shape = [batch, embed_dim]

        return outputs, predictions, scores
    
    # -------------------- validation --------------------------
    def on_validation_epoch_start(self):
        self.val_content={
            "generated_text":[],
            "generated_representation":[],
            "label":[],
            "interacted_items":[],
            "interacted_bools":[],
            "scores":[]
        }
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        targets = batch['target']
        batch_size = len(targets)
        interacted_items = batch['interacted']
        texts, representations, scores = self.generate(batch)

        for b in range(batch_size):
            self.val_content["generated_text"].append(texts[b])
            self.val_content["generated_representation"].append(representations[b])
            self.val_content["label"].append(targets[b])
            self.val_content["scores"].append(scores[b])
            item_bool = torch.zeros([self.item_num])
            items = [int(item) for item in interacted_items[b].split(" ")[:-1]]
            item_bool[items] = 1
            self.val_content["interacted_bools"].append(item_bool)
            self.val_content["interacted_items"].append(interacted_items[b])
            
    def on_validation_epoch_end(self):
        # text score
        pattern = r'$(.*?)$'
        valid_texts = [re.findall(pattern, text)[0] for text in self.val_content["generated_text"]]
        categories = self.item_list['category'].to_list()
        categories = [str(cat) for cat in categories]
        text_score = torch.tensor(utils.calculate_text_score(valid_texts, categories, self.hparams.pi), device=self.gpu)
        valid_representations = torch.stack(self.val_content["generated_representation"], dim=0).squeeze()

        #
        if self.hparams.eval == "cos":
            interacted_items = self.val_content["interacted_items"]
            scores = utils.similarity_score(valid_representations, self.item_gnn_embeds, interacted_items)
            results = torch.argsort(scores, dim=1, descending=True)
        elif self.hparams.eval == "mse":
            item_bool_list = torch.stack(self.val_content["interacted_bools"], dim=0)
            scores = utils.MSE_distance(valid_representations, self.item_gnn_embeds) # the less the better
            scores[item_bool_list == 1] = 1000000  # remove the interacted items, [user_num, item_num]
            results = torch.argsort(scores, dim=1, descending=False)
        elif self.hparams.eval == 'inner':   
            val_scores = torch.stack(self.val_content["scores"], dim=0).squeeze()
            val_scores = val_scores * (1 + text_score)
            item_bool_list = torch.stack(self.val_content["interacted_bools"], dim=0)
            val_scores[item_bool_list == 1] = 0  # remove the interacted items, [user_num, item_num]
            results = torch.argsort(val_scores, dim=1, descending=True)
        else:
            print("Please input correct evaluation function: [cos, mse, inner]")
            NotImplementedError
        
        metric, batch = utils.get_metrics(self.val_content["label"], results, self.hparams.top_k)
        # save logs
        self.log('val_hr', metric[0]/batch, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_ndcg', metric[1]/batch, on_step=False, on_epoch=True, prog_bar=True)
        self.log('metric', metric[1]/batch, on_step=False, on_epoch=True, prog_bar=True)
    
    # -------------------- test ----------------------
    def on_test_epoch_start(self):
        self.test_content={
            "generated_text":[],
            "generated_representation":[],
            "label":[],
            "interacted_items":[],
            "interacted_bools":[],
            "scores":[]
        }
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        targets = batch['target']
        interacted_items = batch['interacted']
        batch_size = len(targets)
        texts, representations, scores = self.generate(batch)

        for b in range(batch_size):
            self.test_content["generated_text"].append(texts[b])
            self.test_content["generated_representation"].append(representations[b])
            self.test_content["label"].append(targets[b])
            self.test_content["scores"].append(scores[b])
            item_bool = torch.zeros([self.item_num])
            items = [int(item) for item in interacted_items[b].split(" ")[:-1]]
            item_bool[items] = 1
            self.test_content["interacted_bools"].append(item_bool)
            self.test_content["interacted_items"].append(interacted_items[b])
            
    def on_test_epoch_end(self):
        # text score, 0.1
        pattern = r'$(.*?)$'
        test_texts = [re.findall(pattern, text)[0] for text in self.test_content["generated_text"]]
        categories = self.item_list['category'].to_list()
        categories = [str(cat) for cat in categories]
        text_score = utils.calculate_text_score(test_texts, categories, self.hparams.pi)
        test_representations = torch.stack(self.test_content["generated_representation"], dim=0).squeeze()

        # 
        if self.hparams.eval == "cos":
            interacted_items = self.test_content["interacted_items"]
            scores = utils.similarity_score(test_representations, self.item_gnn_embeds, interacted_items)
            results = torch.argsort(scores, dim=1, descending=True)
        elif self.hparams.eval == "mse":
            item_bool_list = torch.stack(self.test_content["interacted_bools"], dim=0)
            scores = utils.MSE_distance(test_representations, self.item_gnn_embeds) # the less the better
            scores[item_bool_list == 1] = 1000000  # remove the interacted items, [user_num, item_num]
            results = torch.argsort(scores, dim=1, descending=False)
        elif self.hparams.eval == 'inner':   
            test_scores = torch.stack(self.val_content["scores"], dim=0).squeeze()
            test_scores = test_scores * (1 + text_score)
            item_bool_list = torch.stack(self.test_content["interacted_bools"], dim=0)
            test_scores[item_bool_list == 1] = 0  # remove the interacted items, [user_num, item_num]
            results = torch.argsort(test_scores, dim=1, descending=True)
        else:
            print("Please input correct evaluation function: [cos, mse, inner]")
            NotImplementedError

        metric, batch = utils.get_metrics(self.test_content["label"], results, self.hparams.top_k)
        # save logs
        self.log('test_hr', metric[0]/batch, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_ndcg', metric[1]/batch, on_step=False, on_epoch=True, prog_bar=True)
        self.log('metric', metric[1]/batch, on_step=False, on_epoch=True, prog_bar=True)
        
    # 
    def configure_optimizers(self):  # this function will run automatically
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        
        if self.hparams.conditioning == 'mlp':
            if self.hparams.user_token:
                optimizer = torch.optim.AdamW([
                    {'params': self.model.parameters(), 'lr': self.hparams.lr, 'weight_decay':weight_decay, 'betas':[0.9, 0.98]},
                    {'params': self.diffusion_net.parameters(), 'lr': self.hparams.lr, 'weight_decay':weight_decay, 'betas':[0.9, 0.98]},
                    {'params': self.item_encoder.parameters(), 'lr': self.hparams.lr, 'weight_decay':weight_decay},
                    {'params': self.user_encoder.parameters(), 'lr': self.hparams.lr, 'weight_decay':weight_decay},
                    {'params': self.output_decoder.parameters(), 'lr': self.hparams.lr, 'weight_decay':weight_decay},
                ])
            else:
                optimizer = torch.optim.AdamW([
                    {'params': self.model.parameters(), 'lr': self.hparams.lr, 'weight_decay':weight_decay, 'betas':[0.9, 0.98]},
                    {'params': self.diffusion_net.parameters(), 'lr': self.hparams.lr, 'weight_decay':weight_decay, 'betas':[0.9, 0.98]},
                    {'params': self.item_encoder.parameters(), 'lr': self.hparams.lr, 'weight_decay':weight_decay},
                    {'params': self.output_decoder.parameters(), 'lr': self.hparams.lr, 'weight_decay':weight_decay},
                ])
            
        else:
            if self.hparams.user_token:
                optimizer = torch.optim.AdamW([
                    {'params': self.model.parameters(), 'lr': self.hparams.lr, 'weight_decay':weight_decay, 'betas':[0.9, 0.98]},
                    {'params': self.diffusion_net.parameters(), 'lr': self.hparams.lr, 'weight_decay':weight_decay, 'betas':[0.9, 0.98]},
                    {'params': self.item_encoder.parameters(), 'lr': self.hparams.lr, 'weight_decay':weight_decay},
                    {'params': self.user_encoder.parameters(), 'lr': self.hparams.lr, 'weight_decay':weight_decay},
                ])
            else:
                optimizer = torch.optim.AdamW([
                    {'params': self.model.parameters(), 'lr': self.hparams.lr, 'weight_decay':weight_decay, 'betas':[0.9, 0.98]},
                    {'params': self.diffusion_net.parameters(), 'lr': self.hparams.lr, 'weight_decay':weight_decay, 'betas':[0.9, 0.98]},
                    {'params': self.item_encoder.parameters(), 'lr': self.hparams.lr, 'weight_decay':weight_decay},
                ])  
            
            
        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            max_step = self.trainer.max_steps
            warmup_steps = max_step // 20
            print(f'max_step: {max_step}')
            print(f'warmup_steps: {warmup_steps}')
            if self.hparams.lr_scheduler == 'cosine':
                self.scheduler = LinearWarmupCosineLRScheduler(optimizer,
                                                  max_step=max_step,
                                                  min_lr=self.hparams.lr_decay_min_lr,
                                                  init_lr=self.hparams.lr,
                                                  warmup_steps=warmup_steps,
                                                  warmup_start_lr=self.hparams.lr_warmup_start_lr)
            else:
                self.scheduler = None
                raise ValueError('Invalid lr_scheduler type!')
            return optimizer
    
    # ------------------------- functions ----------------------------
    def insert_diffusion_tokens(self, user, items, input_ids, input_embeds, batch_size):
        for b in range(batch_size):
            batch_input_ids = input_ids[b]
            batch_items = items[b].split(' ')
            batch_items = [int(item) for item in batch_items]
            batch_items_gnn_embeds = self.item_gnn_embeds[batch_items]
            batch_collaborative_embeds = self.item_encoder(batch_items_gnn_embeds.to(self.gpu))
            
            # replacing
            indice = torch.nonzero(batch_input_ids == self.diffusion_token_ids[0]).squeeze()  # find the first diffusion token
            if self.hparams.user_token:
                user_gnn_embed = self.user_gnn_embeds[int(user[b])].unsqueeze(0)  # [1, rec_dim]
                user_collaborative_embeds = self.user_encoder(user_gnn_embed.to(self.gpu))  # [k, 1, llm_dim]
                user_indice = torch.nonzero(batch_input_ids == self.user_diffusion_token_ids[0]).squeeze()
                
            for i in range(self.hparams.k):
                input_embeds[b, indice+i, :] = batch_collaborative_embeds[i].to(input_embeds.dtype)
                if self.hparams.user_token:
                    input_embeds[b, user_indice+i, :] = user_collaborative_embeds[i].to(input_embeds.dtype)
        return input_embeds

    def insert_diffusion_tokens_valid(self, user, items, input_ids, input_embeds, batch_size):
        for b in range(batch_size):
            batch_input_ids = input_ids[b]
            batch_items = items[b].split(' ')
            batch_items = [int(item) for item in batch_items]
            batch_items_gnn_embeds = self.item_gnn_embeds[batch_items]  # batch_items_gnn_embeds.shape = [item_num, rec_dim]
            batch_collaborative_embeds = self.item_encoder(batch_items_gnn_embeds.to(self.gpu))  # shape = [k, item_num, llm_dim]
            
            # replacing
            indice = torch.nonzero(batch_input_ids == self.diffusion_token_ids[0]).squeeze()  # find the first diffusion token
            if self.hparams.user_token:
                user_gnn_embed = self.user_gnn_embeds[int(user[b])].unsqueeze(0)  # [1, rec_dim]
                user_collaborative_embeds = self.user_encoder(user_gnn_embed.to(self.gpu))  # [k, 1, llm_dim]
                user_indice = torch.nonzero(batch_input_ids == self.user_diffusion_token_ids[0]).squeeze()
                
            for i in range(self.hparams.k):
                input_embeds[b, indice+i, :] = batch_collaborative_embeds[i].to(input_embeds.dtype)
                if self.hparams.user_token:
                    input_embeds[b, user_indice+i, :] = user_collaborative_embeds[i].to(input_embeds.dtype)

        return input_embeds

    def from_remap_ids_to_encoded_embeds(self, str_ids):
        remap_ids = [int(id) for id in str_ids]
        embeds = self.item_gnn_embeds[remap_ids].to(self.gpu)
        return embeds

    # Dispersive Loss implementation (InfoNCE-L2 variant), code source: https://github.com/raywang4/DispLoss
    def disp_loss(self, z):
        # z = z.reshape((z.shape[0],-1)) # flatten, [batch, dim]
        diff = torch.nn.functional.pdist(z).pow(2)/z.shape[1] # pairwise distance
        diff = torch.concat((diff, diff, torch.zeros(z.shape[0]).cuda()))  # match JAX implementation of full BxB matrix
        return - torch.log(torch.exp(-diff).mean()) # calculate loss
    
    # for conventional contrastive learning
    def negative_sampling(self, item_gnn_embeds, n=10):
        random_indices = random.sample(range(0, self.item_num), 10)
        neg_x = item_gnn_embeds[random_indices].to(self.gpu)
        return torch.mean(neg_x, dim=0).unsqueeze(0)
    
    # ------------------------ loading -------------------------
    def load_new_tokens(self):
        self.indicators = ['<diff>', '<\diff>']
        dif_tokens = []
        user_dif_tokens = []
        for k in range(self.hparams.k):
            dif_tokens.append(f'<collaborative_{str(k)}>')
            user_dif_tokens.append(f'<user_collaborative_{str(k)}>')

        add_tokens = self.indicators + dif_tokens + user_dif_tokens
        self.tokenizer.add_tokens(add_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        diffusion_token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in dif_tokens]
        user_diffusion_token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in user_dif_tokens]
        return diffusion_token_ids, user_diffusion_token_ids
        
    def load_diffusion(self):
        # get hyper params
        input_text = "Hello world."
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(input_ids.to(self.gpu))
            logits = outputs.logits
            self.logit_dim = logits.shape[2]  # logit dim
        input_embeds = self.model.get_input_embeddings()(input_ids.to(self.gpu))  # ids to embeds
        _, _, self.token_dim = input_embeds.shape  # token dimension

        # seq_size = conditioning length
        self.diffusion_net = DreamRec.Tenc(self.hparams.hidden_factor, self.item_gnn_embeds, self.hparams.seq_size, self.hparams.dropout, self.hparams.diffuser_type, self.gpu)
        self.ddpm = DreamRec.diffusion(self.hparams.timesteps, self.hparams.beta_start, self.hparams.beta_end, self.hparams.w, self.gpu, self.hparams.beta_sche)
    
    def load_MLPEncoder(self):
        self.item_encoder = utils.MLPEncoder(self.gnn_dim, self.token_dim, self.hparams.k).to(self.gpu)
        self.user_encoder = utils.MLPEncoder(self.gnn_dim, self.token_dim, self.hparams.k).to(self.gpu)
        if self.hparams.conditioning == 'mlp':
            self.output_decoder = utils.MLPDecoder(self.hparams.max_gen_length, self.hparams.seq_size, self.token_dim, self.gnn_dim).to(self.gpu)
    
    def load_representation(self):
        # load meta data
        data_dir = '../data/' + self.hparams.dataset
        self.item_list = pd.read_csv(data_dir+'/item_list.txt', header=0, index_col=None, sep=' ')
        
        # load the pre-trained GNN representations and textual representations here!
        self.gnn_embeds = torch.load('../src/lgn-'+ self.hparams.dataset +'-' + str(self.hparams.rec_dim) + '.pth.tar') 
        self.user_gnn_embeds = self.gnn_embeds['embedding_user.weight'].to(self.gpu)  # requires_grad = False
        self.item_gnn_embeds = self.gnn_embeds['embedding_item.weight'].to(self.gpu)  # requires_grad = False
        self.item_num, self.gnn_dim = self.item_gnn_embeds.shape
        self.user_num, _ = self.user_gnn_embeds.shape

        print('Representations are loaded. item_num, user_num, gnn_dim =', self.item_num, self.user_num, self.gnn_dim)
        
    def load_llm(self):
        # Llama Config
        model_name = 'Llama-3.2-1B-Instruct'  # model
        hf_token = "enter_your_HF_TOKEN" # hf_token
        model_source = 'meta-llama/'
        model_id = model_source + model_name
        torch_dtype = torch.float16
        attn_implementation = "eager"  # eager, FlashAttention, ...
        cache_dir='/home/haohao/.huggingface'  # base model save_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, cache_dir=cache_dir, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # parameter quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=hf_token,
            quantization_config=bnb_config,
            device_map=self.gpu,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation
        )

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
        )
        self.model = get_peft_model(model, peft_config)
        print('Loading LLAMA Done')