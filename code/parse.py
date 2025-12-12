import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go ContRec")
    
    # 
    parser.add_argument('--model', type=str, default='llama', help="the used language model")
    parser.add_argument('--test_only', default=False, action='store_true')
    parser.add_argument('--get_ckpt', default=False, action='store_true')
    parser.add_argument('--ckpt', type=str)
    
    # ----------------- encoding ----------------
    parser.add_argument('--k', type=int, default=2, help='K-way')
    parser.add_argument('--n', type=int, default=2, help='n layers')
    parser.add_argument('--user_token', default=False, action='store_true')
    parser.add_argument('--latent_dim', default=2048, type=int, help='The latent dimension of the embeddings that are sent into LLMs.')

    # ---------------- rec tokenizer -----------------------
    parser.add_argument('--n_token', type=int, default=256, help="the token number of each codebook")
    parser.add_argument('--n_book', type=int, default=3, help="the number of codebooks")
    parser.add_argument('--pretrain', action='store_true', default=False, help="if run vq: True or False (default).")
    parser.add_argument('--rec_tokenizer', type=str, default='continuous', help="[discrete, continuous]")
    parser.add_argument('--item_textual_information', action='store_true', default=False, help="if include textual information.")
    parser.add_argument('--continuous_tokenizer_model', type=str, default='sigma', help="[vallina, sigma]")     # if continuous
    parser.add_argument('--discrete_tokenizer_model', type=str, default='MQ', help="[VQ, RQ, MQ]")    # if discrete
    parser.add_argument('--content', type=str, default='collaborative', help="[collaborative, textual]")

    # ---------------- diffusion -----------------------
    parser.add_argument('--alpha', type=float, default=4, help='The weight of diffusion loss')
    parser.add_argument('--n_steps', type=int, default=1000, help='n_steps')
    parser.add_argument('--margin', type=float, default=2)
    parser.add_argument('--conditioning', type=str, default='mlp', help="[ave, mlp]")
    parser.add_argument('--beta_start', type=float, default=0.0001)
    parser.add_argument('--beta_end', type=float, default=0.02)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--timesteps', type=int, default=200, help='timesteps for diffusion')
    parser.add_argument('--hidden_factor', type=int, default=64, help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--beta_sche', default='exp', help='')
    parser.add_argument('--diffuser_type', type=str, default='mlp1', help='type of diffuser.')
    parser.add_argument('--w', type=float, default=2.0, help='dropout ')
    parser.add_argument('--seq_size', type=int, default=4, help='The conditioning length')
    
    # --------------- general --------------------
    parser.add_argument('--cuda', type=str, default='0', help="the used cuda")
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--shuffle', default=False, action='store_true')
    parser.add_argument('--eval', type=str, default='mse', help='[mse, inner, cos]')
    
    # learning
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--lr_scheduler', default='cosine', choices=['cosine'], type=str)
    parser.add_argument('--lr_decay_min_lr', default=1e-9, type=float)
    parser.add_argument('--lr_warmup_start_lr', default=1e-7, type=float)
    parser.add_argument('--auto_lr_find', default=False, action='store_true')
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--max_epochs', default=100, type=int)
    
    # llm
    parser.add_argument('--max_input_length', default=1024, type=int)
    parser.add_argument('--max_gen_length', default=64, type=int)
    
    # --------------- recommendation -----------------------------
    parser.add_argument('--dataset', default='beauty', type=str, help='[lastfm, ml1m, beauty, game]')    
    parser.add_argument('--rec_dim', default=512, type=int)  
    parser.add_argument('--top_k', default=10, type=int)
    parser.add_argument('--topks', nargs='?', default="[10, 20]", help="@k test list")
    parser.add_argument('--pi', default=0.05, type=float)
    parser.add_argument('--task', default='leave-one-out', choices=['leave-one-out', 'random-select'], type=str)  
    
    return parser.parse_args()