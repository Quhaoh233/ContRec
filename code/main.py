import sys
import lightning as L
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import parse
import torch
import utils
from model_interface import ModelInterface
from data_interface import DataInterface
from rec_tokenizer import vae
from rec_tokenizer import vq


def load_callbacks(args):
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='metric',
        mode='max',
        patience=10,
        min_delta=0.001
    ))

    callbacks.append(plc.ModelCheckpoint(
        monitor='metric',
        dirpath='../ckpt/' + args.dataset,
        filename='{epoch:02d}-{metric:.3f}',
        save_top_k=-1,
        mode='max',
        save_last=True,
        # train_time_interval=args.val_check_interval
        every_n_epochs=4
    ))

    if args.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(
            logging_interval='step'))
    return callbacks


def main(args):
    print(f"Welcome to ContRec! Now, we are working on Dataset: {args.dataset}.")
    L.seed_everything(args.seed)

    # # tokenizer pre-training
    # if args.rec_tokenizer == 'continuous':
    #     vae.learning(args)
    # elif args.rec_tokenizer == 'discrete':
    #     vq.learning(args)
    # else:
    #     NotImplementedError
    #     print('Please input the correct item tokenizer type: discrete or continuous.')

    # backbone finetuning
    callbacks = load_callbacks(args)
    model = ModelInterface(**vars(args))
    if args.get_ckpt:
        ckpt_path = '../ckpt/' + args.dataset + '/' + args.ckpt + '.ckpt'
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        
    data_module = DataInterface(args)
    trainer = pl.Trainer(devices=[int(args.cuda)], accelerator='cuda', max_epochs=args.max_epochs, logger=True, callbacks=callbacks, check_val_every_n_epoch=3)  # single device

    if args.test_only:
        pass
    else:
        trainer.fit(model=model, datamodule=data_module)  # train and valid
    trainer.test(model=model, datamodule=data_module)  # test


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    args = parse.parse_args()
    main(args)
