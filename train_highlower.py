from functools import partial
import argparse
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F

from paddlenlp.transformers import AutoModel, AutoTokenizer
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.utils.log import logger

from data import create_dataloader, read_text_pair, convert_example, ClassQADataset, ClassQADataset1
from model import QuestionMatching, QMAttensionMultiLayer

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--train_set", type=str, required=True,
                    help="The full path of train_set_file")
parser.add_argument("--dev_set", type=str, required=True,
                    help="The full path of dev_set_file")
parser.add_argument("--save_dir", default='./checkpoint', type=str,
                    help="The output directory where the model checkpoints will be written.")
parser.add_argument("--plm_name", default='ernie-3.0-xbase-zh',
                    type=str, help="pretrained transformer name")
parser.add_argument("--model_name", default='QuestionMatching', type=str,
                    help="available [QMAttensionMultiLayer, QuestionMatching]")

parser.add_argument("--max_seq_length", default=256, type=int, help="The maximum total input sequence length after tokenization. "
                    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument('--max_steps', default=-1, type=int,
                    help="If > 0, set total number of training steps to perform.")
parser.add_argument("--train_batch_size", default=32, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--eval_batch_size", default=128, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=5e-5, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight decay if we apply some.")
parser.add_argument("--epochs", default=3, type=int,
                    help="Total number of training epochs to perform.")
parser.add_argument("--eval_step", default=100, type=int,
                    help="Step interval for evaluation.")
parser.add_argument('--save_step', default=10000, type=int,
                    help="Step interval for saving checkpoint.")
parser.add_argument("--warmup_proportion", default=0.0, type=float,
                    help="Linear warmup proption over the training process.")
parser.add_argument("--init_from_ckpt", type=str, default=None,
                    help="The path of checkpoint to be loaded.")
parser.add_argument('--resume', action='store_true',
                    help="default False, or resume from model_best to continue train for ease. --init_from_ckpt will be checked first")
parser.add_argument('--fp16', action='store_true', help="default False")

parser.add_argument("--seed", type=int, default=1000,
                    help="Random seed for initialization.")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu",
                    help="Select which device to train model, defaults to gpu.")
parser.add_argument("--rdrop_coef", default=0.0, type=float, help="The coefficient of"
                    "KL-Divergence loss in R-Drop paper, for more detail please refer to https://arxiv.org/abs/2106.14448), if rdrop_coef > 0 then R-Drop works")
parser.add_argument("--dropout_qm", default=0.1, type=float,
                    help="query matching model dropout.")
parser.add_argument("--num_dropout", default=5, type=int,
                    help="Total number of dropout layers in AttensionModel")


def do_train(args):
    device = paddle.set_device(args.device)

    train_ds = ClassQADataset(list(read_text_pair(data_path=args.train_set, is_test=False)))

    dev_ds = load_dataset(read_text_pair,
                          data_path=args.dev_set,
                          is_test=False,
                          lazy=False)

    pretrained_model = AutoModel.from_pretrained(args.plm_name)
    tokenizer = AutoTokenizer.from_pretrained(args.plm_name)
    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_length)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # text_pair_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # text_pair_segment
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]

    train_data_loader = create_dataloader(train_ds,
                                          mode='train',
                                          batch_size=args.train_batch_size,
                                          batchify_fn=batchify_fn,
                                          trans_fn=trans_func)

    dev_data_loader = create_dataloader(dev_ds,
                                        mode='dev',
                                        batch_size=args.eval_batch_size,
                                        batchify_fn=batchify_fn,
                                        trans_fn=trans_func)
    
    if args.model_name == 'QMAttensionMultiLayer':
        model = QMAttensionMultiLayer(
            pretrained_model, rdrop_coef=args.rdrop_coef, dropout=args.dropout_qm, num_dropout=args.num_dropout)
    elif args.model_name == 'QuestionMatching':
        model = QuestionMatching(pretrained_model, rdrop_coef=args.rdrop_coef, dropout=args.dropout_qm)
    
    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
        logger.info(f"load model from {args.init_from_ckpt}")
    elif args.resume:
        resume_file = os.path.join(args.save_dir, 'model_best', 'model_state.pdparams')
        state_dict = paddle.load(resume_file)
        model.set_dict(state_dict)
        logger.info(f"resume training model from {resume_file}")
    else:
        logger.info(f"not load model from init_from_ckpt or resume")
    
    criterion = paddle.nn.loss.CrossEntropyLoss()
    # metric = paddle.metric.Accuracy()
    
    def loss_fn(input, hs_list, labels):
        print(input, hs_list, labels)
        logits = input
        return criterion(logits, labels)
    
    def accuracy_fn(input, lables):
        logits, _ = input
        return metric(logits, lables)
    
    class MyAccuracy(paddle.metric.Accuracy):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
        
        def compute(self, pred, label, *args):
            pred, _ = pred
            super().compute(pred, label, *args)
    
    metric = MyAccuracy()

    num_training_steps = len(train_data_loader) * args.epochs
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                            args.warmup_proportion)
    # logger.info(f"total num_training_steps={num_training_steps}, epoch={args.epochs}")
    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    
    clip = paddle.nn.ClipGradByValue(min=-5, max=5)
    # clip = paddle.nn.ClipGradByGlobalNorm(5)
    
    model = paddle.Model(model)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        grad_clip=clip,
        multi_precision=False,
        apply_decay_param_fun=lambda x: x in decay_params)
    
    model.prepare(optimizer, loss_fn, metric)

    print(args)
    if args.init_from_ckpt:
        model.load(args.init_from_ckpt)
        print("Loaded checkpoint from %s" % args.init_from_ckpt)

    benchmark_logger = paddle.callbacks.ProgBarLogger(log_freq=10,
                                                      verbose=3)

    model.fit(train_data=train_data_loader,
              eval_data=dev_data_loader,
              epochs=args.epochs,
              eval_freq=1,
              save_freq=1,
              save_dir=args.save_dir,
              callbacks=[benchmark_logger])


if __name__ == "__main__":
    args = parser.parse_args()
    do_train(args)
