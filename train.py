# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import argparse
from glob import glob
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F
import paddle.nn as nn

from paddlenlp.transformers import AutoModel, AutoTokenizer
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.utils.log import logger

from data import create_dataloader, read_text_pair, convert_example, ClassQADataset
from predict import predict, eval_write, do_predict, eval_test
from modelutils import (
    set_seed,
    label_smooth_loss,
    nnlloss_on_logits,
    evaluate,
    get_model,
    calculate_params,
    freeze,
    unfreeze,
)
from model import FGM
from scipy.special import softmax
from scipy import stats
from pytorch_attension_model import (
    read_examples,
    test_input,
    prob_postprocess,
    postprocess,
)


args = None
output_file_name = None
# yapf: disable

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_set", type=str, required=True, help="The full path of train_set_file")
    parser.add_argument("--dev_set", type=str, required=True, help="The full path of dev_set_file")
    parser.add_argument("--save_dir", default='./checkpoint', type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--plm_name", default='ernie-3.0-xbase-zh', type=str, help="pretrained transformer name")
    parser.add_argument("--model_name", default='QuestionMatching', type=str,
                        help="available [QMAttensionMultiLayer, QMAttensionMultiLayer1, QuestionMatching, QMAttensionMultiLayerCnn]")
    parser.add_argument("--test_file_path", default='test.tsv', type=str, help="")

    parser.add_argument("--max_seq_length", default=256, type=int, help="The maximum total input sequence length after tokenization. "
        "Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--max_steps', default=-1, type=int, help="If > 0, set total number of training steps to perform.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--epochs", default=3, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--plm_freeze", default=-1, type=int, help="layers to freeze from bottom")
    
    parser.add_argument("--eval_after_epochs", default=2, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--eval_step", default=100, type=int, help="Step interval for evaluation.")
    parser.add_argument('--save_step', default=10000, type=int, help="Step interval for saving checkpoint.")
    parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Linear warmup proption over the training process.")
    parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
    parser.add_argument('--resume', action='store_true', help="default False, or resume from model_best to continue train for ease. --init_from_ckpt will be checked first")
    parser.add_argument('--fp16', action='store_true', help="default False")
    parser.add_argument('--pad_to_max', action='store_true', help="default False")
    parser.add_argument('--smooth_label', action='store_true', help="default False")

    parser.add_argument("--seed", type=int, default=1000, help="Random seed for initialization.")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--rdrop_coef", default=0.0, type=float, help="The coefficient of"
        "KL-Divergence loss in R-Drop paper, for more detail please refer to https://arxiv.org/abs/2106.14448), if rdrop_coef > 0 then R-Drop works")
    parser.add_argument("--dropout_qm", default=0.1, type=float, help="query matching model dropout.")
    parser.add_argument("--num_dropout", default=5, type=int, help="Total number of dropout layers in AttensionModel")
    parser.add_argument("--purpose", default='', type=str, help="note for this run, dont' input space")
    parser.add_argument("--cv_fold", default=5, type=int, help="cross validation fold")
    parser.add_argument("--use_cls", default=0, type=int, help="0: not use; 1: cls, 2:mean, for QMAttensionMultiLayer1")
    parser.add_argument("--rnn", default='', type=str, help="none, lstm, gru")
    parser.add_argument("--att_rnn", default='', type=str, help="none, lstm, gru")
    parser.add_argument("--attension", default='additive', type=str, help="choise: additive, location, for QMAttensionMultiLayer1-2")
    parser.add_argument("--input_file", default='data/train/LCQMC/test', type=str, required=False, help="The full path of input file")
    parser.add_argument('--fp_white_list', nargs='+',
                        default=[], help="'elementwise_add', 'batch_norm', 'sync_batch_norm', 'softmax', 'gelu'")
    
    parser.add_argument("--loss_type", default=0, type=int, help="0: default cross_entropy, 1: use log_softmax + NLLLoss")
    # Cnn
    parser.add_argument("--out_channels", default=0, type=int,
                        help="0: use num_hidden_layers of plm, for Cnn")
    parser.add_argument("--dist_coef", default=0.0, type=float, help="The coefficient of")
    # QuestionMatchingDist
    
    
    # flow control 
    parser.add_argument('--eval_dm', action='store_true', help="default False, eval for dm competition")
    parser.add_argument('--eval_test', action='store_true', help="default False, eval test on lq and other dataset")
    
    parser.add_argument("--attack", default='', type=str, help="none, fgm")
    
    global args
    args = parser.parse_args()
    return args


def cosine_loss(y_pred, y_true):
    # y_pred is cosine. y_true is the 0-1 label.
    
    y_pred = y_pred * 20

    # 4. 取出负例-正例的差值
    y_pred = y_pred[:, None] - y_pred[None, :]  # 这里是算出所有位置 两两之间余弦的差值
    # 矩阵中的第i行j列  表示的是第i个余弦值-第j个余弦值
    y_true = y_true[:, None] < y_true[None, :]   # 取出负例-正例的差值
    y_true = y_true.astype('float32')
    y_pred = y_pred - (1 - y_true) * 1e12
    y_pred = paddle.reshape(y_pred, shape=[-1])
    y_pred = paddle.concat([paddle.to_tensor(np.array([0]).astype("float32")), y_pred], axis=0)
            
    return paddle.logsumexp(y_pred, axis=0)


def do_train(args):
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)

    train_ds = ClassQADataset(list(read_text_pair(data_path=args.train_set, is_test=False)), sort=False)

    dev_ds = load_dataset(read_text_pair,
                          data_path=args.dev_set,
                          is_test=False,
                          lazy=False)

    pretrained_model = AutoModel.from_pretrained(args.plm_name)
    tokenizer = AutoTokenizer.from_pretrained(args.plm_name)

    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         max_seq_length=args.max_seq_length,
                         pad_to_max=args.pad_to_max)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # text_pair_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # text_pair_segment
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]

    train_data_loader = create_dataloader(train_ds,
                                          mode='train',
                                          batch_size=args.train_batch_size,
                                          batchify_fn=batchify_fn,
                                          num_workers=0,
                                          trans_fn=trans_func)

    dev_data_loader = create_dataloader(dev_ds,
                                        mode='dev',
                                        batch_size=args.eval_batch_size,
                                        batchify_fn=batchify_fn,
                                        trans_fn=trans_func)

    # freeze(pretrained_model)
    model = get_model(pretrained_model, args=args)
    # paddle.summary(model, input_size=[(1, args.max_seq_length), (1, args.max_seq_length)], dtypes=['int64', 'int64'])
    
    fgm = FGM(model) if args.attack == 'fgm' else None
    
    for layer in model.sublayers():
        # print(type(layer).__name__, layer._full_name)
        if isinstance(layer, nn.TransformerEncoderLayer):
            
            if int(layer._full_name.split("_")[-1]) <= args.plm_freeze:
                freeze(layer)
                logger.info(f'freeze layer: {layer._full_name}')
    
    total_p, trainable_p, untrainble_p = calculate_params(model)
    logger.info(f"name={args.model_name}, total params={total_p}, trainable={trainable_p}, untrainable={untrainble_p}")

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)
        logger.info(f"load model from {args.init_from_ckpt}")
    elif args.resume:
        resume_file: str = os.path.join(args.save_dir, 'model_best', 'model_state.pdparams')
        state_dict = paddle.load(resume_file)
        model.set_dict(state_dict)
        del state_dict
        paddle.device.cuda.empty_cache()
        import gc
        gc.collect()
        logger.info(f"resume training model from {resume_file}")
    else:
        logger.info(f"not load model from init_from_ckpt or resume")

    model = paddle.DataParallel(model, find_unused_parameters=True)

    num_training_steps = len(train_data_loader) * args.epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)
    logger.info(f"total num_training_steps={num_training_steps}, epoch={args.epochs}")
    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    
    # clip = paddle.nn.ClipGradByValue(min=-5, max=5)
    clip = paddle.nn.ClipGradByGlobalNorm(10)
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        # grad_clip=clip,
        # multi_precision=args.fp16,
        apply_decay_param_fun=lambda x: x in decay_params)
    
    # optimizer = paddle.optimizer.Lamb(
    #     learning_rate=lr_scheduler, parameters=model.parameters(), lamb_weight_decay=args.weight_decay)
    
    if args.loss_type == 1:
        criterion = partial(nnlloss_on_logits, loss_func=nn.NLLLoss())
        logger.info("using NLLLoss")
    elif args.smooth_label:
        criterion = nn.CrossEntropyLoss(soft_label=True)
        criterion = partial(label_smooth_loss, loss_func=criterion)
        logger.info("using smooth_label")
    else:
        criterion = nn.CrossEntropyLoss(soft_label=False)
        logger.info("using cross entropy")

    binary_criterion = nn.BCELoss()
    metric = paddle.metric.Accuracy()

    global_step = 0
    best_accuracy = 0.0
    best_step = 0

    tic_train = time.time()
    
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

    if args.fp16:
        logger.info(f'using fp16 for training with while_list={args.fp_white_list}')

    def _forward(_model, _metric=None):
        h, dropout_hs, kl_loss =  _model(input_ids=input_ids, token_type_ids=token_type_ids, loss_fn=criterion, labels=labels)
        if _metric:
            correct = _metric.compute(h, labels)
            _metric.update(correct)
            acc = _metric.accumulate()
        loss = 0
        for i in range(len(dropout_hs)):
            loss += criterion(dropout_hs[i], labels)
        loss /= len(dropout_hs)
        # loss += kl_loss * args.rdrop_coef
        
        if kl_loss > 0:
            if args.rdrop_coef > 0:
                loss += kl_loss * args.rdrop_coef
            else:
                loss += kl_loss
        return loss 

    for epoch in range(1, args.epochs + 1):
        spearman = []
        for step, batch in enumerate(train_data_loader, start=1):
            if num_training_steps - global_step < num_training_steps % args.eval_step:
                # logger.info(f'skip step {global_step}')
                break
            
            input_ids, token_type_ids, labels = batch  # return numpy data
            # print('input shape: ', input_ids.shape)
            kl_loss=0
            cs_loss=0
            if args.model_name == 'QuestionMatching':
                if args.fp16:
                    with paddle.amp.auto_cast(custom_white_list=args.fp_white_list, level='O1'):
                        logits1, kl_loss = model(input_ids=input_ids, token_type_ids=token_type_ids)
                        correct = metric.compute(logits1, labels)
                else:
                    logits1, kl_loss = model(input_ids=input_ids, token_type_ids=token_type_ids)
                    correct = metric.compute(logits1, labels)
                metric.update(correct)
                acc = metric.accumulate()

                ce_loss = criterion(logits1, labels)
                if kl_loss > 0:
                    loss = ce_loss + kl_loss * args.rdrop_coef
                else:
                    loss = ce_loss
                
                scaled = scaler.scale(loss) # loss scale, multiply by the coefficient loss_scaling
                scaled.backward() # backward
                
            elif args.model_name == 'QuestionMatchingDist':
                if args.fp16:
                    with paddle.amp.auto_cast(custom_white_list=args.fp_white_list, level='O1'):
                        logits1, kl_loss, cosine_sim = model(input_ids=input_ids, token_type_ids=token_type_ids)
                        correct = metric.compute(logits1, labels)
                        spearman_corr = stats.spearmanr(paddle.squeeze(labels, 1).numpy(), cosine_sim.numpy()).correlation
                        spearman.append(spearman_corr)
                else:
                    logits1, kl_loss, cosine_sim = model(input_ids=input_ids, token_type_ids=token_type_ids)
                    correct = metric.compute(logits1, labels)
                    spearman_corr = stats.spearmanr(paddle.squeeze(labels, 1).numpy(), cosine_sim.numpy()).correlation
                    spearman.append(spearman_corr)
                metric.update(correct)
                acc = metric.accumulate()

                ce_loss = criterion(logits1, labels)
                if kl_loss > 0:
                    loss = ce_loss + kl_loss * args.rdrop_coef
                else:
                    loss = ce_loss
                
                if args.dist_coef > 0:
                    # loss += args.dist_coef * binary_criterion(cosine_sim, paddle.squeeze(labels, 1).astype("float32"))
                    cs_loss = args.dist_coef * cosine_loss(cosine_sim, paddle.squeeze(labels, 1).astype("float32"))
                    loss += cs_loss
                    
                scaled = scaler.scale(loss) # loss scale, multiply by the coefficient loss_scaling
                scaled.backward() # backward
            elif args.model_name in ['QMAttensionMultiLayer', 'QMAttensionMultiLayer1', 'QMAttensionMultiLayer2', 'QMAttensionMultiLayerCnn']:
                ce_loss = 0
                if args.fp16:
                    with paddle.amp.auto_cast(custom_white_list=args.fp_white_list, level='O1'):
                    # with paddle.amp.auto_cast(custom_white_list={'elementwise_add', "batch_norm", "sync_batch_norm", "softmax", "gelu"}, level='O1'):
                    # with paddle.amp.auto_cast(level='O1'):
                        h, dropout_hs, kl_loss =  model(input_ids=input_ids, token_type_ids=token_type_ids, loss_fn=criterion, labels=labels)
                        correct = metric.compute(h, labels)
                        metric.update(correct)
                        acc = metric.accumulate()
                        loss = 0
                        for i in range(len(dropout_hs)):
                            loss += criterion(dropout_hs[i], labels)
                        loss /= len(dropout_hs)
                        # loss += kl_loss * args.rdrop_coef
                        
                        if kl_loss > 0:
                            if args.rdrop_coef > 0:
                                loss += kl_loss * args.rdrop_coef
                            else:
                                loss += kl_loss
                        
                        scaled = scaler.scale(loss) # loss scale, multiply by the coefficient loss_scaling
                        scaled.backward() # backward
                        
                        if args.attack == 'fgm':
                            fgm.attack()
                            loss = _forward(model, _metric=None)
                            scaled = scaler.scale(loss) # loss scale, multiply by the coefficient loss_scaling
                            scaled.backward() # backward
                            fgm.restore()
                else:
                    h, dropout_hs, kl_loss  =  model(input_ids=input_ids, token_type_ids=token_type_ids, loss_fn=criterion, labels=labels)
                    correct = metric.compute(h, labels)
                    metric.update(correct)
                    acc = metric.accumulate()
                    loss = 0
                    for i in range(len(dropout_hs)):
                        loss += criterion(dropout_hs[i], labels)
                    loss /= len(dropout_hs)
                    # loss += kl_loss * args.rdrop_coef
                    if kl_loss > 0:
                        if args.rdrop_coef > 0:
                            loss += kl_loss * args.rdrop_coef
                        else:
                            loss += kl_loss
                
            
            global_step += 1
            if global_step % 50 == 0 and rank == 0:
                if spearman:
                    logger.info(
                    "global step %d, epoch: %d, batch: %d, loss: %.4f, ce_loss: %.4f, kl_loss: %.4f, cs_loss: %.4f, accu: %.4f, ave-spear: %.2f,  speed: %.2f step/s"
                    %
                    (global_step, epoch, step, loss, ce_loss, kl_loss, cs_loss, acc, sum(spearman)/len(spearman), 50/(time.time() - tic_train)))
                else:
                    logger.info(
                    "global step %d, epoch: %d, batch: %d, loss: %.4f, ce_loss: %.4f., kl_loss: %.4f, accu: %.4f, speed: %.2f step/s"
                    %
                    (global_step, epoch, step, loss, ce_loss, kl_loss, acc, 50 /
                     (time.time() - tic_train)))
                tic_train = time.time()
            if args.fp16:
                # scaled = scaler.scale(loss) # loss scale, multiply by the coefficient loss_scaling
                # scaled.backward() # backward
                
                scaler.step(optimizer)  # Update parameters (divide the parameter gradient by the coefficient loss_scaling and then update the parameters)
                scaler.update() # Based on dynamic loss_scaling policy update loss_scaling coefficient
                optimizer.clear_grad()
            else:
                # loss.backward()
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.clear_grad()

            if global_step % args.eval_step == 0 and rank == 0 and epoch > args.eval_after_epochs:
                if args.fp16:
                    with paddle.amp.auto_cast(custom_white_list=args.fp_white_list, level='O1'):
                        accuracy = evaluate(model, criterion, metric, dev_data_loader)
                else:
                    accuracy = evaluate(model, criterion, metric, dev_data_loader)
                    
                logger.info(
                    f"before current eval, best_accuracy={best_accuracy}, best_step={best_step}")
                if accuracy > best_accuracy:
                    save_dir = os.path.join(args.save_dir, "model_%s" % "best")
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_param_path = os.path.join(save_dir, 'model_state.pdparams')
                    paddle.save(model.state_dict(), save_param_path)
                    tokenizer.save_pretrained(save_dir)
                    best_accuracy = accuracy
                    best_step = global_step
                    
                elif global_step % args.save_step == 0:
                    save_dir = os.path.join(args.save_dir,
                                            "model_%d" % global_step)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_param_path = os.path.join(save_dir, 'model_state.pdparams')
                    paddle.save(model.state_dict(), save_param_path)
                    tokenizer.save_pretrained(save_dir)

            if global_step == args.max_steps:
                return
    
        if args.eval_test:
            eval_test(args=args, model=model, tokenizer=tokenizer)
    
    resume_file = os.path.join(args.save_dir, 'model_best', 'model_state.pdparams')
    logger.info(f"load best training model from {resume_file}")
    state_dict = paddle.load(resume_file)
    model.set_dict(state_dict)
    
    # do prediction
    if args.eval_dm:
        # load best model for eval
        logger.info(f"writing prediction to {output_file_name}")
        do_predict(model, tokenizer, args.test_file_path, output_file_name,
                   max_seq_length=args.max_seq_length, batch_size=args.eval_batch_size, pad_to_max=args.pad_to_max)
        
    return model, tokenizer, 

def do_cv(fold=5, args=None):
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)

    all_data = list(read_text_pair(data_path=args.train_set, is_test=False)) + \
        list(read_text_pair(data_path=args.train_set, is_test=False))
    
    x_indices = list(range(len(all_data)))
    lables = [item['label'] for item in all_data]
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=fold)
    y_probs_list = []
    for _fold, (train_index, test_index) in enumerate(skf.split(x_indices, lables)):
        train_data = [all_data[i] for i in train_index]
        dev_data = [all_data[i] for i in test_index]

        train_ds = ClassQADataset(train_data)
        dev_ds = ClassQADataset(dev_data, is_train=False) # not shuffle ab


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

        model = get_model(pretrained_model, args=args)

        if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
            state_dict = paddle.load(args.init_from_ckpt)
            model.set_dict(state_dict)
            logger.info(f"load model from {args.init_from_ckpt}")
        elif args.resume:
            resume_file = os.path.join(
                args.save_dir, f'model_best_cv{_fold}', 'model_state.pdparams')
            if os.path.exists(resume_file):
                state_dict = paddle.load(resume_file)
                model.set_dict(state_dict)
                logger.info(f"resume training model from {resume_file}")
            else:
                logger.info(f"training model file {resume_file} not exist, cannot resume")
        else:
            logger.info(f"not load model from init_from_ckpt or resume")

        model = paddle.DataParallel(model)

        num_training_steps = len(train_data_loader) * args.epochs

        lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                            args.warmup_proportion)
        logger.info(
            f"total num_training_steps={num_training_steps}, epoch={args.epochs}")
        # Generate parameter names needed to perform weight decay.
        # All bias and LayerNorm parameters are excluded.
        decay_params = [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in ["bias", "norm"])
        ]

        # clip = paddle.nn.ClipGradByValue(min=-5, max=5)
        clip = paddle.nn.ClipGradByGlobalNorm(5)
        optimizer = paddle.optimizer.AdamW(
            learning_rate=lr_scheduler,
            parameters=model.parameters(),
            weight_decay=args.weight_decay,
            grad_clip=clip,
            multi_precision=False,
            apply_decay_param_fun=lambda x: x in decay_params)

        # optimizer = paddle.optimizer.Lamb(
        #     learning_rate=lr_scheduler, parameters=model.parameters(), lamb_weight_decay=args.weight_decay)

        if args.smooth_label:
            criterion = paddle.nn.loss.CrossEntropyLoss(soft_label=True)
            criterion = partial(label_smooth_loss, loss_func=criterion)
        else:
            criterion = paddle.nn.loss.CrossEntropyLoss(soft_label=False)

        metric = paddle.metric.Accuracy()

        global_step = 0
        best_accuracy = 0.0
        best_step = 0

        tic_train = time.time()

        scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

        if args.fp16:
            logger.info(f'using fp16 for training')

        save_dir_best_model = f"model_best_cv{_fold}"
        for epoch in range(1, args.epochs + 1):
            for step, batch in enumerate(train_data_loader, start=1):
                input_ids, token_type_ids, labels = batch

                if args.model_name == 'QuestionMatching':
                    if args.fp16:
                        with paddle.amp.auto_cast(custom_white_list={'elementwise_add'}, level='O1'):
                            logits1, kl_loss = model(
                                input_ids=input_ids, token_type_ids=token_type_ids)
                            correct = metric.compute(logits1, labels)
                    else:
                        logits1, kl_loss = model(
                            input_ids=input_ids, token_type_ids=token_type_ids)
                        correct = metric.compute(logits1, labels)
                    metric.update(correct)
                    acc = metric.accumulate()

                    ce_loss = criterion(logits1, labels)
                    if kl_loss > 0:
                        loss = ce_loss + kl_loss * args.rdrop_coef
                    else:
                        loss = ce_loss
                elif args.model_name == 'QMAttensionMultiLayer' or args.model_name == 'QMAttensionMultiLayer1':
                    ce_loss = 0
                    kl_loss = 0
                    if args.fp16:
                        with paddle.amp.auto_cast(custom_white_list={'elementwise_add'}, level='O1'):
                            h, dropout_hs, kl_loss = model(
                                input_ids=input_ids, token_type_ids=token_type_ids)
                            correct = metric.compute(h, labels)
                            metric.update(correct)
                            acc = metric.accumulate()
                            loss = 0
                            for i in range(len(dropout_hs)):
                                loss += criterion(dropout_hs[i], labels)
                            loss /= len(dropout_hs)
                            if args.rdrop_coef > 0:
                                loss = loss + kl_loss * args.rdrop_coef
                    else:
                        h, dropout_hs, kl_loss = model(
                            input_ids=input_ids, token_type_ids=token_type_ids)
                        correct = metric.compute(h, labels)
                        metric.update(correct)
                        acc = metric.accumulate()
                        loss = 0
                        for i in range(len(dropout_hs)):
                            loss += criterion(dropout_hs[i], labels)
                        loss /= len(dropout_hs)
                        if args.rdrop_coef > 0:
                            loss = loss + kl_loss * args.rdrop_coef

                global_step += 1
                if global_step % 50 == 0 and rank == 0:
                    logger.info(
                        "global step %d, epoch: %d, batch: %d, loss: %.4f, ce_loss: %.4f., kl_loss: %.4f, accu: %.4f, speed: %.2f step/s"
                        %
                        (global_step, epoch, step, loss, ce_loss, kl_loss, acc, 50 /
                        (time.time() - tic_train)))
                    tic_train = time.time()
                if args.fp16:
                    # loss scale, multiply by the coefficient loss_scaling
                    scaled = scaler.scale(loss)
                    scaled.backward()  # backward
                    # Update parameters (divide the parameter gradient by the coefficient loss_scaling and then update the parameters)
                    scaler.step(optimizer)
                    scaler.update()  # Based on dynamic loss_scaling policy update loss_scaling coefficient
                    optimizer.clear_grad()
                else:
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.clear_grad()

                if global_step % args.eval_step == 0 and rank == 0:
                    # paddle.device.cuda.empty_cache()
                    if args.fp16:
                        with paddle.amp.auto_cast(custom_white_list={'elementwise_add'}, level='O1'):
                            accuracy = evaluate(
                                model, criterion, metric, dev_data_loader)
                    else:
                        accuracy = evaluate(
                            model, criterion, metric, dev_data_loader)

                    logger.info(
                        f"before current eval, best_accuracy={best_accuracy}, best_step={best_step}")
                    if accuracy > best_accuracy:
                        save_dir = os.path.join(args.save_dir, save_dir_best_model)
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        save_param_path = os.path.join(save_dir,
                                                    'model_state.pdparams')
                        paddle.save(model.state_dict(), save_param_path)
                        tokenizer.save_pretrained(save_dir)
                        best_accuracy = accuracy
                        best_step = global_step
                    elif global_step % args.save_step == 0:
                        save_dir = os.path.join(args.save_dir, save_dir_best_model, 
                                                "model_%d" % global_step)
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        save_param_path = os.path.join(save_dir, save_dir_best_model,
                                                    'model_state.pdparams')
                        paddle.save(model.state_dict(), save_param_path)
                        tokenizer.save_pretrained(save_dir)

                if global_step == args.max_steps:
                    return
        # do prediction
        if True:
            # load best model for eval
            current_cv_output = f"cv{_fold}_" + output_file_name
            logger.info(f"writing prediction to {current_cv_output}")
            resume_file = os.path.join(args.save_dir, save_dir_best_model, 'model_state.pdparams')
            state_dict = paddle.load(resume_file)
            model.set_dict(state_dict)
            logger.info(f"load best training model from {resume_file}")

            y_probs = do_predict(model, tokenizer, args.test_file_path, current_cv_output,
                    max_seq_length=args.max_seq_length, batch_size=args.eval_batch_size)
            y_probs_list.append(y_probs) # 已经做过softmax
        
        del model
        del pretrained_model
        paddle.device.cuda.empty_cache()
    
    logger.info(f"writing prediction to results/{output_file_name}")
    # y_probs_list = [softmax(y_logit, axis=1) for y_logit in y_probs_list]
    
    y_probs = np.mean(y_probs_list, axis=0)
    y_preds = np.argmax(y_probs, axis=1)

    with open(os.path.join("results", output_file_name), 'w', encoding="utf-8") as f:
        for y_pred in y_preds:
            f.write(str(y_pred) + "\n")

    test_examples = read_examples(test_input, split='test')

    # 后处理
    y_probs = softmax(y_probs, axis=1)
    oof_test = prob_postprocess(y_probs)
    logger.info(f'oof_test shape {oof_test.shape}')
    y_preds_post = np.argmax(oof_test, axis=1)

    post = postprocess(test_examples, y_preds_post)

    with open(os.path.join("results", "post_" + output_file_name), 'w', encoding="utf-8") as f:
        for y_pred in post:
            f.write(str(y_pred) + "\n")

    logger.info("finished evaluation")
        

if __name__ == "__main__":
    args = parse_args()
    train_data = args.train_set.split("/")[1]

    args.save_dir = f"{args.plm_name}_{args.model_name}_cls{args.use_cls}_att-{args.attension}_{train_data}_lr{args.learning_rate}_dc{args.weight_decay}_wup{args.warmup_proportion}_fp16{args.fp16}_dp{args.dropout_qm}_num_dp{args.num_dropout}_cv{args.cv_fold}_rdrop{args.rdrop_coef}_smooth{args.smooth_label}_freeze{args.plm_freeze}_pad_to_max{args.pad_to_max}{args.purpose}"

    output_file_name = args.save_dir + ".csv"
    logger.info(output_file_name)
    if args.cv_fold <= 1:
        model, tokenizer = do_train(args=args)
        if args.eval_test:
            from predict import eval_test
            eval_test(args=args, model=model, tokenizer=tokenizer)
    else:
        do_cv(fold=args.cv_fold, args=args)
