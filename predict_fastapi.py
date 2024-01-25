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

from copy import deepcopy
from functools import partial
import argparse
import sys
import os
import random
import time
from fastapi import FastAPI
import uvicorn

import numpy as np
import pandas as pd
import paddle
import paddle.nn.functional as F
from paddlenlp.transformers import AutoModel, AutoTokenizer
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.metrics import AccuracyAndF1
from loguru import logger

from data import create_dataloader, read_text_pair, convert_example
from model import QuestionMatching, QMAttensionMultiLayer
from scipy.special import softmax
from pytorch_attension_model import read_examples, read_data, test_input, prob_postprocess, postprocess

from data import create_dataloader, read_text_pair, convert_example, ClassQADataset, create_eval_dataloader, create_test_data_loader
from modelutils import evaluate, predict, get_model


def parse_args_predict():
    # yapf: disable
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="The full path of input file")
    parser.add_argument("--result_file", type=str, required=True, help="The result file name")
    parser.add_argument("--init_from_ckpt", type=str, required=True,
                        help="The path to model parameters to be loaded.")
    parser.add_argument("--max_seq_length", default=256, type=int, help="The maximum total input sequence length after tokenization. "
        "Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--eval_batch_size", default=128,
                        type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
    parser.add_argument("--plm_name", default='ernie-3.0-xbase-zh', type=str, help="pretrained transformer name")
    parser.add_argument("--model_name", default='QuestionMatching', type=str, help="available [QMAttensionMultiLayer, QuestionMatching]")
    parser.add_argument("--dropout_qm", default=0.1, type=float, help="query matching model dropout.")
    parser.add_argument("--num_dropout", default=5, type=int, help="Total number of dropout layers in AttensionModel")
    parser.add_argument("--use_cls", default=0, type=int, help="0: not use; 1: cls, 2:mean, for QMAttensionMultiLayer1")
    parser.add_argument("--attension", default='additive', type=str, help="choise: additive, location, for QMAttensionMultiLayer1-2")
    parser.add_argument("--rdrop_coef", default=0.0, type=float, help="The coefficient of"
                        "KL-Divergence loss in R-Drop paper, for more detail please refer to https://arxiv.org/abs/2106.14448), if rdrop_coef > 0 then R-Drop works")
    # yapf: enable
    return parser.parse_args()


def do_predict(model, tokenizer, test_file_path, output_file_name, max_seq_length=64, batch_size=32, pad_to_max=False):
    data_loader = create_test_data_loader(test_file_path, tokenizer, max_seq_length=max_seq_length, batch_size=batch_size, pad_to_max=pad_to_max)
    return eval_write(model, data_loader, output_file_name)


def analyse(logits, labels, test_examples, prefix="analysis"):
    import pandas as pd
    labels = [x[0] for x in labels]
    y_probs = softmax(logits, axis=1)
    y_preds = np.argmax(y_probs, axis=1)
    probs1 = y_probs[:, 1]
    txt_a = [x.s1 for x in test_examples]
    txt_b = [x.s2 for x in test_examples]
    data = {'probs1': probs1, 'y_preds': y_preds, 'labels': labels, 'txta': txt_a, 'txtb': txt_b}
    print(probs1[:3], y_preds[:3], labels[:3], txt_a[:3])
    
    df: pd.DataFrame = pd.DataFrame(data)
    df.to_csv(f"analysis/{prefix}.tsv", index=False, header=True, sep='\t')
    # selecting rows based on condition
    rslt_df = df[df['y_preds'] != df['labels']]
    rslt_df.to_csv(f"analysis/{prefix}_error.tsv", index=False, header=True, sep='\t')
        

def eval_write(model, test_data_loader, output_file_name, output_dir="results"):
    y_probs = predict(model, test_data_loader)
    print('y_probs shape', y_probs.shape, y_probs[:5])
    y_preds = np.argmax(y_probs, axis=1)

    with open(os.path.join(output_dir, output_file_name), 'w', encoding="utf-8") as f:
        for y_pred in y_preds:
            f.write(str(y_pred) + "\n")
            
    test_examples = read_examples(test_input, split='test')
    
       # 后处理
    y_probs = softmax(y_probs, axis=1)
    
    oof_test = prob_postprocess(y_probs)
    print('oof_test shape', oof_test.shape)
    y_preds_post = np.argmax(oof_test, axis=1)
    with open(os.path.join(output_dir, "post_nopinyin_" + output_file_name), 'w', encoding="utf-8") as f:
        for y_pred in y_preds_post:
            f.write(str(y_pred) + "\n")
    

    post = postprocess(test_examples, y_preds_post)

    with open(os.path.join(output_dir, "post_" + output_file_name), 'w', encoding="utf-8") as f:
        for y_pred in post:
            f.write(str(y_pred) + "\n")
            
            
    post = postprocess(test_examples, y_preds)

    with open(os.path.join(output_dir, "post_noprob_adjust_" + output_file_name), 'w', encoding="utf-8") as f:
        for y_pred in post:
            f.write(str(y_pred) + "\n")

    logger.info("finished evaluation")
    
    return y_probs # origin softmax probs.


def eval_test(args=None, model=None, tokenizer=None):
    """_summary_: eval test of lq datasets (with label, not for competition)
    """
    if args is None:
        from train import parse_args
        args = parse_args()
    
    paddle.set_device(args.device)
    
    is_model_provided = model is not None
    
    if model is None:
        pretrained_model = AutoModel.from_pretrained(args.plm_name)
        tokenizer = AutoTokenizer.from_pretrained(args.plm_name)
        model = get_model(pretrained_model, args)
    data_loader = create_eval_dataloader(
        args.input_file, tokenizer, max_seq_length=args.max_seq_length, eval_batch_size=args.eval_batch_size, pad_to_max=args.pad_to_max)
    criterion = paddle.nn.loss.CrossEntropyLoss(soft_label=False)
    # metric = paddle.metric.Accuracy()
    metric = AccuracyAndF1()
    
    data_dict={}
    
    if is_model_provided:
        logger.info(f'using provided model')
    elif args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        keys = model.state_dict().keys()
        origin = [paddle.sum(v).numpy()[0] for v in model.state_dict().values()]
    
        state_dict = paddle.load(args.init_from_ckpt)
        loaded = [paddle.sum(v).numpy()[0] for v in state_dict.values()]
        print("keys:", len(keys), len(state_dict))
        
        model.set_dict(state_dict)
        
        set_loaded = [paddle.sum(v).numpy()[0] for v in model.state_dict().values()]
        data_dict['keys'] = keys
        data_dict['origin'] = origin
        data_dict['loaded'] = loaded
        data_dict['set_loaded'] = set_loaded
        # model.eval()
        # new_state_dict = model.state_dict()
        # print(type(state_dict), type(new_state_dict))
        # for k, v in state_dict.items():
        #     print(k, paddle.sum(v).numpy(), paddle.sum(new_state_dict[k]).numpy())
        logger.info(f"load model from {args.init_from_ckpt}")
    else:
        logger.info(f"did not find the parameters file")
        raise RuntimeError("cannot eval use default parameters")
        
    # evaluate(model, criterion, metric, data_loader)
    data_pred_dict = {}
    ret_dict = evaluate(model, criterion, metric, data_loader, return_dict=True)
    data_pred_dict['logit1'] = ret_dict['logits'][:, 0]
    data_pred_dict['label1'] = ret_dict['labels'].flatten()
    data_pred_dict['ids1'] = ret_dict['ids']
    
    data_dict['eval0'] = [paddle.sum(v).numpy()[0] for v in model.state_dict().values()]
    
    ret_dict = evaluate(model, criterion, metric, data_loader, return_dict=True)
    data_dict['eval1'] = [paddle.sum(v).numpy()[0] for v in model.state_dict().values()]
    
    data_pred_dict['logit2'] = ret_dict['logits'][:, 0]
    data_pred_dict['label2'] = ret_dict['labels'].flatten()
    data_pred_dict['ids2'] = ret_dict['ids']
    # ret_dict = evaluate(model, criterion, metric, data_loader, return_dict=True)
    # ret_dict = evaluate(model, criterion, metric, data_loader, return_dict=True)
    
    df: pd.DataFrame = pd.DataFrame(data_dict)
    df.to_csv(f"analysis/eval_params1.tsv", index=False, header=True, sep='\t')
    
    df: pd.DataFrame = pd.DataFrame(data_pred_dict)
    df.to_csv(f"analysis/eval_pred.tsv", index=False, header=True, sep='\t')
    
    # 后处理
    y_probs = ret_dict["logits"]
    labels = ret_dict["labels"]
    
    total_loss = ret_dict["loss"]
    y_probs = softmax(y_probs, axis=1)
    
    test_examples = read_data(args.input_file)
    logger.info(f"{y_probs.shape} VS {len(test_examples)}")
    
    analyse(y_probs, labels, test_examples)
    
    for i, p0 in enumerate([0.6]):
        y_probs1 = deepcopy(y_probs)
        
        # compute just pinyin
        if i == 0:
            y_preds_post = np.argmax(y_probs1, axis=1)
            # post = postprocess(test_examples, y_preds_post, debug=(i==0)).astype(np.float32)
            post = postprocess(test_examples, y_preds_post, debug=True).astype(np.float32)
            
            post_preds = np.array([[1., 0] if lb==0 else [0., 1] for lb in post])
            correct = metric.compute(paddle.to_tensor(post_preds), paddle.to_tensor(labels))
            metric.update(correct)
            accu = metric.accumulate()
            
            if isinstance(metric, paddle.metric.Accuracy):
                logger.info("dev_loss: {:.4}, accuracy: {:.4}".format(total_loss, accu))
            elif isinstance(metric, AccuracyAndF1):
                logger.info(f"only pinyin pp: dev_loss: {total_loss:.4f}, acc: {accu[0]:.4f}, precision: {accu[1]:.4f}, recall: {accu[2]:.4f}, f1: {accu[3]:.4f} acc and f1: {accu[4]:.4f}")
            metric.reset()
        
        # compute just prob adjust
        oof_test = prob_postprocess(y_probs1, p0=p0)
        metric.reset()
        correct = metric.compute(paddle.to_tensor(oof_test), paddle.to_tensor(labels))
        metric.update(correct)
        accu = metric.accumulate()
        analyse(oof_test, labels, test_examples, prefix=f"analysis_prob_adjust_p{p0}")
        
        
        if isinstance(metric, paddle.metric.Accuracy):
            logger.info("dev_loss: {:.5}, accuracy: {:.5}".format(total_loss, accu))
        elif isinstance(metric, AccuracyAndF1):
            logger.info(f"p0={p0}: dev_loss: {total_loss:.4f}, acc: {accu[0]:.4f}, precision: {accu[1]:.4f}, recall: {accu[2]:.4f}, f1: {accu[3]:.4f} acc and f1: {accu[4]:.4f}")
        metric.reset()
        
        
        # comput prob ajust and pinyin correction.
        y_preds_post = np.argmax(oof_test, axis=1)
        # post = postprocess(test_examples, y_preds_post, debug=(i==0)).astype(np.float32)
        post = postprocess(test_examples, y_preds_post).astype(np.float32)
        
        post_preds = np.array([[1., 0] if lb==0 else [0., 1] for lb in post])
        correct = metric.compute(paddle.to_tensor(post_preds), paddle.to_tensor(labels))
        metric.update(correct)
        accu = metric.accumulate()
        
        if isinstance(metric, paddle.metric.Accuracy):
            logger.info("dev_loss: {:.5}, accuracy: {:.5}".format(total_loss, accu))
        elif isinstance(metric, AccuracyAndF1):
            logger.info(f"prob adjust + pinyin: dev_loss: {total_loss:.4f}, acc: {accu[0]:.4f}, precision: {accu[1]:.4f}, recall: {accu[2]:.4f}, f1: {accu[3]:.4f} acc and f1: {accu[4]:.4f}")
        metric.reset()
        
        

def init_model(args=None, model=None, tokenizer=None):
    if args is None:
        from train import parse_args
        args = parse_args()

    paddle.set_device(args.device)

    is_model_provided = model is not None

    if model is None:
        pretrained_model = AutoModel.from_pretrained(args.plm_name)
        tokenizer = AutoTokenizer.from_pretrained(args.plm_name)
        model = get_model(pretrained_model, args)
    # criterion = paddle.nn.loss.CrossEntropyLoss(soft_label=False)
    # # metric = paddle.metric.Accuracy()
    # metric = AccuracyAndF1()

    data_dict = {}

    if is_model_provided:
        logger.info(f'using provided model')
    elif args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)

        # model.eval()
        # new_state_dict = model.state_dict()
        # print(type(state_dict), type(new_state_dict))
        # for k, v in state_dict.items():
        #     print(k, paddle.sum(v).numpy(), paddle.sum(new_state_dict[k]).numpy())
        logger.info(f"load model from {args.init_from_ckpt}")
        
    else:
        logger.info(f"did not find the parameters file")
        raise RuntimeError("cannot eval use default parameters")
    
    return model, args, tokenizer


from modelutils import calculate_params, freeze
model, args, tokenizer = init_model()
freeze(model)

app = FastAPI()


@app.get("/")
# @cache(expire=60)
async def classify(query, title, max_seq_length=64):
    encoded_inputs = tokenizer(text=[query],
                               text_pair=[title],
                               max_seq_len=max_seq_length, pad_to_max_seq_len=False)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]
    print(input_ids)
    
    model.eval()
    
    total_p, trainable_p, untrainble_p = calculate_params(model)
    logger.info(f"name={args.model_name}, total params={total_p}, trainable={trainable_p}, untrainable={untrainble_p}")
    outputs = model(input_ids=paddle.to_tensor(input_ids),
                        token_type_ids=paddle.to_tensor(token_type_ids),
                        do_evaluate=True)
    # logits = outputs[0]
    print(outputs)
    return "success"
    

if __name__ == '__main__':
    from pathlib import Path
    UVICORN_WORKERS = 1
    try:
        logger.info(
            """\n
 /**
 *                             _ooOoo_
 *                            o8888888o
 *                            88" . "88
 *                            (| -_- |)
 *                            O\  =  /O
 *                         ____/`---'\____
 *                       .'  \\|     |//  `.
 *                      /  \\|||  :  |||//  \
 *                     /  _||||| -:- |||||-  \
 *                     |   | \\\  -  /// |   |
 *                     | \_|  ''\---/''  |   |
 *                     \  .-\__  `-`  ___/-. /
 *                   ___`. .'  /--.--\  `. . __
 *                ."" '<  `.___\_<|>_/___.'  >'"".
 *               | | :  `- \`.;`\ _ /`;.`/ - ` : | |
 *               \  \ `-.   \_ __\ /__ _/   .-` /  /
 *          ======`-.____`-.___\_____/___.-`____.-'======
 *                             `=---='
 *          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 *                     佛祖保佑        永无BUG
*/
            """
        )
        logger.info(f"{UVICORN_WORKERS} workers will be spawn")
        uvicorn.run(app=f'{Path(__file__).stem}:app', host='0.0.0.0', port=8000,
                    workers=UVICORN_WORKERS)
    except Exception as e:
        logger.error(f'FastAPI start filed ❗❗❗: {e}')
