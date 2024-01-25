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
from pytorch_attension_model import (
    read_examples,
    read_data,
    test_input,
    prob_postprocess,
    postprocess,
)

from data import (
    create_dataloader,
    read_text_pair,
    convert_example,
    ClassQADataset,
    create_eval_dataloader,
    create_test_data_loader,
)
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


def do_predict(
    model,
    tokenizer,
    test_file_path,
    output_file_name,
    max_seq_length=64,
    batch_size=32,
    pad_to_max=False,
):
    data_loader = create_test_data_loader(
        test_file_path,
        tokenizer,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        pad_to_max=pad_to_max,
    )
    return eval_write(model, data_loader, output_file_name)


def analyse(logits, labels, test_examples, prefix="analysis"):
    import pandas as pd

    labels = [x[0] for x in labels]
    y_probs = softmax(logits, axis=1)
    y_preds = np.argmax(y_probs, axis=1)
    probs1 = y_probs[:, 1]
    txt_a = [x.s1 for x in test_examples]
    txt_b = [x.s2 for x in test_examples]
    data = {
        "probs1": probs1,
        "y_preds": y_preds,
        "labels": labels,
        "txta": txt_a,
        "txtb": txt_b,
    }
    print(probs1[:3], y_preds[:3], labels[:3], txt_a[:3])

    df: pd.DataFrame = pd.DataFrame(data)
    df.to_csv(f"analysis/{prefix}.tsv", index=False, header=True, sep="\t")
    # selecting rows based on condition
    rslt_df = df[df["y_preds"] != df["labels"]]
    rslt_df.to_csv(f"analysis/{prefix}_error.tsv", index=False, header=True, sep="\t")


def eval_write(model, test_data_loader, output_file_name, output_dir="results"):
    y_probs = predict(model, test_data_loader)
    print("y_probs shape", y_probs.shape, y_probs[:5])
    y_preds = np.argmax(y_probs, axis=1)

    with open(os.path.join(output_dir, output_file_name), "w", encoding="utf-8") as f:
        for y_pred in y_preds:
            f.write(str(y_pred) + "\n")

    test_examples = read_examples(test_input, split="test")

    # 后处理
    y_probs = softmax(y_probs, axis=1)

    oof_test = prob_postprocess(y_probs)
    print("oof_test shape", oof_test.shape)
    y_preds_post = np.argmax(oof_test, axis=1)
    with open(
        os.path.join(output_dir, "post_nopinyin_" + output_file_name),
        "w",
        encoding="utf-8",
    ) as f:
        for y_pred in y_preds_post:
            f.write(str(y_pred) + "\n")

    post = postprocess(test_examples, y_preds_post)

    with open(
        os.path.join(output_dir, "post_" + output_file_name), "w", encoding="utf-8"
    ) as f:
        for y_pred in post:
            f.write(str(y_pred) + "\n")

    post = postprocess(test_examples, y_preds)

    with open(
        os.path.join(output_dir, "post_noprob_adjust_" + output_file_name),
        "w",
        encoding="utf-8",
    ) as f:
        for y_pred in post:
            f.write(str(y_pred) + "\n")

    logger.info("finished evaluation")

    return y_probs  # origin softmax probs.


def eval_test(args=None, model=None, tokenizer=None, return_metric=False):
    """_summary_: eval test of lq datasets (with label, not for competition)"""
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
        args.input_file,
        tokenizer,
        max_seq_length=args.max_seq_length,
        eval_batch_size=args.eval_batch_size,
        pad_to_max=args.pad_to_max,
    )
    criterion = paddle.nn.loss.CrossEntropyLoss(soft_label=False)
    # metric = paddle.metric.Accuracy()
    metric = AccuracyAndF1()

    if is_model_provided:
        logger.info(f"using provided model")
    elif args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)

        logger.info(f"load model from {args.init_from_ckpt}")
    else:
        logger.info(f"did not find the parameters file")
        raise RuntimeError("cannot eval use default parameters")

    # evaluate(model, criterion, metric, data_loader)
    data_pred_dict = {}
    if args.model_name in ["QMAttensionMultiLayer1"]:
        ret_dict = evaluate(
            model,
            criterion,
            metric,
            data_loader,
            return_dict=True,
            return_attention=True,
        )
        average = np.mean(ret_dict["attentions"], axis=0)
        print("attentions", average, np.sum(average))
    else:
        ret_dict = evaluate(model, criterion, metric, data_loader, return_dict=True)
    # df: pd.DataFrame = pd.DataFrame(data_pred_dict)
    # df.to_csv(f"analysis/eval_pred.tsv", index=False, header=True, sep='\t')

    if return_metric:
        return ret_dict["metrics"]

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
            post = postprocess(test_examples, y_preds_post, debug=True).astype(
                np.float32
            )

            post_preds = np.array([[1.0, 0] if lb == 0 else [0.0, 1] for lb in post])
            correct = metric.compute(
                paddle.to_tensor(post_preds), paddle.to_tensor(labels)
            )
            metric.update(correct)
            accu = metric.accumulate()

            if isinstance(metric, paddle.metric.Accuracy):
                logger.info("dev_loss: {:.4}, accuracy: {:.4}".format(total_loss, accu))
            elif isinstance(metric, AccuracyAndF1):
                logger.info(
                    f"only pinyin pp: dev_loss: {total_loss:.4f}, acc: {accu[0]:.4f}, precision: {accu[1]:.4f}, recall: {accu[2]:.4f}, f1: {accu[3]:.4f} acc and f1: {accu[4]:.4f}"
                )
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
            logger.info(
                f"p0={p0}: dev_loss: {total_loss:.4f}, acc: {accu[0]:.4f}, precision: {accu[1]:.4f}, recall: {accu[2]:.4f}, f1: {accu[3]:.4f} acc and f1: {accu[4]:.4f}"
            )
        metric.reset()

        # comput prob ajust and pinyin correction.
        y_preds_post = np.argmax(oof_test, axis=1)
        # post = postprocess(test_examples, y_preds_post, debug=(i==0)).astype(np.float32)
        post = postprocess(test_examples, y_preds_post).astype(np.float32)

        post_preds = np.array([[1.0, 0] if lb == 0 else [0.0, 1] for lb in post])
        correct = metric.compute(paddle.to_tensor(post_preds), paddle.to_tensor(labels))
        metric.update(correct)
        accu = metric.accumulate()

        if isinstance(metric, paddle.metric.Accuracy):
            logger.info("dev_loss: {:.5}, accuracy: {:.5}".format(total_loss, accu))
        elif isinstance(metric, AccuracyAndF1):
            logger.info(
                f"prob adjust + pinyin: dev_loss: {total_loss:.4f}, acc: {accu[0]:.4f}, precision: {accu[1]:.4f}, recall: {accu[2]:.4f}, f1: {accu[3]:.4f} acc and f1: {accu[4]:.4f}"
            )
        metric.reset()


if __name__ == "__main__":
    if True:
        eval_test()
    else:
        args = parse_args_predict()

        paddle.set_device(args.device)

        pretrained_model = AutoModel.from_pretrained(args.plm_name)
        tokenizer = AutoTokenizer.from_pretrained(args.plm_name)

        test_data_loader = create_test_data_loader(
            args.input_file,
            tokenizer,
            max_seq_length=args.max_seq_length,
            batch_size=args.eval_batch_size,
        )
        model = get_model(pretrained_model, args)

        if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
            state_dict = paddle.load(args.init_from_ckpt)
            model.set_dict(state_dict)
            print("Loaded parameters from %s" % args.init_from_ckpt)
        else:
            raise ValueError(
                "Please set --params_path with correct pretrained model file"
            )

        eval_write(model, test_data_loader, args.result_file)
