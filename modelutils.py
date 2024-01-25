import random, os
import numpy as np
import paddle
import paddle.nn.functional as F
import paddle.nn as nn
from paddlenlp.metrics import AccuracyAndF1
from paddlenlp.transformers import AutoModel, AutoTokenizer
from loguru import logger
from model import QuestionMatching, QMAttensionMultiLayer, QMAttensionMultiLayer1, QMAttensionMultiLayer2, QMAttensionMultiLayerCnn, QuestionMatchingDist


# yapf: enable
def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


def label_smooth_loss(scores, labels, loss_func):
    """label smooth loss"""
    labels = paddle.squeeze(labels)
    labels = F.one_hot(labels, num_classes=2)
    labels = F.label_smooth(labels, epsilon=0.1)
    labels = paddle.squeeze(labels, axis=1)
    loss = loss_func(scores, labels)
    return loss


class LabelSmoothingCrossEntropy(nn.Layer):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):

        confidence = 1. - self.smoothing
        log_probs = F.log_softmax(pred, axis=-1)
        idx = paddle.stack([paddle.arange(log_probs.shape[0]), target], axis=1)
        nll_loss = paddle.gather_nd(-log_probs, index=idx)
        smooth_loss = paddle.mean(-log_probs, axis=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss

        return loss.mean()


def nnlloss_on_logits(logits, labels, loss_func):
    return loss_func(F.log_softmax(logits), labels)


@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader, return_dict=False, return_attention=False):
    """
    Given a dataset, it evals model and computes the metric.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    metric.reset()
    losses = []
    total_num = 0
    logits_list = []
    labels_list = []
    input_ids_list = []
    attention_list = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        
        total_num += len(labels)
        input_ids_list += [len(x) for x in input_ids.numpy().tolist()]
        
        if return_attention:
            outputs = model(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            return_attention=return_attention,
                            do_evaluate=True)
            assert len(outputs) == 4
            attention_list.append(outputs[-1].numpy())
        else:
            outputs = model(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            do_evaluate=True)
        logits = outputs[0]
        logits_list.append(logits.numpy())

        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        labels_list.append(labels.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
    
    accu = metric.accumulate()
    total_loss = np.mean(losses)
    if isinstance(metric, paddle.metric.Accuracy):
        logger.info("dev_loss: {:.4}, accuracy: {:.4}, total_num:{}".format(
            total_loss, accu, total_num))
    elif isinstance(metric, AccuracyAndF1):
        logger.info(f"dev_loss: {total_loss:.4f}, acc: {accu[0]:.4f}, precision: {accu[1]:.4f}, recall: {accu[2]:.4f}, f1: {accu[3]:.4f} acc and f1: {accu[4]:.4f}, total_num:{total_num}")
    model.train()
    metric.reset()
    if return_dict and return_attention:
        return {"loss": total_loss, "metrics": accu, "logits": np.concatenate(logits_list, axis=0), "labels": np.concatenate(labels_list, axis=0), 'ids': [str(_ids) for _ids in input_ids_list], 'attentions': np.concatenate(attention_list, axis=0)}
    elif return_dict:
        return {"loss": total_loss, "metrics": accu, "logits": np.concatenate(logits_list, axis=0), "labels": np.concatenate(labels_list, axis=0), 'ids': [str(_ids) for _ids in input_ids_list]}
    return accu


def predict(model, data_loader):
    """
    Predicts the data labels.

    Args:
        model (obj:`QuestionMatching`): A model to calculate whether the question pair is semantic similar or not.
        data_loaer (obj:`List(Example)`): The processed data ids of text pair: [query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids]
    Returns:
        results(obj:`List`): cosine similarity of text pairs.
    """
    batch_logits = []

    model.eval()

    with paddle.no_grad():
        for batch_data in data_loader:
            input_ids, token_type_ids = batch_data

            input_ids = paddle.to_tensor(input_ids)
            token_type_ids = paddle.to_tensor(token_type_ids)

            output = model(input_ids=input_ids, token_type_ids=token_type_ids, do_evaluate=True) # 可能输出tuple的长度不一样。
            batch_logit = output[0]

            batch_logits.append(batch_logit.numpy())

        batch_logits = np.concatenate(batch_logits, axis=0)

        return batch_logits


def get_model(pretrained_model=None, args=None):
    if pretrained_model is None:
        pretrained_model = AutoModel.from_pretrained(args.plm_name)
    
    if args.model_name == 'QMAttensionMultiLayer':
        model = QMAttensionMultiLayer(
            pretrained_model, rdrop_coef=args.rdrop_coef, dropout=args.dropout_qm, num_dropout=args.num_dropout)
    elif args.model_name == 'QMAttensionMultiLayer1':
        model = QMAttensionMultiLayer1(
            pretrained_model, rdrop_coef=args.rdrop_coef, dropout=args.dropout_qm, num_dropout=args.num_dropout, attension_type=args.attension, use_cls=args.use_cls, rnn=args.rnn, att_rnn=args.att_rnn)
        # paddle.summary(model, input_size=[(1, args.max_seq_length), (1, args.max_seq_length)], dtypes=['int64', 'int64'])
    elif args.model_name == 'QMAttensionMultiLayer2':
        model = QMAttensionMultiLayer2(
            pretrained_model, rdrop_coef=args.rdrop_coef, dropout=args.dropout_qm, num_dropout=args.num_dropout, attension_type=args.attension, use_cls=args.use_cls)
    elif args.model_name == 'QuestionMatching':
        model = QuestionMatching(pretrained_model, rdrop_coef=args.rdrop_coef, dropout=args.dropout_qm)
    elif args.model_name == 'QuestionMatchingDist':
        model = QuestionMatchingDist(pretrained_model, rdrop_coef=args.rdrop_coef, dropout=args.dropout_qm)
    elif args.model_name == 'QMAttensionMultiLayerCnn':
        model = QMAttensionMultiLayerCnn(pretrained_model, dropout=args.dropout_qm, seq_len=args.max_seq_length, out_channels=args.out_channels)
    return model


def calculate_params(model):
    n_train = 0
    n_non_train = 0
    for p in model.parameters():
        if p.trainable:
            n_train += np.prod(p.shape)
        else:
            n_non_train += np.prod(p.shape)
    return n_train + n_non_train, n_train, n_non_train


def freeze(model):
    for p in model.parameters():
        p.trainable = False

def unfreeze(model):
    for p in model.parameters():
        p.trainable = True
