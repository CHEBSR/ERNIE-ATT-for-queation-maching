# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import paddlenlp as ppnlp
from paddlenlp.utils.log import logger


class FGM:
    """针对embedding层梯度上升干扰的对抗训练方法,Fast Gradient Method（FGM）"""

    def __init__(self, model):
        self.model = model
        self.backup = {}
        # for name, param in self.model.named_parameters():
        #     print(name, param.shape)
        logger.info(f"finish init FGM")

    def attack(self, epsilon=0.5, emb_name="word_embeddings"):
        # emb_name这个参数要换成你模型中embedding的参数名
        # emb_name: ptm.embeddings.word_embeddings.weight
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and emb_name in name:  # 检验参数是否可训练及范围
                self.backup[name] = param.numpy()  # 备份原有参数值
                grad_tensor = paddle.to_tensor(param.grad)  # param.grad是个numpy对象
                norm = paddle.norm(grad_tensor)  # norm化
                if norm != 0:
                    r_at = epsilon * grad_tensor / norm

                    new_v = param + r_at.astype("float32")
                    # print(r_at, param, r_at.astype("float32"), new_v)
                    param.set_value(new_v.astype("float32"))  # 在原有embed值上添加向上梯度干扰

    def restore(self, emb_name="word_embeddings"):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if not param.stop_gradient and emb_name in name:
                assert name in self.backup
                param.set_value(self.backup[name])  # 将原有embed参数还原
        self.backup = {}


class QuestionMatching(nn.Layer):
    def __init__(self, pretrained_model, dropout=None, rdrop_coef=0.0):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout: nn.Dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        # num_labels = 2 (similar or dissimilar)
        self.classifier = nn.Linear(self.ptm.config["hidden_size"], 2)
        self.rdrop_coef = rdrop_coef
        self.rdrop_loss = ppnlp.losses.RDropLoss()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        do_evaluate=False,
    ):

        last_hidden_state, pooled_output = self.ptm(
            input_ids, token_type_ids, position_ids, attention_mask
        )

        # cls_embedding1 = last_hidden_state[:, 0, :]
        # cls_embedding1 = paddle.mean(last_hidden_state, axis=1)  # 之前 default
        cls_embedding1 = pooled_output

        cls_embedding1 = self.dropout(cls_embedding1)
        logits1 = self.classifier(cls_embedding1)

        # For more information about R-drop please refer to this paper: https://arxiv.org/abs/2106.14448
        # Original implementation please refer to this code: https://github.com/dropreg/R-Drop
        if self.rdrop_coef > 0 and not do_evaluate:
            _, cls_embedding2 = self.ptm(
                input_ids, token_type_ids, position_ids, attention_mask
            )
            cls_embedding2 = self.dropout(cls_embedding2)
            logits2 = self.classifier(cls_embedding2)
            kl_loss = self.rdrop_loss(logits1, logits2)
        else:
            kl_loss = 0.0

        return logits1, kl_loss


class QuestionMatchingDist(nn.Layer):
    def __init__(self, pretrained_model, dropout=None, rdrop_coef=0.0):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout: nn.Dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        # num_labels = 2 (similar or dissimilar)
        self.classifier = nn.Linear(self.ptm.config["hidden_size"], 2)
        self.rdrop_coef = rdrop_coef
        self.rdrop_loss = ppnlp.losses.RDropLoss()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        do_evaluate=False,
    ):

        last_hidden_state, pooled_output = self.ptm(
            input_ids, token_type_ids, position_ids, attention_mask
        )

        # cls_embedding1 = last_hidden_state[:, 0, :]
        # cls_embedding1 = paddle.mean(last_hidden_state, axis=1)  # 之前 default
        cls_embedding1 = pooled_output

        cls_embedding1 = self.dropout(cls_embedding1)
        logits1 = self.classifier(cls_embedding1)

        # For more information about R-drop please refer to this paper: https://arxiv.org/abs/2106.14448
        # Original implementation please refer to this code: https://github.com/dropreg/R-Drop
        if self.rdrop_coef > 0 and not do_evaluate:
            _, cls_embedding2 = self.ptm(
                input_ids, token_type_ids, position_ids, attention_mask
            )
            cls_embedding2 = self.dropout(cls_embedding2)
            logits2 = self.classifier(cls_embedding2)
            kl_loss = self.rdrop_loss(logits1, logits2)
        else:
            kl_loss = 0.0

        cosine_sim = self._get_cosine(
            input_ids[:, 1:], token_type_ids[:, 1:], last_hidden_state[:, 1:, :]
        )  # remove cls token

        return logits1, kl_loss, cosine_sim

    def _get_cosine(self, input_ids, token_type_ids, last_hidden_state):
        mask_origin = np.array(
            input_ids != 0, dtype="int32"
        )  # # [batch_size, seq_length]
        mask = paddle.to_tensor(mask_origin, stop_gradient=True, dtype="float32")

        text_a_mask = (
            np.array(token_type_ids == 0, dtype="int32") * mask_origin
        )  # # [batch_size, seq_length]
        text_b_mask = np.array(
            token_type_ids == 1, dtype="int32"
        )  # # [batch_size, seq_length]

        text_a_mask = paddle.to_tensor(text_a_mask, stop_gradient=True, dtype="float32")
        text_a_mask = paddle.unsqueeze(text_a_mask, [2])  # [batch_size, seq_length, 1]
        text_b_mask = paddle.to_tensor(text_b_mask, stop_gradient=True, dtype="float32")
        text_b_mask = paddle.unsqueeze(text_b_mask, [2])  # [batch_size, seq_length, 1]

        text_a_embeding = paddle.sum(
            last_hidden_state * text_a_mask, axis=1
        ) / paddle.sum(text_a_mask, axis=1)
        text_b_embeding = paddle.sum(
            last_hidden_state * text_b_mask, axis=1
        ) / paddle.sum(text_b_mask, axis=1)

        text_a_embeding = F.normalize(text_a_embeding, p=2, axis=-1)
        text_b_embeding = F.normalize(text_b_embeding, p=2, axis=-1)
        cosine_sim = paddle.sum(text_a_embeding * text_b_embeding, axis=-1)
        # return F.relu(cosine_sim)
        return cosine_sim


class QMAttensionMultiLayer(nn.Layer):
    def __init__(self, pretrained_model, dropout=None, rdrop_coef=0.0, num_dropout=5):
        super().__init__()
        self.ptm = pretrained_model
        dropout = dropout if dropout is not None else 0.1
        # self.dropout = nn.Dropout(dropout)

        self.dropouts = [nn.Dropout(dropout) for _ in range(num_dropout)]

        # num_labels = 2 (similar or dissimilar)
        # self.classifier = nn.Linear(self.ptm.config["hidden_size"], 2)

        self.fc = nn.Linear(self.ptm.config["hidden_size"] * 2, 2)

        self.layer_attension = self.create_parameter(
            shape=[self.ptm.config["num_hidden_layers"], 1], is_bias=False
        )
        self.rdrop_coef = rdrop_coef
        self.rdrop_loss = ppnlp.losses.RDropLoss()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        labels=None,
        loss_fn=None,
        do_evaluate=False,
    ):

        # cls_embedding1 = pool_embedding
        model_output = self.ptm(
            input_ids,
            token_type_ids,
            position_ids,
            attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = model_output.last_hidden_state  # batch_size, seq_len, hidden_size
        # tuple of word embedding and then sequence of hidden state for each layers.
        all_hidden_states = model_output.hidden_states
        # word_embeding = all_hidden_states[0]  # [batch_size, seq_length, hidden_size]
        hidden_states_all_layers = paddle.stack(
            x=all_hidden_states[1:]
        )  # [num_layers, batch_size, seq_length, hidden_size]

        # batch_size = input_ids.shape[0]

        # only use the cls vector of all layers
        ht_cls = hidden_states_all_layers[
            :, :, :1, :
        ]  # [num_layers, batch_size, 1, hidden_size]
        # layer_attension = paddle.reshape(self.layer_attension, shape=[-1, 1, 1, 1])

        # atten = paddle.sum(ht_cls * layer_attension, axis=[1, 3]) # [num_layer, 1]
        # atten = F.softmax(atten, axis=0)

        atten = F.softmax(self.layer_attension, axis=0)

        feature = paddle.sum(
            ht_cls * paddle.reshape(atten, shape=[-1, 1, 1, 1]), axis=[0, 2]
        )
        f = paddle.mean(last_hidden, 1)

        feature = paddle.concat((feature, f), -1)
        # f = paddle.concat((feature, f), 1)

        drop_hs = []
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                h = self.fc(dropout(feature))
                drop_hs.append(h)
            else:
                hi = self.fc(dropout(feature))
                h = h + hi
                drop_hs.append(hi)

        return h / len(self.dropouts), drop_hs, 0


class AdditiveLinear(nn.Layer):
    def __init__(
        self, in_features, out_features, weight_attr=None, bias_attr=None, name=None
    ):
        super(AdditiveLinear, self).__init__()
        self._dtype = self._helper.get_default_dtype()
        self._weight_attr = weight_attr
        self._bias_attr = bias_attr
        self.weight = self.create_parameter(
            shape=[in_features, out_features],
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False,
        )

        self.weight1 = self.create_parameter(
            shape=[in_features, out_features],
            attr=self._weight_attr,
            dtype=self._dtype,
            is_bias=False,
        )

        self.bias = self.create_parameter(
            shape=[out_features], attr=self._bias_attr, dtype=self._dtype, is_bias=True
        )

        self.V = self.create_parameter(
            shape=[out_features], dtype=self._dtype, is_bias=False
        )

        self.name = name

    def forward(self, input, input1):
        """_summary_

        Args:
            input (_type_): hidden layers: batch_size * num_layers * hidden_size
            input1 (_type_): embedding input mean: batch_size * hidden_size

        Returns:
            attension: batch_size * num_layers
        """
        # out: batch_size * num_layers * hidden_size (output_dim)
        out = F.linear(
            x=input, weight=self.weight, bias=self.bias, name=self.name
        ) + paddle.unsqueeze(
            F.linear(x=input1, weight=self.weight1, name=self.name), axis=1
        )

        out = F.tanh(out)  # activation

        attension = paddle.matmul(out, self.V)  # batch_size * num_layers
        attension = F.softmax(attension, axis=1)
        return attension

    def extra_repr(self):
        name_str = ", name={}".format(self.name) if self.name else ""
        return "in_features={}, out_features={}, dtype={}{}".format(
            self.weight.shape[0], self.weight.shape[1], self._dtype, name_str
        )


class LocationAttension(nn.Layer):
    def __init__(
        self, in_features, out_features, weight_attr=None, bias_attr=None, name=None
    ):
        super(LocationAttension, self).__init__()

        self.fc = nn.Linear(in_features, out_features)
        self.name = name

    def forward(self, input, input1):
        """_summary_

        Args:
            input (_type_): hidden layers: batch_size * num_layers * hidden_size
            input1 (_type_): embedding input mean: batch_size * hidden_size, but ignored

        Returns:
            attension: batch_size * num_layers
        """
        # out: batch_size * num_layers * hidden_size (output_dim)
        out = self.fc(input1)
        attension = F.softmax(out, axis=1)
        return attension

    def extra_repr(self):
        name_str = ", name={}".format(self.name) if self.name else ""
        return "in_features={}, out_features={}, dtype={}{}".format(
            self.weight.shape[0], self.weight.shape[1], self._dtype, name_str
        )


class QMAttensionMultiLayer1(nn.Layer):
    def __init__(
        self,
        pretrained_model,
        dropout=None,
        rdrop_coef=0.0,
        num_dropout=5,
        attension_type="additive",
        use_cls=0,
        rnn=None,
        att_rnn=None,
    ):
        """_summary_

        Args:
            pretrained_model (_type_): _description_
            dropout (_type_, optional): _description_. Defaults to None.
            rdrop_coef (float, optional): _description_. Defaults to 0.0.
            num_dropout (int, optional): _description_. Defaults to 5.
            attension_type (str, optional): _description_. Defaults to 'additive'.
            use_cls (int, optional): _description_. Defaults to 0: not use. 1: use cls, 2: use mean
        """
        super().__init__()
        self.ptm = pretrained_model
        dropout = dropout if dropout is not None else 0.1
        # self.dropout = nn.Dropout(dropout)

        self.dropouts = nn.LayerList([nn.Dropout(dropout) for _ in range(num_dropout)])

        # num_labels = 2 (similar or dissimilar)
        # self.classifier = nn.Linear(self.ptm.config["hidden_size"], 2)
        self.use_cls = use_cls
        if use_cls == 0:
            self.fc = nn.Linear(self.ptm.config["hidden_size"], 2)
        elif use_cls in [1, 2, 3]:
            self.fc = nn.Linear(self.ptm.config["hidden_size"] * 2, 2)

        # additive attension.
        bias_attr = False
        self.output_dim = self.ptm.config["hidden_size"]
        if attension_type == "additive":
            self.add_attension = AdditiveLinear(
                self.ptm.config["hidden_size"], self.output_dim
            )
        elif attension_type == "location":
            self.add_attension = LocationAttension(
                self.ptm.config["hidden_size"], self.ptm.config["num_hidden_layers"]
            )
        self.rnn = rnn
        self.att_rnn = att_rnn
        if att_rnn == "gru":
            self.attrnn_layer = nn.GRU(
                input_size=self.ptm.config["hidden_size"],
                hidden_size=self.ptm.config["hidden_size"],
                num_layers=1,
                dropout=dropout,
            )
            logger.info(f"using {rnn}")
        elif att_rnn == "lstm":
            self.attrnn_layer = nn.LSTM(
                input_size=self.ptm.config["hidden_size"],
                hidden_size=self.ptm.config["hidden_size"],
                num_layers=1,
                dropout=dropout,
            )
        else:
            logger.info(f"not using rnn")

        if rnn == "gru":
            self.rnn_layer = nn.GRU(
                input_size=self.ptm.config["hidden_size"],
                hidden_size=self.ptm.config["hidden_size"],
                num_layers=1,
                dropout=dropout,
            )
            logger.info(f"using {rnn}")
        elif rnn == "lstm":
            self.rnn_layer = nn.LSTM(
                input_size=self.ptm.config["hidden_size"],
                hidden_size=self.ptm.config["hidden_size"],
                num_layers=1,
                dropout=dropout,
            )
            logger.info(f"using {rnn}")
        else:
            logger.info(f"not using rnn")

        # self.layer_attension = self.create_parameter(shape=[self.ptm.config['num_hidden_layers'], 1], is_bias=False)
        self.rdrop_coef = rdrop_coef
        self.rdrop_loss = ppnlp.losses.RDropLoss()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        labels=None,
        loss_fn=None,
        return_attention=False,
        do_evaluate=False,
    ):

        _attentions = None

        def _forward():
            # input_ids: list of batch_size * seq_length
            # cls_embedding1 = pool_embedding
            model_output = self.ptm(
                input_ids,
                token_type_ids,
                position_ids,
                attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = (
                model_output.last_hidden_state
            )  # batch_size, seq_len, hidden_size
            # tuple of word embedding and then sequence of hidden state for each layers.
            all_hidden_states = model_output.hidden_states
            word_embeding = all_hidden_states[
                0
            ]  # [batch_size, seq_length, hidden_size]

            mask_origin = np.array(
                input_ids != 0, dtype="int32"
            )  # # [batch_size, seq_length]
            mask = paddle.to_tensor(mask_origin, stop_gradient=True, dtype="float32")
            mask = paddle.unsqueeze(mask, [2])  # [batch_size, seq_length, 1]

            # # just use the mean as the feature of input origin. - this is probably wrong due to padding.
            # input_feature = paddle.mean(word_embeding, axis=1)

            hidden_states_all_layers = paddle.stack(
                x=all_hidden_states[1:]
            )  # [num_layers, batch_size, seq_length, hidden_size]

            # only use the cls vector of all layers
            ht_cls = hidden_states_all_layers[
                :, :, 0, :
            ]  # [num_layers, batch_size, hidden_size]
            ht_cls = paddle.transpose(
                ht_cls, [1, 0, 2]
            )  # [batch_size, num_layers, hidden_size]
            if self.rnn and self.rnn != 'none':
                ht_cls, _last_hidden = self.rnn_layer(ht_cls)

            if self.att_rnn and self.att_rnn != 'none':
                _, input_feature = self.attrnn_layer(
                    word_embeding,
                    sequence_length=paddle.sum(
                        paddle.to_tensor(mask_origin, stop_gradient=True), axis=1
                    ),
                )
                # _, input_feature = self.attrnn_layer(word_embeding)
                input_feature = paddle.squeeze(input_feature, axis=0)
            else:
                # just use the mean as the feature of input origin, but have to ignore padding token.
                input_feature = paddle.sum(word_embeding * mask, axis=1) / paddle.sum(
                    mask, axis=1
                )

            attension = self.add_attension(
                ht_cls, input_feature
            )  # batch_size * num_layers
            _attentions = attension
            attension = paddle.unsqueeze(
                attension, axis=[2]
            )  # batch_size * num_layers * 1

            feature = paddle.sum(
                ht_cls * attension, axis=[1]
            )  # batch_size * hidden_size

            if self.use_cls == 1:
                f = last_hidden[:, 0, :]
                feature = paddle.concat((feature, f), -1)
            elif self.use_cls == 2:
                f = paddle.sum(last_hidden * mask, axis=1) / paddle.sum(mask, axis=1)
                # f = paddle.mean(last_hidden, 1)
                feature = paddle.concat((feature, f), -1)
            elif self.use_cls == 3:
                feature = paddle.concat((feature, model_output.pooler_output), -1)

            drop_hs = []
            for i, dropout in enumerate(self.dropouts):
                if i == 0:
                    h = self.fc(dropout(feature))
                    drop_hs.append(h)
                else:
                    hi = self.fc(dropout(feature))
                    h = h + hi
                    drop_hs.append(hi)
            if return_attention:
                return h / len(self.dropouts), drop_hs, _attentions
            return h / len(self.dropouts), drop_hs, last_hidden[:, 0, :]

        if return_attention:
            h, drop_hs, _attentions = _forward()
        else:
            h, drop_hs, cls = _forward()

        if self.rdrop_coef > 0 and not do_evaluate:
            h1, drop_hs1, cls1 = _forward()
            kl_loss = self.rdrop_loss(h, h1)
            # kl_loss = self.rdrop_loss(cls, cls1)
        else:
            kl_loss = 0.0

        if return_attention:
            return h, drop_hs, kl_loss, _attentions

        return h, drop_hs, kl_loss
        # return h/len(self.dropouts), drop_hs, kl_loss


class QMAttensionMultiLayer2(nn.Layer):
    def __init__(
        self,
        pretrained_model,
        dropout=None,
        rdrop_coef=0.0,
        num_dropout=5,
        attension_type="additive",
        use_cls=0,
        interval=3,
        additional_loss=3,
    ):
        """_summary_

        Args:
            pretrained_model (_type_): _description_
            dropout (_type_, optional): _description_. Defaults to None.
            rdrop_coef (float, optional): _description_. Defaults to 0.0.
            num_dropout (int, optional): _description_. Defaults to 5.
            attension_type (str, optional): _description_. Defaults to 'additive'.
            use_cls (int, optional): _description_. Defaults to 0: not use. 1: use cls, 2: use mean
        """
        super().__init__()
        self.ptm = pretrained_model
        dropout = dropout if dropout is not None else 0.1
        # self.dropout = nn.Dropout(dropout)

        self.dropouts = [nn.Dropout(dropout) for _ in range(num_dropout)]

        # num_labels = 2 (similar or dissimilar)
        # self.classifier = nn.Linear(self.ptm.config["hidden_size"], 2)
        self.use_cls = use_cls
        if use_cls == 0:
            self.fc = nn.Linear(self.ptm.config["hidden_size"], 2)
        elif use_cls in [1, 2]:
            self.fc = nn.Linear(self.ptm.config["hidden_size"] * 2, 2)

        # additive attension.
        bias_attr = False
        self.output_dim = self.ptm.config["hidden_size"]
        if attension_type == "additive":
            self.add_attension = AdditiveLinear(
                self.ptm.config["hidden_size"], self.output_dim
            )
        elif attension_type == "location":
            self.add_attension = LocationAttension(
                self.ptm.config["hidden_size"], self.ptm.config["num_hidden_layers"]
            )

        # self.layer_attension = self.create_parameter(shape=[self.ptm.config['num_hidden_layers'], 1], is_bias=False)
        self.rdrop_coef = rdrop_coef
        self.rdrop_loss = ppnlp.losses.RDropLoss()

        self.interval = interval
        self.additional_loss = additional_loss
        self.additional_fcs = [
            nn.Linear(self.ptm.config["hidden_size"], 2)
            for _ in range(self.additional_loss)
        ]

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        labels=None,
        loss_fn=None,
        do_evaluate=False,
    ):
        def _forward():
            # input_ids: list of batch_size * seq_length
            # cls_embedding1 = pool_embedding
            model_output = self.ptm(
                input_ids,
                token_type_ids,
                position_ids,
                attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = (
                model_output.last_hidden_state
            )  # batch_size, seq_len, hidden_size
            # tuple of word embedding and then sequence of hidden state for each layers.
            all_hidden_states = model_output.hidden_states
            # [batch_size, seq_length, hidden_size]
            word_embeding = all_hidden_states[0]

            # [batch_size, seq_length]
            mask = np.array(input_ids != 0, dtype="float32")
            mask = paddle.to_tensor(mask, stop_gradient=True)
            mask = paddle.unsqueeze(mask, [2])  # [batch_size, seq_length, 1]

            # just use the mean as the feature of input origin, but have to ignore padding token.
            input_feature = paddle.sum(word_embeding * mask, axis=1) / paddle.sum(
                mask, axis=1
            )

            # # just use the mean as the feature of input origin. - this is probably wrong due to padding.
            # input_feature = paddle.mean(word_embeding, axis=1)

            # [num_layers, batch_size, seq_length, hidden_size]
            hidden_states_all_layers = paddle.stack(x=all_hidden_states[1:])

            # only use the cls vector of all layers
            # [num_layers, batch_size, hidden_size]
            ht_cls = hidden_states_all_layers[:, :, 0, :]

            auxi_loss = 0
            if not do_evaluate:
                for i in range(len(self.additional_fcs)):
                    layer_id = -(i + 1) * self.interval
                    ht_f = ht_cls[layer_id, :, :]  # bs_size * hidden_size
                    logit = self.additional_fcs[i](ht_f)
                    auxi_loss += loss_fn(logit, labels)

            # [batch_size, num_layers, hidden_size]
            ht_cls = paddle.transpose(ht_cls, [1, 0, 2])

            attension = self.add_attension(
                ht_cls, input_feature
            )  # batch_size * num_layers

            # batch_size * num_layers * 1
            attension = paddle.unsqueeze(attension, axis=[2])

            # batch_size * hidden_size
            feature = paddle.sum(ht_cls * attension, axis=[1])

            if self.use_cls == 1:
                f = last_hidden[:, 0, :]
                feature = paddle.concat((feature, f), -1)
            elif self.use_cls == 2:
                f = paddle.sum(last_hidden * mask, axis=1) / paddle.sum(mask, axis=1)
                # f = paddle.mean(last_hidden, 1)
                feature = paddle.concat((feature, f), -1)

            drop_hs = []
            for i, dropout in enumerate(self.dropouts):
                if i == 0:
                    h = self.fc(dropout(feature))
                    drop_hs.append(h)
                else:
                    hi = self.fc(dropout(feature))
                    h = h + hi
                    drop_hs.append(hi)

            return h / len(self.dropouts), drop_hs, auxi_loss

        h, drop_hs, auxi_loss = _forward()

        if self.rdrop_coef > 0 and not do_evaluate:
            h1, _, auxi_loss = _forward()
            kl_loss = self.rdrop_loss(h, h1) + 0.1 * auxi_loss
        else:
            kl_loss = 0.1 * auxi_loss

        return h, drop_hs, kl_loss
        # return h/len(self.dropouts), drop_hs, kl_loss


class QMAttensionMultiLayerCnn(nn.Layer):
    def __init__(
        self, pretrained_model, dropout=None, seq_len=64, out_channels=0, rdrop_coef=0.0
    ):
        """_summary_

        Args:
            pretrained_model (_type_): _description_
            dropout (_type_, optional): _description_. Defaults to None.
            rdrop_coef (float, optional): _description_. Defaults to 0.0.
            num_dropout (int, optional): _description_. Defaults to 5.
            attension_type (str, optional): _description_. Defaults to 'additive'.
            use_cls (int, optional): _description_. Defaults to 0: not use. 1: use cls, 2: use mean

            https://www.paddlepaddle.org.cn/documentation/docs/en/api/paddle/nn/Conv2D_en.html#conv2d
        """
        super().__init__()
        self.ptm = pretrained_model
        in_channels = self.ptm.config["num_hidden_layers"]
        out_channels = (
            self.ptm.config["num_hidden_layers"] if out_channels < 1 else out_channels
        )
        # out_channels = 10
        conv_kernel_sizes = (3, self.ptm.config["hidden_size"])
        padding = (1, 1)
        strides = (1, 1)
        out_h = int(
            (seq_len + 2 * padding[0] - ((conv_kernel_sizes[0] - 1) + 1)) / 1 + 1
        )
        out_w = 1

        pool_kernel_size = (3, 3)

        dropout = dropout if dropout is not None else 0.1
        # self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=conv_kernel_sizes,
            padding=padding,
        )
        # bs * out_c * （seq_len - 3 + 1) * 1    -- padding=0
        # bs * out_c * （seq_len - 3 + 1) * 3    -- padding=1

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2D(kernel_size=pool_kernel_size, stride=strides)
        # pool: bs * out_c * (out_h - 3 + 1) * 1
        self.dropout = nn.Dropout(dropout)
        self.flat = nn.Flatten()
        linear_in = out_channels * (out_h - pool_kernel_size[0] + 1)
        self.fc = nn.Linear(linear_in, 2)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None,
        labels=None,
        loss_fn=None,
        do_evaluate=False,
    ):

        # input_ids: list of batch_size * seq_length
        # cls_embedding1 = pool_embedding
        model_output = self.ptm(
            input_ids,
            token_type_ids,
            position_ids,
            attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # last_hidden = model_output.last_hidden_state  # batch_size, seq_len, hidden_size
        # tuple of word embedding and then sequence of hidden state for each layers.
        all_hidden_states = model_output.hidden_states
        # [batch_size, seq_length, hidden_size]
        # word_embeding = all_hidden_states[0]

        # [batch_size, seq_length]
        mask = np.array(input_ids != 0, dtype="float32")
        mask = paddle.to_tensor(mask, stop_gradient=True)
        mask = paddle.unsqueeze(mask, [2])  # [batch_size, seq_length, 1]
        mask = paddle.unsqueeze(mask, [0])  # [1, batch_size, seq_length, 1]

        # [num_layers, batch_size, seq_length, hidden_size]
        hidden_states_all_layers = paddle.stack(x=all_hidden_states[1:])

        # x = paddle.transpose(hidden_states_all_layers, [1, 0, 2, 3])
        x = paddle.transpose(hidden_states_all_layers * mask, [1, 0, 2, 3])

        x = self.pool(self.dropout(self.relu(self.conv(self.dropout(x)))))
        x = self.fc(self.dropout(self.flat(self.dropout(x))))

        h, drop_hs, auxi_loss = x, [x], 0

        # if self.rdrop_coef > 0 and not do_evaluate:
        #     h1, _, auxi_loss = _forward()
        #     kl_loss = self.rdrop_loss(h, h1) + 0.1 * auxi_loss
        # else:
        #     kl_loss = 0.1 * auxi_loss

        return h, drop_hs, auxi_loss
