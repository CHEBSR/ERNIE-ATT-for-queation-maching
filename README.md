# 问题匹配鲁棒性评测


## 背景

问题匹配（Question Matching）任务旨在判断两个自然问句之间的语义是否等价，是自然语言处理领域一个重要研究方向。问题匹配同时也具有很高的商业价值，在信息检索、智能客服等领域发挥重要作用。

近年来，神经网络模型虽然在一些标准的问题匹配评测集合上已经取得与人类相仿甚至超越人类的准确性，但在处理真实应用场景问题时，性能大幅下降，在简单（人类很容易判断）的问题上无法做出正确判断（如下图），影响产品体验的同时也会造成相应的经济损失。

|       问题1        |        问题2         | 标签(Label) | Model |
| :----------------: | :------------------: | :---------: | :-----: |
|  婴儿吃什么蔬菜好  | 婴儿吃什么`绿色`蔬菜好 |      0      |    1    |
|  关于`牢房`的电视剧  |   关于`监狱`的电视剧   |      1      |    0    |
| 心率过`快`有什么问题 |  心率过`慢`有什么问题  |      0      |    1    |
| 黑色`裤子`配什么`上衣` |  黑色`上衣`配什么`裤子` |      0      |    1    |



## 快速开始

### 代码结构说明

以下是本项目主要代码结构及说明：
```
question_matching/
├── model.py # 匹配模型组网
├── data.py # 训练样本的数据读取、转换逻辑
├── predict.py # 模型预测脚本，输出测试集的预测结果: 0,1
└── train.py # 模型训练评估
```

### 数据准备
本项目使用竞赛提供的 LCQMC、BQ 这 2 个数据集的训练集合集作为训练集，使用这 2 个数据集的验证集合集作为验证集。

运行如下命令生成本项目所使用的训练集和验证集。
```shell
cat ./data/train/LCQMC/train ./data/train/BQ/train  > train.txt
cat ./data/train/LCQMC/dev ./data/train/BQ/dev > dev.txt
```
训练集数据格式为 3 列: text_a \t text_b \t label, 样例数据如下:
```text
喜欢打篮球的男生喜欢什么样的女生    爱打篮球的男生喜欢什么样的女生  1
我手机丢了，我想换个手机    我想买个新手机，求推荐  1
大家觉得她好看吗    大家觉得跑男好看吗？    0
求秋色之空漫画全集  求秋色之空全集漫画  1
晚上睡觉带着耳机听音乐有什么害处吗？    孕妇可以戴耳机听音乐吗? 0
```
验证集的数据格式和训练集相同，样例如下:
```
开初婚未育证明怎么弄？  初婚未育情况证明怎么开？    1
谁知道她是网络美女吗？  爱情这杯酒谁喝都会醉是什么歌    0
男孩喝女孩的尿的故事    怎样才知道是生男孩还是女孩  0
这种图片是用什么软件制作的？    这种图片制作是用什么软件呢？    1
```

### 模型训练
运行如下命令，即可复现本项目中基于 ERNIE3.0-xbase 的模型效果:

```shell
$unset CUDA_VISIBLE_DEVICES
python -u -m paddle.distributed.launch --gpus "0,1,2,3" train.py \
       --train_set train.txt \
       --dev_set dev.txt \
       --device gpu \
       --eval_step 100 \
       --save_dir ./checkpoints \
       --train_batch_size 32 \
       --learning_rate 2E-5 \
       --rdrop_coef 0.0
```

可支持配置的参数：
* `train_set`: 训练集的文件。
* `dev_set`：验证集数据文件。
* `rdrop_coef`：可选，控制 R-Drop 策略正则化 KL-Loss 的系数；默认为 0.0, 即不使用 R-Drop 策略。
* `train_batch_size`：可选，批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `learning_rate`：可选，Fine-tune的最大学习率；默认为5e-5。
* `weight_decay`：可选，控制正则项力度的参数，用于防止过拟合，默认为0.0。
* `epochs`: 训练轮次，默认为3。
* `warmup_proption`：可选，学习率 warmup 策略的比例，如果 0.1，则学习率会在前 10% 训练 step 的过程中从 0 慢慢增长到 learning_rate, 而后再缓慢衰减，默认为 0.0。
* `init_from_ckpt`：可选，模型参数路径，热启动模型训练；默认为None。
* `seed`：可选，随机种子，默认为1000。
* `device`: 选用什么设备进行训练，可选cpu或gpu。如使用gpu训练则参数gpus指定GPU卡号。

程序运行时将会自动进行训练，评估。同时训练过程中会自动保存模型在指定的`save_dir`中。

训练过程中每一次在验证集上进行评估之后，程序会根据验证集的评估指标是否优于之前最优的模型指标来决定是否存储当前模型，如果优于之前最优的验证集指标则会存储当前模型，否则则不存储，因此训练过程结束之后，模型存储路径下 step 数最大的模型则对应验证集指标最高的模型, 一般我们选择验证集指标最高的模型进行预测。

如：
```text
checkpoints/
├── model_10000
│   ├── model_state.pdparams
│   ├── tokenizer_config.json
│   └── vocab.txt
└── ...
```

**NOTE:**
* 如需恢复模型训练，则可以设置`init_from_ckpt`， 如`init_from_ckpt=checkpoints/model_100/model_state.pdparams`。


### 开始预测
训练完成后，在指定的 checkpoints 路径下会自动存储在验证集评估指标最高的模型，运行如下命令开始生成预测结果:
```shell
$ unset CUDA_VISIBLE_DEVICES
python -u \
    predict.py \
    --device gpu \
    --params_path "./checkpoints/model_10000/model_state.pdparams" \
    --batch_size 128 \
    --input_file "${test_set}" \
    --result_file "predict_result"
```

输出预测结果示例如下:
```text
0
1
0
1
```