# ERNIR-ATT for the problem matching robustness task


## Background

Question Matching (QM) task aims to determine whether the semantic equivalence between two natural questions is equivalent or not, which is an important research direction in the field of natural language processing. Question Matching also has high commercial value and plays an important role in information retrieval, intelligent customer service and other fields.

In recent years, although the neural network model has achieved accuracy similar to or even exceeding that of humans in some standard sets of problem matching reviews, the performance drops dramatically when dealing with real application scenarios, and it is unable to make correct judgments on simple (easy for humans to judge) problems (as shown in the figure below), which affects the product experience and also causes corresponding economic losses.

Translated with DeepL.com (free version)

|       Question1        |        Question2         | Label | Model |
| :----------------: | :------------------: | :---------: | :-----: |
|  婴儿吃什么蔬菜好  | 婴儿吃什么`绿色`蔬菜好 |      0      |    1    |
|  关于`牢房`的电视剧  |   关于`监狱`的电视剧   |      1      |    0    |
| 心率过`快`有什么问题 |  心率过`慢`有什么问题  |      0      |    1    |
| 黑色`裤子`配什么`上衣` |  黑色`上衣`配什么`裤子` |      0      |    1    |



## Quick Start

### Code Structure Description

Below is the main code structure and description of this project:
```
ERNIE-ATT-for-question-matching/
├── model.py # Matching Model Networking
├── data.py # Data reading, transformation logic for training samples
├── predict.py # Model prediction script that outputs predictions for the test set: 0,1
└── train.py # Model Training Evaluation
```

### Data preparation
This project uses the training ensemble set of the well-known LCQMC and BQ datasets as the training set and the validation ensemble set of these two datasets as the validation set.

Data Acquisition Source Statement:
```text
LCQMC and BQ corpus utilized in our experiments are owned by Harbin Institute of Technology (HIT), 
who acts as its sole distributor. Therefore, the dataset we use are owned by a third party. 
For guidance on how to request access to these datasets, please refer to the following website
: https://www.luge.ai/#/luge/dataDetail?id=14, https://www.luge.ai/#/luge/dataDetail?id=15. 
Additionally, more detailedinformation can be found in the respective papers that introduced
the LCQMC and BQ corpus, as referenced in the citation section of our paper.
```
The training set data format is 3 columns: text_a \t text_b \t label, Sample data are as follows:
```text
喜欢打篮球的男生喜欢什么样的女生    爱打篮球的男生喜欢什么样的女生  1
我手机丢了，我想换个手机    我想买个新手机，求推荐  1
大家觉得她好看吗    大家觉得跑男好看吗？    0
求秋色之空漫画全集  求秋色之空全集漫画  1
晚上睡觉带着耳机听音乐有什么害处吗？    孕妇可以戴耳机听音乐吗? 0
```
The data format of the validation set is the same as the training set, with the following samples:
```
开初婚未育证明怎么弄？  初婚未育情况证明怎么开？    1
谁知道她是网络美女吗？  爱情这杯酒谁喝都会醉是什么歌    0
男孩喝女孩的尿的故事    怎样才知道是生男孩还是女孩  0
这种图片是用什么软件制作的？    这种图片制作是用什么软件呢？    1
```

### model training
The outcome of the ERNIE-ATT-based model in this project can be reproduced by running the following command:

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

Configurable parameters can be supported:
* `train_set`: file for the training set.
* `dev_set`：Validation set data file.
* `rdrop_coef`：Optional, controls the coefficient by which the R-Drop policy regularizes KL-Loss; 
* `train_batch_size`：Optional, batch size, please adjust it in combination with the video memory, if there is insufficient video memory, please adjust this parameter down appropriately; the default is 32.
* `learning_rate`：Optionally, the maximum learning rate for Fine-tune; the default is 5e-5.
* `weight_decay`：Optional, parameter controlling the strength of the regular term to prevent overfitting, default is 0.0.
* `epochs`: The number of training rounds.
* `warmup_proption`：Optionally, the percentage of the learning rate warmup strategy, if 0.1, the learning rate will slowly grow from 0 to learning_rate during the first 10% of training steps, and then slowly decay.
* `init_from_ckpt`：Optional, model parameter path, hot start model training; default is None.
* `seed`：Optional, randomized seed, defaults to 1000.
* `device`: What device is selected for training, either cpu or gpu. if gpu is used for training then the parameter gpus specifies the GPU number.

The program will automatically train, evaluate when it runs. Also the training process will automatically save the model in the specified `save_dir`.

After each evaluation on the validation set during the training process, the program will decide whether to store the current model according to whether the evaluation indexes of the validation set are better than the previous optimal model indexes; if they are better than the previous optimal indexes of the validation set, then the current model will be stored; otherwise, it will not be stored, so that after the training process is over, the model with the largest number of steps under the model storage path corresponds to the model with the highest indexes of the validation set, and generally we choose the model with the highest indexes of the validation set to make predictions.

e.g.：
```text
checkpoints/
├── model_10000
│   ├── model_state.pdparams
│   ├── tokenizer_config.json
│   └── vocab.txt
└── ...
```

**NOTE:**
* To resume model training, set `init_from_ckpt`, e.g. `init_from_ckpt=checkpoints/model_10000/model_state.pdparams`.


### Prediction
After the training is completed, the model with the highest evaluation metrics in the validation set is automatically stored under the specified checkpoints path, and the prediction results are generated by running the following command.
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

A sample output prediction is as follows.
```text
0
1
0
1
```