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

import paddle
import numpy as np

import random
from paddlenlp.datasets import MapDataset, IterDataset
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from functools import partial
from paddle.io import DataLoader


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      num_workers=0,
                      trans_fn=None) -> DataLoader:
    if trans_fn:
        dataset: MapDataset = dataset.map(trans_fn)

    shuffle: bool = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(dataset,
                                                          batch_size=batch_size,
                                                          shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle)

    return DataLoader(dataset=dataset,
                      batch_sampler=batch_sampler,
                      collate_fn=batchify_fn,
                      # persistent_workers=True,
                      prefetch_factor=20,
                      num_workers=num_workers,
                      persistent_workers=True,
                      return_list=True)


class ClassQADataset(MapDataset):
    def __init__(self, data, is_train=True, sort=False, *args, **kwargs):
        self.is_train: bool = is_train
        super().__init__(data, *args, **kwargs)
        if sort:
            self.new_data = sorted(self.new_data, key=lambda x: len(
                x['query1']) + len(x['query2']), reverse=True)

    def __getitem__(self, idx):
        """
        Basic function of `MapDataset` to get sample from dataset with a given 
        index.
        """
        if self.is_train and random.random() > 0.5:
            sample = {'query1': self.new_data[idx]['query2'],
                      'query2': self.new_data[idx]['query1'], 'label': self.new_data[idx]['label']}
        else:
            sample = self.new_data[idx]
        # if idx < 50:
        #     print(idx, self.new_data[idx]['query2'], self.new_data[idx]['query1'], len(self.new_data[idx]['query1']) + len(self.new_data[idx]['query2']) )
        return self._transform(sample) if self._transform_pipline else sample


def read_text_pair(data_path, is_test=False):
    """Reads data."""
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.rstrip().split("\t")
            if is_test == False:
                if len(data) != 3:
                    print('wrong data', line)
                    continue
                yield {'query1': data[0], 'query2': data[1], 'label': data[2]}
            else:
                if len(data) != 2:
                    print('wrong data', line)
                    continue
                # yield {'query1': data[1], 'query2': data[0]}
                yield {'query1': data[0], 'query2': data[1]}


def convert_example(example, tokenizer, max_seq_length=512, pad_to_max=False, is_test=False):

    query, title = example["query1"], example["query2"]

    encoded_inputs = tokenizer(text=query,
                               text_pair=title,
                               max_seq_len=max_seq_length, pad_to_max_seq_len=pad_to_max)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    else:
        return input_ids, token_type_ids


def create_eval_dataloader(data_path, tokenizer, max_seq_length=64, eval_batch_size=64, pad_to_max=False) -> DataLoader:
    dev_ds = load_dataset(read_text_pair,
                          data_path=data_path,
                          is_test=False,
                          lazy=False)

    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         pad_to_max=pad_to_max,
                         max_seq_length=max_seq_length)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # text_pair_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # text_pair_segment
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]

    _data_loader: DataLoader = create_dataloader(dev_ds,
                                     mode='dev',
                                     batch_size=eval_batch_size,
                                     batchify_fn=batchify_fn,
                                     trans_fn=trans_func)
    return _data_loader


def create_test_data_loader(test_file_path, tokenizer, max_seq_length=64, batch_size=32, pad_to_max=False) -> DataLoader:
    trans_func = partial(convert_example,
                         tokenizer=tokenizer,
                         max_seq_length=max_seq_length,
                         pad_to_max=pad_to_max,
                         is_test=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment_ids
    ): [data for data in fn(samples)]

    test_ds = load_dataset(read_text_pair,
                           data_path=test_file_path,
                           is_test=True,
                           lazy=False)

    test_data_loader: DataLoader = create_dataloader(test_ds,
                                         mode='predict',
                                         batch_size=batch_size,
                                         batchify_fn=batchify_fn,
                                         trans_fn=trans_func)

    return test_data_loader
