# encoding: utf-8
import warnings
import numpy as np


class DefaultConfig(object):
    seed = 0

    # dataset options
    dataset = 'testData'
    datatype = 'person'
    mode = 'retrieval'
    # optimization options
    loss = 'triplet'
    optim = 'adam'
    max_epoch = 60
    train_batch = 32 
    test_batch = 32
    adjust_lr = False
    lr = 0.0001

    gamma = 0.1
    weight_decay = 5e-4
    momentum = 0.9
    random_crop = False
    margin = None
    num_instances = 4
    num_gpu = 1
    evaluate = False
    savefig = None 
    re_ranking = False

    # model options
    model_name = 'bfe'  # triplet, softmax_triplet, bfe, ide
    last_stride = 2
    pretrained_model = None
    
    continue_train = False
    continue_model = None

    # miscs
    print_freq = 20
    eval_step = 20
    save_dir = './pytorch-ckpt/Market'
    workers = 0
    start_epoch = 0
    best_rank = -np.inf

    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
            if 'cls' in self.dataset:
                self.mode='class'
            if 'market' in self.dataset or 'cuhk' in self.dataset or 'duke' in self.dataset:
                self.datatype = 'person'
            elif 'cub' in self.dataset:
                self.datatype = 'cub'
            elif 'car' in self.dataset:
                self.datatype = 'car'
            elif 'clothes' in self.dataset:
                self.datatype = 'clothes'
            elif 'product' in self.dataset:
                self.datatype = 'product'

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in DefaultConfig.__dict__.items()
                if not k.startswith('_')}

opt = DefaultConfig()
