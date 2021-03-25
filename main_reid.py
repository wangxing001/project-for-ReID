# encoding: utf-8
import os
import sys
from os import path as osp
from pprint import pprint

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
# from prefetch_generator import BackgroundGenerator

from config import opt
from datasets import data_manager
from datasets.data_loader import ImageData, ImageData1, ImageData2
from datasets.samplers import RandomIdentitySampler
from models.networks import ResNetBuilder, IDE, Resnet, BFE
from trainers.evaluator import ResNetEvaluator
from trainers.trainer import cls_tripletTrainer
from utils.loss import CrossEntropyLabelSmooth, TripletLoss, Margin
from utils.LiftedStructure import LiftedStructureLoss
from utils.DistWeightDevianceLoss import DistWeightBinDevianceLoss
from utils.serialization import Logger, save_checkpoint
from utils.transforms import TestTransform, TrainTransform
from shutil import copyfile
import socket
from _thread import *

def train(**kwargs):
    opt._parse(kwargs)
    # set random seed and cudnn benchmark
    torch.manual_seed(opt.seed)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.makedirs(opt.save_dir, exist_ok=True)
    use_gpu = torch.cuda.is_available()
    sys.stdout = Logger(osp.join(opt.save_dir, 'log_train.txt'))

    print('=========user config==========')
    pprint(opt._state_dict())
    print('============end===============')

    if use_gpu:
        print('currently using GPU')
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(opt.seed)
    else:
        print('currently using cpu')
    pin_memory = True if use_gpu else False
    summary_writer = SummaryWriter(osp.join(opt.save_dir, 'tensorboard_log'))
    # -------------- model and parameter loading ------------------
    print('initializing model ...')
    if opt.model_name == 'bfe':
        if opt.datatype == "person":
            model = BFE(751, 1.0, 0.33)
        else:
            model = BFE(751, 0.5, 0.5)

    optim_policy = model.parameters()

    if opt.pretrained_model:
        state_dict = torch.load(opt.pretrained_model)['state_dict']
        model.load_state_dict(state_dict, False)
        print('load pretrained model ' + opt.pretrained_model)
             
    print('model size: {:.5f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))
    if use_gpu:
        model = nn.DataParallel(model).cuda()
        # model.cuda()
    reid_evaluator = ResNetEvaluator(model)
    # -------------------------- load end -------------------------
    def handleDataset():
        print('initializing dataset {}'.format(opt.dataset))
        # 之前不需要动
        dataset = data_manager.init_dataset(name=opt.dataset, mode=opt.mode)

        # for query images
        queryloader = DataLoader(
            ImageData(dataset.query, TestTransform(opt.datatype)),
            batch_size=opt.test_batch, num_workers=opt.workers, # test_batch = 1
            pin_memory=pin_memory
        )

        # for target image
        galleryloader = DataLoader(
            ImageData(dataset.target, TestTransform(opt.datatype)),
            batch_size=opt.test_batch, num_workers=opt.workers,
            pin_memory=pin_memory
        )

        queryFliploader = DataLoader(
            ImageData(dataset.query, TestTransform(opt.datatype, True)),
            batch_size=opt.test_batch, num_workers=opt.workers,
            pin_memory=pin_memory
        )

        galleryFliploader = DataLoader(
            ImageData(dataset.target, TestTransform(opt.datatype, True)),
            batch_size=opt.test_batch, num_workers=opt.workers,
            pin_memory=pin_memory
        )

        return queryloader, galleryloader, queryFliploader, galleryFliploader

    # def deleteDirImage():

    def recv_and_send_data(clnt_sock):
        # 循环接收和发送数据
        strSend = 'Please send messages to me... \n'
        strSend = strSend.encode()
        clnt_sock.send(strSend)
        print("send successfully")
        while True:
            recv_data = clnt_sock.recv(1024)
            
            queryloader, galleryloader, queryFliploader, galleryFliploader = handleDataset()
            cmc = reid_evaluator.evaluate(queryloader, galleryloader, queryFliploader, 
                galleryFliploader, re_ranking=opt.re_ranking, savefig=opt.savefig)
            # reply 需要换成对应的数据
            if recv_data:
                reply = 'cmc : ' + str(cmc) # cmc is numpy.floate
                clnt_sock.sendall(reply.encode())  
                # 删除文件夹数据
                # deleteDirImage
            else:
                break
        clnt_sock.close()

    # ------------------------ start TCP ----------------------------
    serv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("socket creating...")
    # bind
    try:
        serv_sock.bind(('127.0.0.1', 8801))
    except socket.error:
        print("Bind failed ")
        sys.exit()
    print("socket bind successfully")
    # listen
    serv_sock.listen(10)
    print("socket start listening")
    # accept
    while True:
        clnt_sock, clnt_addr = serv_sock.accept()
        print("Connected to IP:port —— ", clnt_addr[0], ' : ', str(clnt_addr[1]))
        # core part
        start_new_thread(recv_and_send_data, (clnt_sock,)) # 元组形式

    # close
    serv_sock.close()
    # ------------------------ close TCP -----------------------------


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    import fire
    fire.Fire()
