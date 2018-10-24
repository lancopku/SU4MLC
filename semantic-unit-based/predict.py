import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import models
import data.dataloader as dataloader
import data.utils as utils
import data.dict as dict
from optims import Optim

from optims import Optim
import lr_scheduler as L

import os
import argparse
import time
import math
import json
import collections
import codecs
import numpy as np

#config
parser = argparse.ArgumentParser(description='train.py')
parser.add_argument('-config', default='config.yaml', type=str,
                    help="config file")
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
parser.add_argument('-restore', default='/home/yangpengcheng/s2s/data/data/log/2018-02-03-09:59:59/best_micro_f1_checkpoint.pt', type=str,
                    help="restore checkpoint")
parser.add_argument('-seed', type=int, default=1234,
                    help="Random seed")
parser.add_argument('-model', default='seq2seq', type=str,
                    help="Model selection")
parser.add_argument('-score', default='', type=str,
                    help="score_fn")
parser.add_argument('-pretrain', action='store_true',
                    help="load pretrain embedding")
parser.add_argument('-limit', type=int, default=0,
                    help="data limit")
parser.add_argument('-log', default='predict', type=str,
                    help="log directory")
parser.add_argument('-unk', action='store_true',
                    help="replace unk")
parser.add_argument('-memory', action='store_true',
                    help="memory efficiency")
parser.add_argument('-beam_size', type=int, default=1,
                    help="beam search size")
parser.add_argument('-label_dict_file', default='/home/yangpengcheng/seq2seq/data/data/topic_sorted.json', type=str,
                    help="label_dict")

opt = parser.parse_args()
config = utils.read_config(opt.config)
torch.manual_seed(opt.seed)

#checkpoint
if opt.restore:
    print('loading checkpoint...\n')
    checkpoints = torch.load(opt.restore)
    config = checkpoints['config']

# cuda
#use_cuda = torch.cuda.is_available() and len(opt.gpus) > 0
use_cuda = True
if use_cuda:
    #torch.cuda.set_device(opt.gpus[0])
    torch.cuda.manual_seed(opt.seed)

#data
print('loading data...\n')
start_time = time.time()
datas = torch.load(config.data)
print('loading time cost: %.3f' % (time.time()-start_time))

testset = datas['valid']

src_vocab, tgt_vocab = datas['dicts']['src'], datas['dicts']['tgt']
config.src_vocab = src_vocab.size()
config.tgt_vocab = tgt_vocab.size()
testloader = dataloader.get_loader(testset, batch_size=config.batch_size, shuffle=False, num_workers=2)

if opt.pretrain:
    pretrain_embed = torch.load(config.emb_file)
else:
    pretrain_embed = None
# model
print('building model...\n')
# getattr()为返回一个对象的属性值
# 如models是一个大类, opt.model为其中的一个属性(seq2seq).
model = getattr(models, opt.model)(config, src_vocab.size(), tgt_vocab.size(), use_cuda,
                       pretrain=pretrain_embed, score_fn=opt.score)

if opt.restore:
    model.load_state_dict(checkpoints['model'])
if use_cuda:
    model.cuda()
if len(opt.gpus) > 1:
    model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)

# optimizer
if opt.restore:
    optim = checkpoints['optim']
else:
    optim = Optim(config.optim, config.learning_rate, config.max_grad_norm,
                  lr_decay=config.learning_rate_decay, start_decay_at=config.start_decay_at)
optim.set_parameters(model.parameters())
if config.schedule:
    scheduler = L.CosineAnnealingLR(optim.optimizer, T_max=config.epoch)

# 总共有多少个参数.
param_count = 0
for param in model.parameters():
    param_count += param.view(-1).size()[0]


# ======================================================================================================================
'''log config'''

# config.log是记录的文件夹, 最后一定是/
# opt.log是此次运行时记录的文件夹的名字
if not os.path.exists(config.log):
    os.mkdir(config.log)
if opt.log == '':
    log_path = config.log + utils.format_time(time.localtime()) + '/'
else:
    log_path = config.log + opt.log + '/'
if not os.path.exists(log_path):
    os.mkdir(log_path)
logging = utils.logging(log_path+'model_config.txt')  # 往这个文件里写记录
logging_csv = utils.logging_csv(log_path+'record.csv') # 往这个文件里写记录
for k, v in config.items():
    logging("%s:\t%s\n" % (str(k), str(v)))
logging("\n")
logging(repr(model)+"\n\n")  # 记录这个文件的框架

logging('total number of parameters: %d\n\n' % param_count)

# updates是已经进行了几个epoch, 防止中间出现程序中断的情况.
if opt.restore:
    updates = checkpoints['updates']
else:
    updates = 0

with open(opt.label_dict_file, 'r') as f:
    label_dict = json.load(f)

def eval(epoch):
    model.eval()
    reference, candidate, source, alignments = [], [], [], []
    for raw_src, src, src_len, raw_tgt, tgt, tgt_len in testloader:
        if len(opt.gpus) > 1:
            samples, alignment = model.module.sample(src, src_len)
        else:
            samples, alignment = model.beam_sample(src, src_len, beam_size=config.beam_size)

        candidate += [tgt_vocab.convertToLabels(s, dict.EOS) for s in samples]
        source += raw_src
        reference += raw_tgt
        alignments += [align for align in alignment]

        # candidate为预测出来的结果, [[],[],[],[]]
        # 为一个二重列表的形式.
        # 大列表的长度为预测样本的个数.
        # 每一个元素都是一个列表, 为预测出来的样本, 没有<BOS>, <EOS>和<PAD>, 每一个都是真实的单词，不是索引
        # source也是一样

    # 如果预测出unk的话, 用出现次数最多的字符替换它, 一般为True.
    if opt.unk:
        cands = []
        for s, c, align in zip(source, candidate, alignments):
            cand = []
            for word, idx in zip(c, align):
                if word == dict.UNK_WORD and idx < len(s):
                    try:
                        cand.append(s[idx])
                    except:
                        cand.append(word)
                        print("%d %d\n" % (len(s), idx))
                else:
                    cand.append(word)
            cands.append(cand)
        candidate = cands

    score = {}
    result = utils.eval_metrics(reference, candidate, label_dict, log_path)
    logging_csv([result['one_error'], result['hamming_loss'], \
                result['macro_f1'], result['macro_precision'], result['macro_recall'],\
                result['micro_f1'], result['micro_precision'], result['micro_recall']])
    print('one_error: %.4f | hamming_loss: %.8f | macro_f1: %.4f | micro_f1: %.4f'
          % (result['one_error'], result['hamming_loss'], result['macro_f1'], result['micro_f1']))
    

if __name__ == '__main__':
    eval(0)
