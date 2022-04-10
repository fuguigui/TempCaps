#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os
import gc
import datetime
# import pynvml
import logging

import torch
import numpy as np
import sys
from collections.abc import Mapping

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

logger = logging.getLogger(__name__)

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def update(self, **new_dict):
        self.__dict__.update(new_dict)


def count_params(model):
    logger.info("****** Counting parameters number ******")
    ttl_num = 0
    for p in model.state_dict():
        size = model.state_dict()[p].size()
        numel = model.state_dict()[p].numel()
        logger.info("\tParam: {}, shape: {}, number: {}".format(p, size, numel))
        ttl_num += numel

    logger.info("Parameter counts: {}".format(ttl_num))

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1]),  int(line_split[2])


def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


def _get_tensors():
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            tensor = obj
        else:
            continue
        if tensor.is_cuda:
            yield tensor

def _write_log(write_str):
    logger.info(write_str)
#
# def gpu_memory_log(device=0, level="epoch"):
#     stack_layer = 1
#     func_name = sys._getframe(stack_layer).f_code.co_name
#     file_name = sys._getframe(stack_layer).f_code.co_filename
#     line = sys._getframe(stack_layer).f_lineno
#     now_time = datetime.datetime.now()
#     log_format = 'LINE:%s, FUNC:%s, FILE:%s, TIME:%s, CONTENT:%s'
#
#     pynvml.nvmlInit()
#     handle = pynvml.nvmlDeviceGetHandleByIndex(device)
#     meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
#
#
#     ts_list = [tensor.size() for tensor in _get_tensors()]
#     new_tensor_sizes = {(type(x),
#                          tuple(x.size()),
#                          ts_list.count(x.size()),
#                          np.prod(np.array(x.size()))*4/1024**2)
#                          for x in _get_tensors()}
#
#     if level == "epoch":
#         write_str = log_format % (line, func_name, file_name, now_time, "")
#         logger.info(write_str)
#         for t, s, n, m in new_tensor_sizes:
#             logger.info('[tensor: %s * Size:%s | Memory: %s M | %s]' %(str(n), str(s), str(m*n)[:6], str(t)))
#         logger.info("memory_allocated:%f Mb" % float(torch.cuda.memory_allocated()/1024**2))
#         logger.info("max_memory_allocated:%f Mb" % float(torch.cuda.max_memory_allocated()/1024**2))
#         logger.info("memory_reserved:%f Mb" % float(torch.cuda.memory_reserved()/1024**2))
#         logger.info("max_memory_reserved:%f Mb" % float(torch.cuda.max_memory_reserved()/1024**2))
#         logger.info("Used Memory:%f Mb" % float(meminfo.used/1024**2))
#         logger.info("Free Memory:%f Mb" % float(meminfo.free/1024**2))
#         logger.info("Total Memory:%f Mb" % float(meminfo.total/1024**2))
#
#     pynvml.nvmlShutdown()
#     return float(meminfo.used/1024**2)


settings = {
    'main_dirName': None,
    'time_scale' : 24, # 24 for ICEWS dataset, 1 for GDELT dataset.
    'CI': None, # confidencial interval, .5 for ICEWS and 1 for GDELT.
    'time_horizon': 50,  # horizon by time prediction.
    'embd_rank': 200,  # hidden dimension of entity/rel embeddings
    'max_hist_len': 100, #maximum history sequence length for get_history
    'cut_pos': 10, #cuttoff position by prediction
}

def load_quadruples(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(int(line_split[3])/settings['time_scale'])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
    times = list(times)
    times.sort()
    return np.asarray(quadrupleList), np.asarray(times)


def isListEmpty(inList):
    if isinstance(inList, list):
        return all(map(isListEmpty, inList))
    return False

def to_device(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor



def str2bool(v: str) -> bool:
    v = v.lower()
    if v == "true":
        return True
    elif v == "false":
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected, got" + str(v) + ".")




class ArgumentGroup(object):
    def __init__(self, parser, title, des):
        self._group = parser.add_argument_group(title=title, description=des)

    def add_arg(self, name, type, default, help, **kwargs):
        type = str2bool if type == bool else type
        self._group.add_argument(
            "--" + name,
            default=default,
            type=type,
            help=help + ' Default: %(default)s.',
            **kwargs)



def collate_fn(batch):
    return list(zip(*batch))


def big_logging(description):
    logger.info("\n")
    logger.info("".join(["*"] * (16 + len(description))))
    logger.info("\t\t{}\t\t".format(description))
    logger.info("".join(["*"] * (16 + len(description))))

def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float() #+ 1 # add 1 is to add self-loop
    # in_deg[torch.nonzero(in_deg == 0, as_tuple=False).view(-1)] = 1
    norm = 1.0 / in_deg
    return norm
