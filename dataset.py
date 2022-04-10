# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
import logging
import pickle
import torch
import numpy as np
import os
from torch.utils.data import Dataset
from collections import defaultdict
from datetime import datetime
from utils import unique_rows
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

logger = logging.getLogger(__name__)

class StaticKGDataset(Dataset):
    def __init__(self, params, device=None):
        logger.info(" ****** Creating Static Knowledge Graph Dataset for {} ******".format(params.dataset))
        self.parse_params(params)
        self.device = device

        self.data = dict()
        self.len = dict()
        self.ent2id = {}
        self.rel2id = {}
        self.all_facts_as_tuples = set()

        if not self.overwrite:
            try:
                self.load_processed()
            except:
                logger.warning("No preprocessed files to load! Doing the preprocess")

        if len(self.data.keys()) < 1:
            self.preprocess()
            self.save_processed()

    def parse_params(self, params):
        self.overwrite = params.overwrite
        self.file_path = params.data_dir
        self.dataset_name = params.dataset
        self.types = params.dataset_types
        self.neg_ratio = params.neg_ratio
        self.num_e = params.num_e
        self.num_r = params.num_r
        self.mode = params.data_mode
        self.measure_relation = getattr(params, 'measure_relation', False)
        self.prefix = "StaticKG_" + self.mode
        if self.mode == "test":
            # self.train_types = params.dataset_types[:-1]
            self.train_types = ['train']
            self.eval_type = "test"
        else:
            self.train_types = ['train']
            self.eval_type = 'valid' if not self.dataset_name == "ICEWS14" else 'test'

        self.print()

    def print(self):
        logger.info(" Dataset configurations")
        for key, val in self.__dict__.items():
            logger.info('{}: {}'.format(key, val))

    def read_file(self,
                 filename):
        logger.info(" ****** Reading data from {} ****** ".format(filename))
        with open(os.path.join(self.file_path + self.dataset_name, filename), "r") as f:
            data = f.readlines()

        facts = []
        for line in data:
            elements = line.strip().split("\t")
            head_id = self.getEntID(elements[0])
            rel_id = self.getRelID(elements[1])
            tail_id = self.getEntID(elements[2])
            fact = [head_id, rel_id, tail_id]
            facts.append(fact)

        return np.asarray(facts)

    def read_file_static(self,
                 filename):
        logger.info(" ****** Reading static data from {} ****** ".format(filename))
        with open(os.path.join(self.file_path + self.dataset_name, filename), "r") as f:
            data = f.readlines()

        facts = []
        for line in data:
            elements = line.strip().split(" ")
            head_id = int(elements[0])
            rel_id = int(elements[1])
            tail_id = int(elements[2])
            fact = [head_id, rel_id, tail_id]
            facts.append(fact)

        return np.asarray(facts)

    def getEntID(self,
                 ent_name):
        if ent_name in self.ent2id:
            return self.ent2id[ent_name]
        self.ent2id[ent_name] = len(self.ent2id)
        return self.ent2id[ent_name]

    def getRelID(self, rel_name):
        if rel_name in self.rel2id:
            return self.rel2id[rel_name]
        self.rel2id[rel_name] = len(self.rel2id)
        return self.rel2id[rel_name]

    def preprocess(self):
        logger.info(" ****** Pre-processing data ****** ")
        pre_quad = None
        if not os.path.exists(os.path.join(self.file_path, self.dataset_name, "train_static.txt")):
            self.transfer_dy2static()

        for model_type in self.train_types:
            quadlists = self.read_file_static(model_type + '_static.txt')
            if pre_quad is None:
                pre_quad = quadlists
            else:
                pre_quad = np.concatenate((pre_quad, quadlists), axis=0)


        self.data['train'] = pre_quad
        self.data['eval'] = self.read_file_static(self.eval_type + '_static.txt')
        q_tupes = set([tuple(q) for q in self.data['train']])
        self.all_facts_as_tuples.update(q_tupes)
        self.len['train'] = self.data['train'].shape[0]
        self.len['eval'] = self.data['eval'].shape[0]



    def transfer_dy2static(self):
        pre_quad = None
        for model_type in ['train', 'valid', 'test']:
            quadlists = self.read_file(model_type + '.txt')
            if pre_quad is None:
                pre_quad = quadlists
            else:
                pre_quad = np.concatenate((pre_quad, quadlists), axis=0)

        pre_quad = unique_rows(pre_quad)
        X_train, X_test = train_test_split(pre_quad, test_size=0.2, random_state=1996)
        x_valid, x_test = train_test_split(X_test, test_size=0.5, random_state=1996)
        np.savetxt(os.path.join(self.file_path, self.dataset_name, "train_static.txt"), X_train, fmt="%d")
        np.savetxt(os.path.join(self.file_path, self.dataset_name, "valid_static.txt"), x_valid, fmt="%d")
        np.savetxt(os.path.join(self.file_path, self.dataset_name, "test_static.txt"), x_test, fmt="%d")


    def load_processed(self):
        logger.info(" ****** Loding processed Static KG Dataset ******")
        logger.info("The files are loaded from {}{}".format(self.file_path, self.dataset_name))

        with open(os.path.join(self.file_path, self.dataset_name, self.prefix + "_data.pkl"), 'rb') as fp:
            self.data = pickle.load(fp)
        with open(os.path.join(self.file_path, self.dataset_name, self.prefix + "_len.pkl"), 'rb') as fp:
            self.len = pickle.load(fp)
        with open(os.path.join(self.file_path, self.dataset_name, self.prefix + "_data_tuples.pkl"), 'rb') as fp:
            self.all_facts_as_tuples = pickle.load(fp)

    def save_processed(self):
        logger.info(" ****** Saving processed Static KG Dataset ******")
        logger.info("The files are saved into {}{}".format(self.file_path, self.dataset_name))

        with open(os.path.join(self.file_path, self.dataset_name, self.prefix + "_data.pkl"), 'wb') as fp:
            pickle.dump(self.data, fp)
        with open(os.path.join(self.file_path, self.dataset_name, self.prefix + "_len.pkl"), 'wb') as fp:
            pickle.dump(self.len, fp)
        with open(os.path.join(self.file_path, self.dataset_name, self.prefix + "_data_tuples.pkl"), 'wb') as fp:
            pickle.dump(self.all_facts_as_tuples, fp)

    def addNegFacts(self, bp_facts):
        pos_neg_group_size = 1 + self.neg_ratio
        facts1 = np.repeat(np.copy(bp_facts), pos_neg_group_size, axis=0)
        facts2 = np.copy(facts1)
        rand_nums1 = np.random.randint(low=1, high=self.num_e, size=facts1.shape[0])
        rand_nums2 = np.random.randint(low=1, high=self.num_e, size=facts2.shape[0])

        for i in range(facts1.shape[0] // pos_neg_group_size):
            rand_nums1[i * pos_neg_group_size] = 0
            rand_nums2[i * pos_neg_group_size] = 0

        facts1[:, 0] = (facts1[:, 0] + rand_nums1) % self.num_e
        facts2[:, 2] = (facts2[:, 2] + rand_nums2) % self.num_e
        return np.concatenate((facts1, facts2), axis=0)

    def shredFacts(self, facts): #takes a batch of facts and shreds it into its columns
        heads = torch.tensor(facts[:,0]).long().to(self.device)
        rels = torch.tensor(facts[:,1]).long().to(self.device)
        tails = torch.tensor(facts[:,2]).long().to(self.device)
        return heads, rels, tails

    def nextBatch(self, batch_idx):
        quadruples = self.data['train'][batch_idx]
        batch = self.shredFacts(self.addNegFacts(quadruples))
        l = torch.zeros(int(len(batch[0])/(self.neg_ratio + 1))).long().to(self.device)
        return batch, l


    def nextEvalBatch(self, batch_id, mode='valid', type='obj'):
        fact = self.data['eval'][batch_id]
        sub, rel, obj = fact
        ret_facts = []
        # excluded_entity = []
        if type == "obj":
            pred_ents = torch.from_numpy(np.array([obj], dtype=int)).to(self.device)
            for i in range(self.num_e):
                ret_facts.append((sub, rel, i))
                # if (sub, rel, i) in self.all_facts_as_tuples and i != obj:
                #     excluded_entity.append(i)
        else:
            pred_ents = torch.from_numpy(np.array([sub], dtype=int)).to(self.device)
            for i in range(self.num_e):
                ret_facts.append((i, rel, obj))
                # if (i, rel, obj) in self.all_facts_as_tuples and i != sub:
                #     excluded_entity.append(i)
        returned_facts = self.shredFacts(np.array(ret_facts))
        # if len(excluded_entity) == 0:
        #     excluded_entity = None
        # else:
        #     excluded_entity = [excluded_entity]
        excluded_entity = None
        if self.measure_relation:
            return returned_facts, pred_ents, excluded_entity, [rel]
        else:
            return returned_facts, pred_ents, excluded_entity


class DyRDataset(Dataset):
    def __init__(self, params, device=None):
        logger.info(" ****** Creating Dynamic-Routing Dataset for: {} ******".format(params.dataset))
        self.parse_params(params)
        self.device = device

        self.data = dict()  # a dict, the key is the type of model, the value is a list of quadruples
        self.times = dict()  # a dict. the keys are the type of model, train or test, the value is a list, saving all the ordered timepoint
        self.len = dict()
        self.ent2id = {}
        self.rel2id = {}

        if self.rel_diff:
            self.entity_history = defaultdict(RelDict)
        else:
            self.entity_history = defaultdict(TimeDict)

        self.rel_sub_history = defaultdict(TimeDict)
        self.rel_obj_history = defaultdict(TimeDict)
        self.time_history = TimeDict()

        if not self.overwrite:
            try:
                self.load_processed()
            except:
                logger.warning("No preprocessed files to load! Doing the preprocess")

        if not self.entity_history:
            self.preprocess()
            self.save_processed()


    def parse_params(self, params):
        self.overwrite = params.overwrite
        self.file_path = params.data_dir
        self.dataset_name = params.dataset
        self.neg_ratio = 0
        self.num_e = params.num_e
        self.num_r = params.num_r
        self.time_scale = params.time_scale
        self.rel_diff = getattr(params, "rel_diff", False)
        self.measure_relation = getattr(params, 'measure_relation', False)
        self.candidate_nums = getattr(params, "candidate_nums", [20, 20, 20])
        self.default_max_time = getattr(params, "default_max_time", 20)
        self.max_time_range = getattr(params, "max_time_range", 10)
        self.start_time = getattr(params, "start_time", "2014-01-01")
        self.future = True if params.task == 'completion' else False
        self.mode = params.data_mode
        self.direct_id = getattr(params, 'direct_id', False)

        self.prefix = "DyR_" + self.mode
        if self.mode == "test":
            self.train_types = params.dataset_types[:-1]
            self.eval_type = "test"
        else:
            self.train_types = ['train']
            self.eval_type = 'valid' if not self.dataset_name == "ICEWS14" else 'test'

        self.print()

    def print(self):
        logger.info(" Dataset configurations")
        for key, val in self.__dict__.items():
            logger.info('{}: {}'.format(key, val))


    def preprocess(self):
        logger.info(" ****** Pre-processing data ****** ")
        pre_quad = None
        pre_times = None
        for model_type in self.train_types:
            quadlists, times = self.read_file(model_type + '.txt')
            if pre_quad is None:
                pre_quad = quadlists
                pre_times = times
            else:
                pre_quad = np.concatenate((pre_quad, quadlists), axis=0)
                pre_times = pre_times.union(times)

        for row in pre_quad:
            self.time_history.extend(row[3], [row[0], row[2]])
            if self.rel_diff:
                self.entity_history[row[0]].add(row[1], row[3], row[2])
                self.entity_history[row[2]].add(row[1], row[3], row[0])
            else:
                self.entity_history[row[0]].add(row[3], row[2])
                self.entity_history[row[2]].add(row[3], row[0])
            self.rel_sub_history[row[1]].add(row[3], row[0])
            self.rel_obj_history[row[1]].add(row[3], row[2])

        self.times['train'] = pre_times
        self.data['train'] = pre_quad
        self.len['train'] = pre_quad.shape[0]

        quadlists, times = self.read_file(self.eval_type + '.txt')
        self.times['eval'] = times
        self.data['eval'] = quadlists
        self.len['eval'] = quadlists.shape[0]
        if not self.future:
            for row in quadlists:
                self.time_history.extend(row[3], [row[0], row[2]])
                if self.rel_diff:
                    self.entity_history[row[0]].add(row[1], row[3], row[2])
                    self.entity_history[row[2]].add(row[1], row[3], row[0])
                else:
                    self.entity_history[row[0]].add(row[3], row[2])
                    self.entity_history[row[2]].add(row[3], row[0])
                self.rel_sub_history[row[1]].add(row[3], row[0])
                self.rel_obj_history[row[1]].add(row[3], row[2])

    def load_processed(self):
        logger.info(" ****** Loding processed Dynamic Routing Dataset ******")
        logger.info("The files are loaded from {}{}".format(self.file_path, self.dataset_name))

        with open(os.path.join(self.file_path, self.dataset_name, self.prefix + "_entity_history.pkl"), 'rb') as fp:
            self.entity_history = pickle.load(fp)
        with open(os.path.join(self.file_path, self.dataset_name, self.prefix + "_rel_sub_history.pkl"), 'rb') as fp:
            self.rel_sub_history = pickle.load(fp)
        with open(os.path.join(self.file_path, self.dataset_name, self.prefix + "_rel_obj_history.pkl"), 'rb') as fp:
            self.rel_obj_history = pickle.load(fp)
        with open(os.path.join(self.file_path, self.dataset_name, self.prefix + "_time_history.pkl"), 'rb') as fp:
            self.time_history = pickle.load(fp)
        with open(os.path.join(self.file_path, self.dataset_name, self.prefix + "_times.pkl"), 'rb') as fp:
            self.times = pickle.load(fp)
        with open(os.path.join(self.file_path, self.dataset_name, self.prefix + "_data.pkl"), 'rb') as fp:
            self.data = pickle.load(fp)
        with open(os.path.join(self.file_path, self.dataset_name, self.prefix + "_len.pkl"), 'rb') as fp:
            self.len = pickle.load(fp)
        # with open(os.path.join(self.file_path, self.dataset_name, self.prefix + "_ent2id.pkl"), 'rb') as fp:
        #     self.ent2id = pickle.load(fp)
        # with open(os.path.join(self.file_path, self.dataset_name, self.prefix + "_rel2id.pkl"), 'rb') as fp:
        #     self.rel2id = pickle.load(fp)

    def save_processed(self):
        logger.info(" ****** Saving processed Dynamic Routing Dataset ******")
        logger.info("The files are saved into {}{}".format(self.file_path, self.dataset_name))

        with open(os.path.join(self.file_path, self.dataset_name, self.prefix + "_entity_history.pkl"), 'wb') as fp:
            pickle.dump(self.entity_history, fp)
        with open(os.path.join(self.file_path, self.dataset_name, self.prefix + "_rel_sub_history.pkl"), 'wb') as fp:
            pickle.dump(self.rel_sub_history, fp)
        with open(os.path.join(self.file_path, self.dataset_name, self.prefix + "_rel_obj_history.pkl"), 'wb') as fp:
            pickle.dump(self.rel_obj_history, fp)
        with open(os.path.join(self.file_path, self.dataset_name, self.prefix + "_time_history.pkl"), 'wb') as fp:
            pickle.dump(self.time_history, fp)
        with open(os.path.join(self.file_path, self.dataset_name, self.prefix + "_times.pkl"), 'wb') as fp:
            pickle.dump(self.times, fp)
        with open(os.path.join(self.file_path, self.dataset_name, self.prefix + "_data.pkl"), 'wb') as fp:
            pickle.dump(self.data, fp)
        with open(os.path.join(self.file_path, self.dataset_name, self.prefix + "_len.pkl"), 'wb') as fp:
            pickle.dump(self.len, fp)
        with open(os.path.join(self.file_path, self.dataset_name, self.prefix + "_ent2id.pkl"), 'wb') as fp:
            pickle.dump(self.ent2id, fp)
        with open(os.path.join(self.file_path, self.dataset_name, self.prefix + "_rel2id.pkl"), 'wb') as fp:
            pickle.dump(self.rel2id, fp)


    def transfer_task(self, mode='f2c'):
        if self.eval_type != 'test':
            raise ValueError('Not the complete dataset with the eval type {}'.format(self.eval_type))

        if mode == 'f2c':
            data_all = np.concatenate((self.data['train'], self.data['eval']), axis=0)
            X_train, X_test = train_test_split(data_all, test_size=0.2, random_state=1996)
            x_valid, x_test = train_test_split(X_test, test_size=0.5, random_state=1996)
            np.savetxt(os.path.join(self.file_path, self.dataset_name, "train_completion.txt"), X_train, fmt="%d")
            np.savetxt(os.path.join(self.file_path, self.dataset_name, "valid_completion.txt"), x_valid, fmt="%d")
            np.savetxt(os.path.join(self.file_path, self.dataset_name, "test_completion.txt"), x_test, fmt="%d")
        elif mode == 'c2f':
            data_all = np.concatenate((self.data['train'], self.data['eval']), axis=0)
            data_sorted = data_all[data_all[:, 3].argsort()]
            ttl_len = data_sorted.shape[0]
            train_len = int(0.8 * ttl_len)
            test_len = int(0.1 * ttl_len)
            train_last_t = data_sorted[train_len, 3]
            valid_last_t = data_sorted[ttl_len - test_len - 1, 3]
            train_data = data_sorted[data_sorted[:, 3]<= train_last_t]
            valid_data = data_sorted[(data_sorted[:, 3]> train_last_t) & (data_sorted[:, 3]<= valid_last_t)]
            test_data = data_sorted[data_sorted[:, 3] > valid_last_t]
            np.savetxt(os.path.join(self.file_path, self.dataset_name, "train_forecast.txt"), train_data, fmt="%d")
            np.savetxt(os.path.join(self.file_path, self.dataset_name, "valid_forecast.txt"), valid_data, fmt="%d")
            np.savetxt(os.path.join(self.file_path, self.dataset_name, "test_forecast.txt"), test_data, fmt="%d")

        else:
            raise NotImplementedError(mode)


    def nextBatch(self, batch_idx, type="obj"):

        quadruples = self.data['train'][batch_idx]
        if len(quadruples.shape) == 1:
            quadruples = np.reshape(quadruples, (1, quadruples.size))

        ent_history = np.zeros((len(batch_idx), self.candidate_nums[0], 2), dtype=int)
        rel_history = np.zeros((len(batch_idx), self.candidate_nums[1], 2), dtype=int)
        time_history = np.zeros((len(batch_idx), self.candidate_nums[2], 2), dtype=int)
        pred_ents = np.zeros(len(batch_idx), dtype=int)

        center_loc = 0 if type == "obj" else 2
        empty_ids = []

        for i, q in enumerate(quadruples):
            if self.candidate_nums[0] > 0:
                if self.rel_diff:
                    ent_h = self.entity_history[q[center_loc]].extract_neighbors(q[1], q[3], q[2-center_loc],
                                                                             self.candidate_nums[0],
                                                                             self.max_time_range,
                                                                             self.future)
                else:
                    ent_h = self.entity_history[q[center_loc]].extract_neighbors(q[3], q[2-center_loc],
                                                                             self.candidate_nums[0],
                                                                             self.max_time_range,
                                                                             self.future)
            else:
                ent_h = np.zeros((0,2))
            if ent_h is None:
                empty_ids.append(i)
                continue

            if self.candidate_nums[1] > 0:
                if type == "obj":
                    rel_h = self.rel_obj_history[q[1]].extract_neighbors(q[3], -1, self.candidate_nums[1],
                                                                     2, self.future)

                else:
                    rel_h = self.rel_sub_history[q[1]].extract_neighbors(q[3], -1,  self.candidate_nums[1],
                                                                      2, self.future)
            else:
                rel_h = np.zeros((0, 2))
            if rel_h is None:
                empty_ids.append(i)
                continue

            if self.candidate_nums[2] > 0:
                tim_h = self.time_history.extract_neighbors(q[3], -1, self.candidate_nums[2],
                                                        1, self.future)
            else:
                tim_h = np.zeros((0, 2))

            if tim_h is None:
                empty_ids.append(i)
                continue

            ent_history[i] = ent_h
            rel_history[i] = rel_h
            time_history[i] = tim_h
            pred_ents[i] = q[2-center_loc]

        if len(empty_ids) > 0:
            deleted_ids = np.asarray(empty_ids).reshape(-1)
            ent_history = np.delete(ent_history, deleted_ids, axis=0)
            rel_history = np.delete(rel_history, deleted_ids, axis=0)
            time_history = np.delete(time_history, deleted_ids, axis=0)
            pred_ents = np.delete(pred_ents, deleted_ids, axis=0)

        all_history = np.concatenate((ent_history, rel_history, time_history), axis=1)
        all_history = torch.from_numpy(all_history).to(self.device)
        pred_ents = torch.from_numpy(pred_ents).to(self.device)
        return all_history, pred_ents

    def nextEvalBatch(self, batch_idx, mode="no_used", type="obj"):
        quadruples = self.data['eval'][batch_idx]

        if len(quadruples.shape) == 1:
            quadruples = np.reshape(quadruples, (1, quadruples.size))

        ent_history = np.zeros((len(batch_idx), self.candidate_nums[0], 2), dtype=int)
        rel_history = np.zeros((len(batch_idx), self.candidate_nums[1], 2), dtype=int)
        time_history = np.zeros((len(batch_idx), self.candidate_nums[2], 2), dtype=int)
        pred_ents = np.zeros(len(batch_idx), dtype=int)
        all_excluded = []

        center_loc = 0 if type == "obj" else 2
        empty_ids = []

        for i, q in enumerate(quadruples):
            all_excluded.append([])
            if self.rel_diff:
                try:
                    ent_h = self.entity_history[q[center_loc]].extract_neighbors(q[1], q[3], q[2-center_loc],
                                                                             self.candidate_nums[0],
                                                                             self.max_time_range,
                                                                             self.future)
                    to_be_excluded = self.entity_history[q[center_loc]].extract_concurrent(q[1], q[3],
                                                                                           q[2 - center_loc])

                except:
                    ent_h = None

            else:
                try:
                    ent_h = self.entity_history[q[center_loc]].extract_neighbors(q[3], q[2-center_loc], self.candidate_nums[0],
                                                                             self.max_time_range,
                                                                             self.future)
                    to_be_excluded = self.entity_history[q[center_loc]].extract_concurrent(q[3], q[2-center_loc])
                except:
                    ent_h = None
            all_excluded[-1].extend(to_be_excluded)
            if ent_h is None:
                empty_ids.append(i)
                continue

            if type == "obj":
                try:
                    rel_h = self.rel_obj_history[q[1]].extract_neighbors(q[3], -1, self.candidate_nums[1],
                                                                     2, self.future)
                except:
                    rel_h = None
            else:
                try:
                    rel_h = self.rel_sub_history[q[1]].extract_neighbors(q[3], -1, self.candidate_nums[1],
                                                                     2, self.future)
                except:
                    rel_h = None

            if rel_h is None:
                empty_ids.append(i)
                continue

            tim_h = self.time_history.extract_neighbors(q[3], -1, self.candidate_nums[2],
                                                        1, self.future)
            if tim_h is None:
                empty_ids.append(i)
                continue

            ent_history[i] = ent_h
            rel_history[i] = rel_h
            time_history[i] = tim_h
            pred_ents[i] = abs(q[2 - center_loc])

        if self.measure_relation:
            rels = quadruples[:, 1]

        if len(empty_ids) > 0:
            deleted_ids = np.asarray(empty_ids).reshape(-1)
            ent_history = np.delete(ent_history, deleted_ids, axis=0)
            rel_history = np.delete(rel_history, deleted_ids, axis=0)
            time_history = np.delete(time_history, deleted_ids, axis=0)
            pred_ents = np.delete(pred_ents, deleted_ids, axis=0)
            for reverse_id in empty_ids[::-1]:
                del all_excluded[reverse_id]
            if self.measure_relation:
                rels = np.delete(rels, empty_ids)

        all_history = np.concatenate((ent_history, rel_history, time_history), axis=1)
        all_history = torch.from_numpy(all_history).to(self.device)
        pred_ents = torch.from_numpy(pred_ents).to(self.device)
        if self.measure_relation:
            return all_history, pred_ents, all_excluded, rels
        else:
            return all_history, pred_ents, all_excluded

    def read_file(self,
                 filename):
        start_time = datetime.strptime(self.start_time, "%Y-%m-%d")
        logger.info(" ****** Reading data from {} ****** ".format(filename))
        with open(os.path.join(self.file_path + self.dataset_name, filename), "r", encoding="utf-8") as f:
            data = f.readlines()

        facts = []
        times = set()
        for line in data:
            if self.direct_id:
                elements = line.strip().split(' ')
                head_id = int(elements[0])
                rel_id = int(elements[1])
                tail_id = int(elements[2])
                timestamp = int(elements[3])
            else:
                elements = line.strip().split("\t")
                head_id = self.getEntID(elements[0])
                rel_id = self.getRelID(elements[1])
                tail_id = self.getEntID(elements[2])
                if self.time_scale > 0:
                    timestamp = int(int(elements[3]) / self.time_scale)
                else:
                    timestamp = (datetime.strptime(elements[3], "%Y-%m-%d") - start_time).days
            times.add(timestamp)
            facts.append([head_id, rel_id, tail_id, timestamp])
        # times = list(times)
        # times.sort()
        # for f in facts:
        #     f[3] = times.index(f[3]) + start_time
        return np.asarray(facts), times

    def getEntID(self, ent_name):
        if ent_name in self.ent2id:
            return self.ent2id[ent_name]
        self.ent2id[ent_name] = len(self.ent2id)
        return self.ent2id[ent_name]

    def getRelID(self, rel_name):
        if rel_name in self.rel2id:
            return self.rel2id[rel_name]
        self.rel2id[rel_name] = len(self.rel2id)
        return self.rel2id[rel_name]

class TimeDict(object):
    def __init__(self):
        self.time = []
        self.history = dict()

    def add(self, time, object):
        if time in self.time:
            self.history[time].append(object)
        else:
            self.time.append(time)
            self.history[time] = [object]
            self.time.sort()

    def extend(self, time, objects):
        if time in self.time:
            self.history[time].extend(objects)
        else:
            self.time.append(time)
            self.history[time] = objects
            self.time.sort()

    def get(self, time):
        return self.history.get(time, [])

    def extract_neighbors(self, time_point, to_exclude, num,  max_time_range=10, future=False):
        if len(self.time) == 0:
            return None
        times = self.time.copy()
        insert_sign = 0
        if time_point not in times:
            times.append(time_point)
            times.sort()
            insert_sign = 1
        idx = times.index(time_point)

        upper_bound = min(idx - insert_sign + max_time_range + 1, len(self.time)) if future else idx
        all_candidate_idx = [i for i in range(max(idx - max_time_range, 0), upper_bound)]
        if len(all_candidate_idx) < 1:
            return None
        all_candidates_ents = []
        all_candidates_times = []
        for i in all_candidate_idx:
            cur_ents = self.history[self.time[i]]
            cur_times = [self.time[i]] * len(cur_ents)
            all_candidates_ents.extend(cur_ents)
            all_candidates_times.extend(cur_times)
        all_candidates = np.array((all_candidates_ents, all_candidates_times))
        if future:
            all_candidates[1] = abs(all_candidates[1] - time_point) + 1
        else:
            all_candidates[1] = time_point - all_candidates[1]
        remove_idx = np.where(np.logical_and(all_candidates[1] == 1, all_candidates[0] == to_exclude))[0]
        all_candidates = np.delete(all_candidates, remove_idx, axis=1).transpose()
        if all_candidates.shape[0] < 1:
            return None
        all_selected_idx = np.random.choice(all_candidates.shape[0], num, replace=True)
        all_selected = all_candidates[all_selected_idx]

        return all_selected

    def extract_concurrent(self, time, center_ent):
        all_ents = set(self.get(time))
        if center_ent in all_ents:
            all_ents.remove(center_ent)
        return list(all_ents)

    def summary(self):
        time_len = []
        for t in self.time:
            time_len.append(len(self.history[t]))
        return time_len


class RelDict(object):
    def __init__(self):
        self.rels = []
        self.time_dicts = defaultdict(TimeDict)

    def add(self, rel, time, object):
        if rel not in self.rels:
            self.rels.append(rel)
        self.time_dicts[rel].add(time, object)

    def extend(self, rel, time, objects):
        if rel not in self.rels:
            self.rels.append(rel)
        self.time_dicts[rel].extend(time, objects)

    def get(self, rel, time):
        return self.time_dicts[rel].get(time)

    def summary(self):
        all_rel_len = dict()
        for r in self.rels:
            all_rel_len[r] = self.time_dicts[r].summary()
        return all_rel_len

    def extract_neighbors(self, rel, time, to_exclude, num, max_time_range=10, future=False):
        if rel not in self.rels:
            return None
        return self.time_dicts[rel].extract_neighbors(time, to_exclude, num, max_time_range, future)

    def extract_concurrent(self, rel, time, center_ent):
        return self.time_dicts[rel].extract_concurrent(time, center_ent)
