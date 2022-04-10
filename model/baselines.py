import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.conv import RelGraphConv
from datetime import datetime


logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

logger = logging.getLogger(__name__)


class TransE(nn.Module):
    def __init__(self, params, device):
        super(TransE, self).__init__()
        self.parse_args(params)
        self.device = device
        self.build_model()
        self.init_embeddings()

    def parse_args(self, params):
        self.num_e = params.num_e
        self.num_r = params.num_r
        self.embed_dim = params.embed_dim
        self.dropout = getattr(params, "dropout", 0.0)
        self.neg_ratio = params.neg_ratio
        self.name = 'TransE'
        self.birth_time = datetime.now().strftime("%d%m-%H%M")

    def build_model(self):
        self.ent_embs = nn.Embedding(self.num_e, self.embed_dim)
        self.rel_embs = nn.Embedding(self.num_r, self.embed_dim)
        self.loss_f = nn.CrossEntropyLoss()

    def init_embeddings(self):
        nn.init.xavier_uniform_(self.ent_embs.weight)
        nn.init.xavier_uniform_(self.rel_embs.weight)

    def forward(self, train_data):
        heads, rels, tails = train_data
        h_embs, r_embs, t_embs = self.ent_embs(heads), self.rel_embs(rels), self.ent_embs(tails)
        scores = h_embs + r_embs - t_embs
        scores = F.dropout(scores, p=self.dropout, training=self.training)
        scores = -torch.norm(scores, dim=1)
        return scores

    def loss(self, train_data, pred_ents):
        scores = self.forward(train_data)
        scores_reshaped = scores.view(-1, self.neg_ratio + 1)
        loss = self.loss_f(scores_reshaped, pred_ents)
        return loss

    def save(self, dir, suffix):
        name = "{}_{}_{}.pth".format(self.name, self.birth_time, suffix)
        logger.info(" Saving {} model as {}".format(self.name, name))
        torch.save(self.state_dict(), os.path.join(dir, name))
        return name

    def load(self, dir, name):
        logger.info("Loading pretrained {} model from {}{}".format(self.name, dir, name))
        if self.device == torch.device("cpu"):
            checkpoint = torch.load(os.path.join(dir, name), map_location=self.device)
        else:
            checkpoint = torch.load(os.path.join(dir, name))

        self.load_state_dict(checkpoint, strict=False)


class DistMult(torch.nn.Module):
    def __init__(self, params, device):
        super(DistMult, self).__init__()
        self.parse_args(params)
        self.device = device
        self.build_model()
        self.init_embeddings()

    def parse_args(self, params):
        self.num_e = params.num_e
        self.num_r = params.num_r
        self.embed_dim = params.embed_dim
        self.dropout = getattr(params, "dropout", 0.0)
        self.neg_ratio = params.neg_ratio
        self.name = 'DistMult'
        self.birth_time = datetime.now().strftime("%d%m-%H%M")


    def build_model(self):
        self.ent_embs = nn.Embedding(self.num_e, self.embed_dim)
        self.rel_embs = nn.Embedding(self.num_r, self.embed_dim)

        self.loss_f = nn.CrossEntropyLoss()

    def init_embeddings(self):
        nn.init.xavier_uniform_(self.ent_embs.weight)
        nn.init.xavier_uniform_(self.rel_embs.weight)

    def forward(self, train_data):
        heads, rels, tails = train_data
        h_embs, r_embs, t_embs = self.ent_embs(heads), self.rel_embs(rels), self.ent_embs(tails)
        scores = (h_embs * r_embs) * t_embs
        scores = F.dropout(scores, p=self.dropout, training=self.training)
        scores = torch.sum(scores, dim=1)
        return scores

    def loss(self, train_data, pred_ents):
        scores = self.forward(train_data)
        scores_reshaped = scores.view(-1, self.neg_ratio + 1)
        loss = self.loss_f(scores_reshaped, pred_ents)
        return loss

    def save(self, dir, suffix):
        name = "{}_{}_{}.pth".format(self.name, self.birth_time, suffix)
        logger.info(" Saving {} model as {}".format(self.name, name))
        torch.save(self.state_dict(), os.path.join(dir, name))
        return name

    def load(self, dir, name):
        logger.info("Loading pretrained {} model from {}{}".format(self.name, dir, name))
        if self.device == torch.device("cpu"):
            checkpoint = torch.load(os.path.join(dir, name), map_location=self.device)
        else:
            checkpoint = torch.load(os.path.join(dir, name))

        self.load_state_dict(checkpoint, strict=False)


class SimplE(torch.nn.Module):
    def __init__(self, params, device):
        super(SimplE, self).__init__()
        self.parse_args(params)
        self.device = device
        self.build_model()
        self.init_embeddings()

    def parse_args(self, params):
        self.num_e = params.num_e
        self.num_r = params.num_r
        self.embed_dim = params.embed_dim
        self.dropout = getattr(params, "dropout", 0.0)
        self.neg_ratio = params.neg_ratio
        self.name = 'SimplE'
        self.birth_time = datetime.now().strftime("%d%m-%H%M")

    def build_model(self):
        self.ent_embs_h = nn.Embedding(self.num_e, self.embed_dim)
        self.ent_embs_t = nn.Embedding(self.num_e, self.embed_dim)
        self.rel_embs_f = nn.Embedding(self.num_r, self.embed_dim)
        self.rel_embs_i = nn.Embedding(self.num_r, self.embed_dim)
        self.loss_f = nn.CrossEntropyLoss()


    def init_embeddings(self):
        nn.init.xavier_uniform_(self.ent_embs_h.weight)
        nn.init.xavier_uniform_(self.ent_embs_t.weight)
        nn.init.xavier_uniform_(self.rel_embs_f.weight)
        nn.init.xavier_uniform_(self.rel_embs_i.weight)

    def getEmbeddings(self, heads, rels, tails):
        h_embs1 = self.ent_embs_h(heads)
        r_embs1 = self.rel_embs_f(rels)
        t_embs1 = self.ent_embs_t(tails)
        h_embs2 = self.ent_embs_h(tails)
        r_embs2 = self.rel_embs_i(rels)
        t_embs2 = self.ent_embs_t(heads)

        return h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2

    def forward(self, train_data):
        heads, rels, tails = train_data
        h_embs1, r_embs1, t_embs1, h_embs2, r_embs2, t_embs2 = self.getEmbeddings(heads, rels, tails)
        scores = ((h_embs1 * r_embs1) * t_embs1 + (h_embs2 * r_embs2) * t_embs2) / 2.0
        scores = F.dropout(scores, p=self.dropout, training=self.training)
        scores = torch.sum(scores, dim=1)
        return scores

    def generate(self, idx):
        h_embs = self.ent_embs_h(idx)
        t_embs = self.ent_embs_t(idx)
        all_embs = torch.cat((h_embs, t_embs), axis=-1)
        return all_embs


    def loss(self, train_data, pred_ents):
        scores = self.forward(train_data)
        scores_reshaped = scores.view(-1, self.neg_ratio + 1)
        loss = self.loss_f(scores_reshaped, pred_ents)
        return loss

    def save(self, dir, suffix):
        name = "{}_{}_{}.pth".format(self.name, self.birth_time, suffix)
        logger.info(" Saving {} model as {}".format(self.name, name))
        torch.save(self.state_dict(), os.path.join(dir, name))
        return name

    def load(self, dir, name):
        logger.info("Loading pretrained {} model from {}{}".format(self.name, dir, name))
        if self.device == torch.device("cpu"):
            checkpoint = torch.load(os.path.join(dir, name), map_location=self.device)
        else:
            checkpoint = torch.load(os.path.join(dir, name))

        self.load_state_dict(checkpoint, strict=False)


class BaseRGCN(nn.Module):
    def __init__(self, params, device):
        super(BaseRGCN, self).__init__()
        self.parse_args(params)
        self.device = device
        self.build_model()
        # self.init_embeddings()


    def parse_args(self, params):
        self.num_nodes = params.num_e
        self.h_dim = params.hid_dim
        self.out_dim = getattr(params, 'out_dim', params.hid_dim)
        self.num_rels = 2 * params.num_r
        self.num_bases = getattr(params, 'num_bases', None)
        self.num_hidden_layers = getattr(params, 'num_hidden_layers', 1)
        self.dropout = params.dropout
        self.use_self_loop = getattr(params, 'use_self_loop', False)

        self.name = 'RGCN'
        self.birth_time = datetime.now().strftime("%d%m-%H%M")


    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def build_input_layer(self):
        return EmbeddingLayer(self.num_nodes, self.h_dim)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "bdd",
                self.num_bases, activation=act, self_loop=True,
                dropout=self.dropout)

    def build_output_layer(self):
        return None

    def forward(self, g, h, r, norm):
        for layer in self.layers:
            h = layer(g, h, r, norm)
        return h


class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, h_dim)

    def forward(self, g, h, r, norm):
        return self.embedding(h.squeeze())


class RGCNLinkPredict(nn.Module):
    def __init__(self, params, device):
        super(RGCNLinkPredict, self).__init__()
        self.parse_args(params)
        self.device = device
        self.build_model()
        self.init_embeddings()

    def parse_args(self, params):
        self.in_dim = params.num_e
        self.h_dim = params.hid_dim
        self.num_r = params.num_r
        self.num_bases = getattr(params, "num_bases", -1)
        self.num_hidden_layers = getattr(params, 'num_hidden_layers', 1)
        self.dropout = params.dropout
        self.reg_param = getattr(params, 'reg_param', 0)
        self.params = params


        self.name = 'RGCN-linkpredict'
        self.birth_time = datetime.now().strftime("%d%m-%H%M")

    def build_model(self):
        self.rgcn = BaseRGCN(self.params, self.device)
        self.w_relation = nn.Parameter(torch.Tensor(self.num_r, self.h_dim))

    def init_embeddings(self):
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g, h, r, norm):
        return self.rgcn.forward(g, h, r, norm)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def loss(self, g, embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss