import torch
import torch.nn as nn
from datetime import datetime
import torch.nn.functional as F
import logging
import os
from model.baselines import SimplE, DistMult, TransE

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

logger = logging.getLogger(__name__)

def squash(x, axis=-2, epsilon=1e-9, squash_weight=1):
    '''
    x: [batch size, num filters, vec len]. e.g. [256, 50, 100]
    s_squared_norm: [batch size, 1, vec len] e.g. [256, 1, 100]
    '''
    s_squared_norm = (x ** 2).sum(axis, keepdim=True) #vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keepdims=True)
    scale = s_squared_norm / (squash_weight + s_squared_norm) / torch.sqrt(s_squared_norm + epsilon) # scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    return scale * x

def dynamic_routing(b_ij, u_hat, num_iterations=3, squash_weight=1, axis=-2):
    u_hat_detached = u_hat.detach()
    for r_iter in range(num_iterations):
        c_ij = F.softmax(b_ij, dim=1) * b_ij.shape[1]  # c_ij[0,:,0,0,0]
        s_ij = (c_ij * u_hat).sum(dim=1, keepdim=True)
        v_j = squash(s_ij, axis=axis, squash_weight=squash_weight)
        if r_iter < num_iterations - 1:
            v_j1 = v_j.expand(-1, b_ij.shape[1], -1, -1, -1)
            u_vj1 = torch.matmul(u_hat_detached.transpose(3, 4), v_j1)
            b_ij = b_ij + u_vj1
    return v_j, c_ij

class MeanAggregator(nn.Module):
    def __init__(self, args):
        super(MeanAggregator, self).__init__()
        self.parse_args(args)
        self.build_model()

    def parse_args(self, args):
        self.in_len = getattr(args, "embed_dim", 100)
        self.out_len = getattr(args, "hid_dim", 200)
        self.candidate_num = getattr(args, "candidate_num", 20)
        self.iter_num = getattr(args, "iter_num", 3)


    def build_model(self, activation=None):
        self.Ws = nn.Linear(self.in_len, self.out_len, bias=True)
        nn.init.xavier_normal_(self.Ws.weight.data)
        self.activation = activation
        self.dropout = nn.Dropout(0.1)


    def forward(self, embeds, weights):
        x_trans = self.Ws(embeds)
        x_trans = self.dropout(x_trans)
        if self.activation:
            x_trans = self.activation(x_trans)
        u_hat = F.normalize(x_trans.mean(dim=1), p=2, dim=-1)
        b_ij = torch.ones(weights.shape, device=embeds.device).float()
        return u_hat, b_ij

    def generate(self, embeds):
        x_trans = self.Ws(embeds)
        x_trans = self.dropout(x_trans)
        if self.activation:
            x_trans = self.activation(x_trans)
        return x_trans


class DyRAggregator(nn.Module):
    def __init__(self, args):
        super(DyRAggregator, self).__init__()
        self.parse_args(args)
        self.build_model()

    def parse_args(self, args):
        self.in_len = getattr(args, "embed_dim", 100)
        self.out_len = getattr(args, "hid_dim", 200)
        self.candidate_num = getattr(args, "candidate_num", 20)
        self.iter_num = getattr(args, "iter_num", 3)


    def build_model(self, activation=None):
        self.rout = dynamic_routing
        self.Ws = nn.Linear(self.in_len, self.out_len, bias=True)
        nn.init.xavier_normal_(self.Ws.weight.data)
        self.activation = activation
        self.dropout = nn.Dropout(0.1)


    def forward(self, embeds, weights):
        x_trans = self.Ws(embeds)
        x_trans = self.dropout(x_trans)
        if self.activation:
            x_trans = self.activation(x_trans)
        u_hat = F.normalize(x_trans.unsqueeze(2).unsqueeze(-1), p=2, dim=-2)
        b_ij = weights.unsqueeze(2).unsqueeze(-1).unsqueeze(-1).float()
        poses, c_ij = self.rout(b_ij, u_hat, num_iterations=self.iter_num)
        poses = poses.squeeze(1).squeeze(-1).squeeze(1)
        c_ij = c_ij.squeeze(-1).squeeze(-1).squeeze(-1)
        return poses, c_ij

    def generate(self, embeds):
        x_trans = self.Ws(embeds)
        x_trans = self.dropout(x_trans)
        if self.activation:
            x_trans = self.activation(x_trans)
        return x_trans

class SimpleDyRAggregator(nn.Module):
    def __init__(self, args):
        super(SimpleDyRAggregator, self).__init__()
        self.parse_args(args)
        self.build_model()

    def parse_args(self, args):
        self.iter_num = getattr(args, "iter_num", 3)
        self.dropout_ratio = getattr(args, "dropout", 0.1)


    def build_model(self, activation=None):
        self.rout = dynamic_routing
        self.activation = activation
        self.dropout = nn.Dropout(self.dropout_ratio)


    def forward(self, embeds, weights):
        x_trans = self.dropout(embeds)
        if self.activation:
            x_trans = self.activation(x_trans)
        u_hat = F.normalize(x_trans.unsqueeze(2).unsqueeze(-1), p=2, dim=-2)
        b_ij = weights.unsqueeze(2).unsqueeze(-1).unsqueeze(-1).float()
        poses, c_ij = self.rout(b_ij, u_hat, num_iterations=self.iter_num)
        poses = poses.squeeze(1).squeeze(-1).squeeze(1)
        c_ij = c_ij.squeeze(-1).squeeze(-1).squeeze(-1)
        return poses, c_ij

    def generate(self, embeds):
        x_trans = self.dropout(embeds)
        if self.activation:
            x_trans = self.activation(x_trans)
        return x_trans

class DyRMLP(nn.Module):
    def __init__(self, args, device):
        super(DyRMLP, self).__init__()
        self.parse_args(args)
        self.device = device
        self.build_model(args)
        self.init_embeddings()

    def parse_args(self, args):
        self.cache_dir = args.cache_dir
        self.num_e = args.num_e
        self.epsilon = float(args.epsilon)
        self.embed_dim = getattr(args, "embed_dim", 100)
        self.neigh_num = getattr(args, "candidate_num", 20)
        self.checkpoint = getattr(args, 'checkpoint', None)
        self.hid_dim = getattr(args, "hid_dim", 20)
        self.weight_base = getattr(args, "weight_base", 1)
        self.iter_num = getattr(args, "iter_num", 3)
        self.alpha = getattr(args, "alpha", 0.5)
        self.dropout_ratio = getattr(args, "dropout", 0.0)
        self.from_pretrained = getattr(args, 'from_pretrained', False)
        self.fix_pretrained = getattr(args, 'fix_pretrained', False)
        self.loss_type = getattr(args, "loss", "cross")
        self.reflect = getattr(args, "reflect", False)
        self.name = "DyRMLP" + getattr(args, "suffix", "")
        self.birth_time = datetime.now().strftime("%m%d-%H%M%S%f")

    def build_model(self, args):
        self.embeddings = nn.Parameter(torch.Tensor(self.num_e, self.embed_dim))
        if self.iter_num > 0:
            if self.hid_dim > 0:
                self.aggregator = DyRAggregator(args)
            else:
                self.aggregator = SimpleDyRAggregator(args)
        else:
            self.aggregator = MeanAggregator(args)
        self.dropout = nn.Dropout(self.dropout_ratio)
        if self.reflect:
            self.decoder_bias = nn.Parameter(torch.zeros(1, self.num_e))
        else:
            self.mlp = nn.Linear(self.hid_dim, self.num_e)
        self.loss_f = nn.NLLLoss()


    def init_embeddings(self):
        if self.from_pretrained:
            logger.info("Loading the pretrained embeddings ...")
            if self.device == torch.device("cpu"):
                checkpoint = torch.load(os.path.join(self.cache_dir, self.checkpoint), map_location=self.device)
            else:
                checkpoint = torch.load(os.path.join(self.cache_dir, self.checkpoint))
            self.embeddings.data = checkpoint['ent_embs.weight']
            self.embeddings.requires_grad = not self.fix_pretrained

        else:
            nn.init.normal_(self.embeddings)
        self.embeddings.data = F.normalize(self.embeddings, p=2, dim=1)
        if self.reflect:
            nn.init.xavier_normal_(self.decoder_bias.data)
        else:
            nn.init.xavier_normal_(self.mlp.weight.data)

    def forward(self, his):
        idx = his[:, :, 0]
        embeds = self.embeddings[idx]
        weights = self.weighting(his[:, :, 1])

        out_embeds, c_ij = self.aggregator(embeds, weights)
        one_hot = F.one_hot(idx, self.num_e)
        p_ij = torch.matmul(one_hot.transpose(1, 2).float(), c_ij.unsqueeze(-1)).squeeze(-1)
        p_ij = F.softmax(p_ij, dim=1)

        if self.reflect:
            all_embeds = self.aggregator.generate(self.embeddings.data)
            scores = torch.matmul(self.dropout(out_embeds.unsqueeze(1)), all_embeds.t()) + self.decoder_bias
            scores = scores.squeeze(1)
        else:
            scores = self.mlp(self.dropout(out_embeds))

        sims = F.softmax(scores, dim=1)
        final_score = torch.log(self.alpha * p_ij + (1 - self.alpha) * sims)
        return final_score

    def loss(self, xs, y_true):
        scores = self.forward(xs)
        loss = self.loss_f(scores, y_true)
        return loss

    def weighting(self, times):
        t_weighted = (self.weight_base + 1.0) / (self.weight_base + times.float())
        return t_weighted

    def save(self, dir, suffix):
        name = "{}_{}_{}.pth".format(self.name, self.birth_time, suffix)
        logger.info(" Saving the trained model as {}".format(name))
        torch.save(self.state_dict(), os.path.join(dir, name))
        return name

    def load(self, dir, name):
        logger.info("Loading the pretrained model from {}{}".format(dir, name))
        if self.device == torch.device("cpu"):
            checkpoint = torch.load(os.path.join(dir, name), map_location=self.device)
        else:
            checkpoint = torch.load(os.path.join(dir, name))

        self.load_state_dict(checkpoint, strict=False)


class BaseDyR(nn.Module):
    def __init__(self, args, device):
        super(BaseDyR, self).__init__()
        self.parse_args(args)
        self.device = device
        self.build_model(args)
        self.init_embeddings()

    def parse_args(self, args):
        self.cache_dir = args.cache_dir
        self.base_name = args.base_model
        self.num_e = args.num_e
        self.epsilon = float(args.epsilon)
        self.embed_dim = getattr(args, "embed_dim", 100)
        self.neigh_num = getattr(args, "candidate_num", 20)
        self.checkpoint = getattr(args, 'checkpoint', None)
        self.hid_dim = getattr(args, "hid_dim", 20)
        self.weight_base = getattr(args, "weight_base", 1)
        self.iter_num = getattr(args, "iter_num", 3)
        self.alpha = getattr(args, "alpha", 0.5)
        self.dropout_ratio = getattr(args, "dropout", 0.0)
        self.from_pretrained = getattr(args, 'from_pretrained', False)
        self.fix_pretrained = getattr(args, 'fix_pretrained', False)
        self.loss_type = getattr(args, "loss", "cross")
        self.reflect = getattr(args, "reflect", False)
        self.name = "BaseDyR" + getattr(args, "suffix", "")
        self.birth_time = datetime.now().strftime("%d%m-%H%M")

    def build_model(self, args):
        if self.base_name == 'simple':
            self.base_model = SimplE(args, self.device)
            args.embed_dim = 2 * args.embed_dim
        elif self.base_name == 'distmult':
            self.base_model = DistMult(args, self.device)
        elif self.base_name == 'transe':
            self.base_model = TransE(args, self.device)

        if self.device == torch.device("cpu"):
            checkpoint = torch.load(os.path.join(self.cache_dir, self.checkpoint), map_location=self.device)
        else:
            checkpoint = torch.load(os.path.join(self.cache_dir, self.checkpoint))
        self.base_model.load_state_dict(checkpoint)
        if self.iter_num > 0:
            if self.hid_dim > 0:
                self.aggregator = DyRAggregator(args)
            else:
                self.aggregator = SimpleDyRAggregator(args)
        else:
            self.aggregator = MeanAggregator(args)
        self.dropout = nn.Dropout(self.dropout_ratio)
        if not self.reflect:
            self.mlp = nn.Linear(self.hid_dim, self.num_e)
        self.loss_f = nn.NLLLoss()


    def init_embeddings(self):
        if not self.reflect:
            nn.init.xavier_normal_(self.mlp.weight.data)

        self.base_model.requires_grad = not self.fix_pretrained

    def forward(self, his):
        idx = his[:, :, 0]
        embeds = self.base_model.generate(idx)
        embeds = F.normalize(embeds, p=2, dim=1)
        weights = self.weighting(his[:, :, 1])
        out_embeds, c_ij = self.aggregator(embeds, weights)
        one_hot = F.one_hot(idx, self.num_e)
        p_ij = torch.matmul(one_hot.transpose(1, 2).float(), c_ij.unsqueeze(-1)).squeeze(-1)
        p_ij = F.softmax(p_ij, dim=1)

        if self.reflect:
            all_embeds = self.aggregator.generate(self.embeddings.data)
            scores = torch.matmul(self.dropout(out_embeds), all_embeds.t())
        else:
            scores = self.mlp(self.dropout(out_embeds))

        sims = F.softmax(scores, dim=1)
        final_score = torch.log(self.alpha * p_ij + (1 - self.alpha) * sims)
        return final_score

    def loss(self, xs, y_true):
        scores = self.forward(xs)
        loss = self.loss_f(scores, y_true)
        return loss

    def weighting(self, times):
        t_weighted = (self.weight_base + 1.0) / (self.weight_base + times.float())
        return t_weighted

    def save(self, dir, suffix):
        name = "{}_{}_{}.pth".format(self.name, self.birth_time, suffix)
        logger.info(" Saving the trained model as {}".format(name))
        torch.save(self.state_dict(), os.path.join(dir, name))
        return name

    def load(self, dir, name):
        logger.info("Loading the pretrained model from {}{}".format(dir, name))
        if self.device == torch.device("cpu"):
            checkpoint = torch.load(os.path.join(dir, name), map_location=self.device)
        else:
            checkpoint = torch.load(os.path.join(dir, name))

        self.load_state_dict(checkpoint, strict=False)
