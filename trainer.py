import logging
import os
import sys
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

logger = logging.getLogger(__name__)

class BaseTrainer(object):
    def __init__(self, dataset, params, model, device):
        self.model = model.to(device)
        self.dataset = dataset
        self.params = params
        self.checkpoint = params.checkpoint
        self.device = device

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.params.lr,
                                          weight_decay=self.params.weight_decay)
        if self.params.resume_train:
            logger.info("===============================")
            logger.info("Resuming from a trained {}".format(self.model.name))
            self.load()
        else:
            logger.info("===============================")
            logger.info("Starting a new training for {} ...".format(self.model.name))
            if not os.path.exists(self.params.cache_dir):
                os.makedirs(self.params.cache_dir)


    def save(self, epoch):
        suffix = "{}_{}".format(epoch, self.params.dataset)
        if not os.path.exists(self.params.cache_dir):
            os.makedirs(self.params.cache_dir)
        name = self.model.save(self.params.cache_dir, suffix)
        logger.info("Saving the trained model and optimizer as {} in {} ...".format(name, self.params.cache_dir))
        torch.save(self.optimizer.state_dict(), os.path.join(self.params.cache_dir, "optimizer_{}".format(name)))

    def load(self):
        logger.info("Loading the pretrained model from the checkpoint {} in {}....".format(self.checkpoint, self.params.cache_dir))
        self.model.load(self.params.cache_dir, self.checkpoint)
        try:
            if self.device == torch.device("cpu"):
                checkpoint = torch.load(os.path.join(self.params.cache_dir,
                                                     "optimizer_{}".format(self.checkpoint)),
                                        map_location=self.device)
            else:
                checkpoint = torch.load(os.path.join(self.params.cache_dir,
                                                     "optimizer_{}".format(self.checkpoint)))

            self.optimizer.load_state_dict(checkpoint)
        except:
            logger.warning("Failing to load pretrained optimizer")

    def train(self, tb_writer=None, aimed_types=['sub', 'obj'], tester=None):
        train_dataloader = DataLoader(torch.arange(self.dataset.len['train']),
                                      batch_size=self.params.batch_size, shuffle=self.params.train_shuffle)
        ttl_cnt = 0
        for epoch in tqdm(range(self.params.max_epochs), desc="Epoch"):
            self.model.train()
            sys.stdout.flush()
            ttl_loss = 0
            batch_cnt = 0
            skipped_data = 0

            for batch_idx in tqdm(train_dataloader, desc="Batch", miniters=100, mininterval=60):
                for missing in aimed_types:
                    self.optimizer.zero_grad()
                    xs, y_true = self.dataset.nextBatch(
                        batch_idx, type=missing)
                    if y_true.shape[0] < 1:
                        skipped_data += batch_idx.shape[0]
                        logger.warning("No history data. Skip!\nAccumulated Number of Missing Data: {} for types: {}".format(skipped_data, aimed_types))
                        continue
                    batch_loss = self.model.loss(xs, y_true)
                    batch_loss.backward()
                    loss = batch_loss.detach().cpu().item()
                    ttl_loss += loss
                    batch_cnt += 1
                    self.optimizer.step()
                    if tb_writer is not None:
                        tb_writer.add_scalar("train/loss", loss, ttl_cnt + batch_cnt)
                    if self.params.verbose:
                        logger.info("Batch {} loss: {}".format(batch_cnt, loss))

            logger.info("loss: {}".format(ttl_loss / batch_cnt))
            ttl_cnt += batch_cnt
            if (epoch + 1) % self.params.save_steps == 0 or (epoch + 1) == self.params.max_epochs:
                self.save(epoch + 1)

            if self.params.eval_steps < 0:
                continue
            if tester is None:
                logger.warning("Non-given Tester: Trying to evaluate the validation dataset during training but no tester is given!")
            if (epoch + 1) % self.params.eval_steps == 0 or (epoch + 1) == self.params.max_epochs:
                tester.test(tb_writer, epoch + 1, valid_or_test="valid", aimed_types=aimed_types)

        logger.info("Training done")
        logger.info("===============================")

class MixTrainer(object):
    def __init__(self, dataset, params, model, device):
        self.model = model.to(device)
        self.dataset = dataset
        self.params = params
        self.checkpoint = params.checkpoint
        self.device = device
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.params.lr,
                                          weight_decay=self.params.weight_decay)
        if self.params.resume_train:
            logger.info("===============================")
            logger.info("Resuming from a trained {}".format(self.model.name))
            self.load()
        else:
            logger.info("===============================")
            logger.info("Starting a new training for {} ...".format(self.model.name))
            if not os.path.exists(self.params.cache_dir):
                os.makedirs(self.params.cache_dir)


    def save(self, epoch):
        suffix = "{}_{}".format(epoch, self.params.dataset)
        if not os.path.exists(self.params.cache_dir):
            os.makedirs(self.params.cache_dir)
        name = self.model.save(self.params.cache_dir, suffix)
        logger.info("Saving the trained model and optimizer as {} in {} ...".format(name, self.params.cache_dir))
        torch.save(self.optimizer.state_dict(), os.path.join(self.params.cache_dir, "optimizer_{}".format(name)))

    def load(self):
        logger.info("Loading the pretrained model from the checkpoint {} in {}....".format(self.checkpoint, self.params.cache_dir))
        self.model.load(self.params.cache_dir, self.checkpoint)
        try:
            if self.device == torch.device("cpu"):
                checkpoint = torch.load(os.path.join(self.params.cache_dir,
                                                     "optimizer_{}".format(self.checkpoint)),
                                        map_location=self.device)
            else:
                checkpoint = torch.load(os.path.join(self.params.cache_dir,
                                                     "optimizer_{}".format(self.checkpoint)))

            self.optimizer.load_state_dict(checkpoint)
        except:
            logger.warning("Failing to load pretrained optimizer")

    def train(self, tb_writer=None, tester=None, early_stop=False):
        train_dataloader = DataLoader(torch.arange(self.dataset.len['train']),
                                      batch_size=self.params.batch_size, shuffle=self.params.train_shuffle)
        ttl_cnt = 0
        for epoch in tqdm(range(self.params.max_epochs), desc="Epoch"):
            self.model.train()
            sys.stdout.flush()
            ttl_loss = 0
            batch_cnt = 0
            skipped_data = 0

            i = 0
            for batch_idx in tqdm(train_dataloader, desc="Batch", miniters=100, mininterval=60):
                i += 1
                if early_stop:
                    if i>3:
                        break

                self.optimizer.zero_grad()
                xs, y_true = self.dataset.nextBatch(batch_idx)
                if y_true.shape[0] < 1:
                    skipped_data += batch_idx.shape[0]
                    logger.warning("No history data. Skip!\nAccumulated Number of Missing Data: {} for types: {}".format(skipped_data, aimed_types))
                    continue
                batch_loss = self.model.loss(xs, y_true)
                batch_loss.backward()
                loss = batch_loss.detach().cpu().item()
                ttl_loss += loss
                batch_cnt += 1
                self.optimizer.step()
                if tb_writer is not None:
                    tb_writer.add_scalar("train/loss", loss, ttl_cnt + batch_cnt)
                if self.params.verbose:
                    logger.info("Batch {} loss: {}".format(batch_cnt, loss))

            logger.info("loss: {}".format(ttl_loss / batch_cnt))
            ttl_cnt += batch_cnt
            if (epoch + 1) % self.params.save_steps == 0 or (epoch + 1) == self.params.max_epochs:
                self.save(epoch + 1)

            if self.params.eval_steps < 0:
                continue
            if tester is None:
                logger.warning("Non-given Tester: Trying to evaluate the validation dataset during training but no tester is given!")
            if (epoch + 1) % self.params.eval_steps == 0 or (epoch + 1) == self.params.max_epochs:
                tester.test(tb_writer, epoch + 1, valid_or_test="valid")

        logger.info("Training done")
        logger.info("===============================")
