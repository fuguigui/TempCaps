import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from measure import Measure
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

logger = logging.getLogger(__name__)

class BaseTester(object):
    def __init__(self, dataset, params, model, device):
        self.model = model.to(device)
        self.model.eval()
        self.dataset = dataset
        self.measures = [Measure(name=name, verbose=getattr(params, "verbose", True)) for name in ['subject', 'object']]
        self.device = device
        self.test_size = params.test_size
        self.cache_dir = params.cache_dir
        self.prefix = params.checkpoint.split(".")[0]
        self.verbose = params.verbose
        self.shuffle = params.eval_shuffle
        self.save_eval = params.save_eval
        self.measure_relation = getattr(params, 'measure_relation', False)
        self.train_types = params.dataset_types[:-1]


    @torch.no_grad()
    def test(self, tb_writer=None, epoch=0, valid_or_test="test", aimed_types=['sub', 'obj']):
        logger.info("===============================")
        logger.info(" ****** Evaluating {} data ******".format(valid_or_test))

        test_dataloader = DataLoader(torch.arange(self.dataset.len['eval']),
                                         batch_size=self.test_size, shuffle=self.shuffle)

        skipped_data = 0
        for batch_idx in tqdm(test_dataloader, desc="Testing",  mininterval=20):
            for i, missing in enumerate(aimed_types):
                if self.measure_relation:
                    xs, y_true, rank_excluded, rels = self.dataset.nextEvalBatch(batch_idx, mode=valid_or_test, type=missing)
                else:
                    xs, y_true, rank_excluded = self.dataset.nextEvalBatch(batch_idx, mode=valid_or_test,
                                                                                 type=missing)
                if y_true.shape[0] < 1:
                    skipped_data += batch_idx.shape[0]
                    logger.warning(
                        "No history data. Skip!\nAccumulated Number of Missing Data: {} for types: {}".format(
                            skipped_data, aimed_types))
                    continue

                y_pred = self.model(xs)
                y_pred = y_pred.detach().cpu().numpy()
                y_true = y_true.detach().cpu().numpy()
                if y_pred.shape[-1] == 1:
                    y_pred = y_pred.squeeze(-1)
                if self.measure_relation:
                    self.measures[i].update(y_pred, y_true, rank_excluded, relations=rels)
                else:
                    self.measures[i].update(y_pred, y_true, rank_excluded)
                self.measures[i].summary()


        for m in self.measures:
            m.summary()
            if self.save_eval:
                m.save(self.cache_dir, self.prefix)

        ttl_measure = None
        if len(aimed_types) > 1:
            ttl_measure = self.summary(self.measures)
            if self.save_eval:
                ttl_measure.save(self.cache_dir, self.prefix)

        if tb_writer is not None:
            for metric in ['hits1', 'hits3', 'hits10', 'mr', 'mrr']:
                tb_writer.add_scalar(valid_or_test + '/' + missing,
                                     self.measures[i].report[metric], epoch)
            if ttl_measure:
                for metric in ['hits1', 'hits3', 'hits10', 'mr', 'mrr']:
                    tb_writer.add_scalar(valid_or_test + '/' + missing,
                                         ttl_measure.report[metric], epoch)


    @classmethod
    def summary(cls, *measures):
        ttl_measure = Measure(*measures, name="total", verbose=True)
        ttl_measure.summary()
        return ttl_measure
