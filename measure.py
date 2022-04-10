import logging
import numpy as np
import os
from scipy.stats import rankdata

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


class Measure:
    def __init__(self, all_list=None, name="measure", verbose=True):
        self.name = name
        self.ranks = []
        self.raws = None
        self.verbose = verbose
        if all_list is not None:
            for l in all_list:
                self.ranks.extend(l.ranks)

    def update(self, y_preds, y_true, to_excluded=None, update_row=False, relations=None):
        if len(y_preds.shape) == 1:
            y_preds = y_preds.reshape(1, y_preds.shape[0])
        if to_excluded is not None:
            y_mins = y_preds.min(axis=-1)
            for i, ids in enumerate(to_excluded):
                y_preds[i, ids] = y_mins[i]
        if update_row:
            if self.raws is None:
                self.raws = y_preds
            else:
                self.raws = np.row_stack((self.raws, y_preds))
        results_with_id = rankdata(-y_preds, method='min', axis=-1)
        # if len(results_with_id.shape) == 1:
        #     results_with_id = results_with_id.reshape(1, results_with_id.shape[0])
        for i, j in enumerate(y_true):
            self.ranks.append(results_with_id[i, j].item())

        if relations is not None:
            if not hasattr(self, 'relations'):
                self.relations = dict()
            for i, j in enumerate(y_true):
                if relations[i] not in self.relations.keys():
                    self.relations[relations[i]] = [results_with_id[i, j].item()]
                else:
                    self.relations[relations[i]].append(results_with_id[i, j].item())


    def summary(self):
        ranks = np.asarray(self.ranks)
        self.report = dict()
        self.report['mr'] = np.mean(ranks)
        self.report['mrr'] = np.mean(1.0 / ranks)
        self.report['hits1'] = np.mean(ranks <= 1)
        self.report['hits3'] = np.mean(ranks <= 3)
        self.report['hits10'] = np.mean(ranks <= 10)

        if hasattr(self, 'relations'):
            self.report_rels = dict()
            for rel, ranks in self.relations.items():
                self.report_rels[rel] = dict()
                ranks = np.asarray(ranks)
                self.report_rels[rel]['mr'] = np.mean(ranks)
                self.report_rels[rel]['mrr'] = np.mean(1.0 / ranks)
                self.report_rels[rel]['hits1'] = np.mean(ranks <= 1)
                self.report_rels[rel]['hits3'] = np.mean(ranks <= 3)
                self.report_rels[rel]['hits10'] = np.mean(ranks <= 10)

        if self.verbose:
            self.print_()

    def print_(self):
        logger.info("Measurement for {}:".format(self.name))
        for key, val in self.report.items():
            logger.info("\t{} = {}".format(key, val))
        if hasattr(self, 'relations'):
            for rel, ranks in self.report_rels.items():
                logger.info('Relation: {}'.format(rel))
                for key, val in ranks.items():
                    logger.info("\t{} = {}".format(key, val))


    def save(self, file_dir, file_prefix):
        file_name = file_prefix + self.name + ".txt"
        with open(os.path.join(file_dir, file_name), 'w') as filehandle:
            for listitem in self.ranks:
                filehandle.write('%s\n' % listitem)

        np.save(os.path.join(file_dir, file_prefix + self.name + "raws.npy"), self.raws)

        if hasattr(self, 'relations'):
            # for rel, ranks in self.relations.items():
            #     file_name = file_prefix + self.name + "_rel_{}.txt".format(rel)
            #     with open(os.path.join(file_dir, file_name), 'w') as filehandle:
            #         for listitem in ranks:
            #             filehandle.write('%s\n' % listitem)
            file_name = file_prefix + self.name + '_rel_summary.txt'
            with open(os.path.join(file_dir, file_name), 'w') as filehandle:
                for rel, ranks in self.report_rels.items():
                    filehandle.write('Relation: {}'.format(rel))
                    for key, val in ranks.items():
                        filehandle.write("\t{} = {}".format(key, val))
                    filehandle.write("\n")

    def load(self, file_dir, file_prefixes):
        if isinstance(file_prefixes, str):
            file_prefixes = [file_prefixes]
        for file_prefix in file_prefixes:
            with open(os.path.join(file_dir, file_prefix), 'r') as filehandle:
                rank = [int(current_place.rstrip()) for current_place in filehandle.readlines()]
            self.ranks.extend(rank)
            raw_scores = np.load(file_prefix.split(".")[0] + "raws.npy")
            if self.raws is None:
                self.raws = raw_scores
            else:
                self.raws = np.row_stack((self.raws, raw_scores))

            # if hasattr(self, 'relations'):
            #     file_name = file_prefix + self.name + "_rels.txt"
            #     with open(os.path.join(file_dir, file_name), 'r') as filehandle:
            #         for rel, ranks in self.report_rels.items():
            #             filehandle.write('%s\n' % rel)
            #             for listitem in ranks:
            #                 filehandle.write('%s\n' % listitem)