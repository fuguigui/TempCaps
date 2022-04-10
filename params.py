import logging
import yaml
import argparse
import os
import utils

from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK']='True'
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

logger = logging.getLogger(__name__)

class Params:
    def __init__(self):
        logger.info("****** Parsing arguments ******")
        parser = argparse.ArgumentParser(description='Experimental Settings')

        model_g = utils.ArgumentGroup(parser, "model", "model configuration and paths.")
        model_g.add_arg("config_path", lambda p: Path(p).absolute(), "configs/dyrmlp.yml",
                        "Path to the json file for the model configuration.")
        model_g.add_arg("from_pretrained", utils.str2bool, False,
                           "Whether to load parameters from a pretrained model")
        model_g.add_arg("fix_pretrained", utils.str2bool, False,
                           "Whether to fix the pretrained model's embeddings")

        train_g = utils.ArgumentGroup(parser, "training", "training options.")
        train_g.add_arg("lr", float, 1e-3, "learning rate")
        train_g.add_arg("seed", int, 1996, "random seed for numpy/pytorch, etc")
        train_g.add_arg("weight_decay", float, 1e-5, "Weight decay rate for L2 regularizer.")
        train_g.add_arg("batch_size", int, 16, "The number of samples in each batch for training.")
        train_g.add_arg("test_size", int, 1028, "The number of samples in each batch for testing.")
        train_g.add_arg("max_epochs", int, 30, "Maximum training epochs.")
        train_g.add_arg("save_steps", int, 5, "The steps interval to save checkpoints.")
        train_g.add_arg("eval_steps", int, 5, "The steps to evaluate validation dataset.")
        train_g.add_arg("loss", str, "cross", "The type of training loss function", choices=["cross", "softplus", "margin", "bpr"])

        data_g = utils.ArgumentGroup(parser, "data", "Data paths, vocab paths and data processing options")
        data_g.add_arg("overwrite", utils.str2bool, False, "Whether to overwrite the preprocessed dataset.")
        data_g.add_arg("neg_ratio", int, 1, "The negative sampling ratio for one positive sample.")
        data_g.add_arg("time_scale", int, 24, "The time scale of two conitunous time points.")
        data_g.add_arg("max_time_range", int, 10, "The visible range of facts when selecting neighbors.")
        data_g.add_arg("direct_id", utils.str2bool, False, "Indicate if the data has been transfered into ID.")

        run_type_g = utils.ArgumentGroup(parser, "run_type", "Experiment running options.")
        run_type_g.add_arg("use_cuda", utils.str2bool, True, "If set, use GPU for training.")
        run_type_g.add_arg("do_train", utils.str2bool, True, "Whether to perform training.")
        run_type_g.add_arg("do_test", utils.str2bool, True, "Whether to perform prediction on test dataset.")
        run_type_g.add_arg("save_eval", utils.str2bool, False, "Whether to save the detailed evaluation records.")
        run_type_g.add_arg("train_shuffle", utils.str2bool, True, "Whether to shuffle the training data.")
        run_type_g.add_arg("eval_shuffle", utils.str2bool, True, "Whether to shuffle the validataion or testing data.")
        run_type_g.add_arg("resume_train", utils.str2bool, False, "whether to resume training from a checkpoint.")
        run_type_g.add_arg("task", str, "completion", "The type of the downstream task", choices = ["completion", "forecasting", "recommendation"])
        run_type_g.add_arg("verbose", utils.str2bool, True, "whether to log detailed information.")
        run_type_g.add_arg("data_mode", str, 'valid', "Valid or Test, using different training data.")

        mem_settings_g = utils.ArgumentGroup(parser, "memory", "memory settings.")
        mem_settings_g.add_arg("cache_dir", str, "cache/", "The path to cache trained model.")
        mem_settings_g.add_arg("data_dir", str, 'data/completion/', "Path to load the dataset.")
        mem_settings_g.add_arg("dataset", str, 'icews14', "The name of the dataset as a folder name in the data_dir")
        mem_settings_g.add_arg("checkpoint", str, "transcapse_1603-1513_50_ICEWS14.pth",
                               "The name of saved checkpoint model for testing or resuming train.")
        mem_settings_g.add_arg("suffix", str, "",
                               "The suffix of saved model's name.")

        parser.add_argument('--checkpoints', nargs='+', default=[])
        args = parser.parse_args()
        for arg in vars(args):
            self.__dict__[arg] = getattr(args, arg)

        with args.config_path.open(mode='r') as yamlfile:
            configs = yaml.safe_load(yamlfile)

        self.update(**configs)
        self.set_params_by_default()
        self.print()


    def update(self, **entries):
        self.__dict__.update(entries)

    def set_params_by_default(self):
        # dataset related default settings
        if self.dataset in ["ICEWS14"]:
            self.dataset_types = ['train', 'test']
            self.eval_steps = -1
        else:
            self.dataset_types = ['train', 'valid', 'test']

        if self.dataset in ["YAGO", "WIKI"]:
            if self.task == 'completion':
                self.direct_id = True
            else:
                self.time_scale = 1

        elif self.dataset in ['icews14', 'icews05-15', 'gdelt', 'icews11-14']:
            self.time_scale = -1
            if self.dataset == "icews05-15":
                self.start_time = "2005-01-01"
            elif self.dataset == "gdelt":
                self.start_time = "2015-04-01"
            elif self.dataset == 'icews11-14':
                if self.task == 'forecasting':
                    self.direct_id = True
                else:
                    self.start_time = '2011-01-01'
        elif self.dataset in ["ICEWS14", "ICEWS18"]:
            self.time_scale = 24
        elif self.dataset in ["GDELT"]:
            self.time_scale = 15

        # model related parameters:
        if self.model in ["transe", "distmult", "simple"]:
            self.test_size = 1
        if "forecast" in self.data_dir:
            self.train_shuffle = False
            self.eval_shuffle = False

        self.num_e, self.num_r, _ = utils.get_total_number(
            self.data_dir + self.dataset, 'stat.txt')

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        if self.hid_dim < 0:
            self.reflect = True


    def print(self):
        logger.info(" Experiment configurations")
        for key, val in self.__dict__.items():
            logger.info('{}: {}'.format(key, val))



