import logging
import numpy as np
import torch
import os

from utils import count_params
from params import Params
from torch.utils.tensorboard import SummaryWriter
from auto import AutoModel, AutoDataset, AutoTrainer, AutoTester
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    params = Params()
    if params.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    torch.manual_seed(params.seed)
    np.random.seed(params.seed)

    model = AutoModel.for_model(params, device)
    count_params(model)

    if params.do_train:
        params.data_mode = 'valid'
        dataset = AutoDataset.for_model(params, device)
        comment = f'batch_size{params.batch_size}lr{params.lr}'
        tb = SummaryWriter(comment=comment)
        trainer = AutoTrainer.for_model(dataset, params, model, device)
        tester = AutoTester.for_model(dataset, params, model, device)
        trainer.train(tb, tester=tester)
        model = trainer.model
    else:
        model.load(params.cache_dir, params.checkpoint)

    if params.do_test:
        params.data_mode = 'test'
        dataset = AutoDataset.for_model(params, device)
        tester = AutoTester.for_model(dataset, params, model, device)
        tester.test(valid_or_test='test')