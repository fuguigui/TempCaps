from collections import OrderedDict
from utils import Struct


from dataset import DyRDataset, StaticKGDataset
from trainer import BaseTrainer, MixTrainer
from tester import BaseTester
from model.dyr import DyRMLP, BaseDyR
from model.baselines import TransE, DistMult, SimplE, RGCNLinkPredict

DATASET_MAPPING = OrderedDict(
    [
        ("dyr", DyRDataset),
        ("transe", StaticKGDataset),
        ("distmult", StaticKGDataset),
        ("simple", StaticKGDataset),
        ("basedyr", DyRDataset),
        ('rgcn', StaticKGDataset)
    ]
)

MODEL_MAPPING = OrderedDict(
    [
        ("dyr", DyRMLP),
        ('transe', TransE),
        ("distmult", DistMult),
        ("simple", SimplE),
        ('basedyr', BaseDyR),
        ('rgcn', RGCNLinkPredict)

    ]
)

TRAINER_MAPPING = OrderedDict(
    [
        ("dyr", BaseTrainer),
        ('transe', MixTrainer),
        ('distmult', MixTrainer),
        ('simple', MixTrainer),
        ('basedyr', BaseTrainer),
        ('rgcn', MixTrainer)
    ]
)

TESTER_MAPPING = OrderedDict(
    [
        ("dyr", BaseTester),
        ('transe', BaseTester),
        ('distmult', BaseTester),
        ('simple', BaseTester),
        ('basedyr', BaseTester),
        ('rgcn', BaseTester)
    ]
)

class AutoDataset:
    r"""
    This is a generic dataset class that will be instantiated as one of the dataset classes of the library
    when created with the :meth:`~auto.AutoDataset.for_model` class method.
    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoDataset is designed to be instantiated "
            "using the `AutoDataset.for_model(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def for_model(cls, params, device):
        if params.model in DATASET_MAPPING:
            dataset_class = DATASET_MAPPING[params.model]
            return dataset_class(params, device)
        raise ValueError(
            f"Unrecognized model identifier: {params.model}. Should contain one of {', '.join(DATASET_MAPPING.keys())}"
        )

class AutoModel:
    r"""
    This is a generic model class that will be instantiated as one of the model classes of the library
    when created with the :meth:`~auto.AutoModel.for_model` class method.
    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.for_model(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def for_model(cls, params, device):
        if params.model in MODEL_MAPPING:
            model_class = MODEL_MAPPING[params.model]
            if params.model in ["naive", "capse"]:
                args = Struct(**params.decoder)
                return model_class(args, device=device)
            else:
                return model_class(params, device=device)
        raise ValueError(
            f"Unrecognized model identifier: {params.model}. Should contain one of {', '.join(MODEL_MAPPING.keys())}"
        )

class AutoTrainer:
    r"""
    This is a generic trainer class that will be instantiated as one of the trainer classes of the library
    when created with the :meth:`~auto.AutoTrainer.for_model` class method.
    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoTrainer is designed to be instantiated "
            "using the `AutoTrainer.for_model(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def for_model(cls, dataset, params, model, device):
        if params.model in TRAINER_MAPPING:
            trainer_class = TRAINER_MAPPING[params.model]
            return trainer_class(dataset, params, model, device)
        raise ValueError(
            f"Unrecognized trainer identifier: {params.model}. Should contain one of {', '.join(TRAINER_MAPPING.keys())}"
        )

class AutoTester:
    r"""
    This is a generic tester class that will be instantiated as one of the tester classes of the library
    when created with the :meth:`~auto.AutoTester.for_model` class method.
    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoTester is designed to be instantiated "
            "using the `AutoTester.for_model(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def for_model(cls, dataset, params, model, device):
        if params.model in TESTER_MAPPING:
            tester_class = TESTER_MAPPING[params.model]
            return tester_class(dataset, params, model, device)
        raise ValueError(
            f"Unrecognized tester identifier: {params.model}. Should contain one of {', '.join(TESTER_MAPPING.keys())}"
        )
