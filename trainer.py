import os
import logging
import math
import itertools
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple

import numpy as np
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset

from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification, PreTrainedModel

from data_processors import DataProcessor, DataProcessorForSequenceClassification, DatasetInfo
from configuration_trainer import TrainerConfig

logger = logging.getLogger(__name__)


class Trainer(ABC):
    def __init__(self, config_path: str = None, config: TrainerConfig = None, **kwargs):
        """
        The list of keys in kwargs here should be generic to all the possible models/architectures
        and not specific to such or such dataset/task.
        """
        if config and not config_path:
            if not isinstance(config, TrainerConfig):
                raise ValueError(
                    "Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. ".format(
                        self.__class__.__name__)
                )
                self.config = config
        elif config_path and not config:
            self.config, unused_kwargs = TrainerConfig.from_trainer(config_path, return_unused_kwargs=True, return_unused_config=True, **kwargs)
        else:
            raise ValueError("the config_path and config parameters cannot be both filled or None.")

        self.strategy_name: bool = unused_kwargs.pop("strategy_name", "onedevice")
        self.data_processor_config: Dict = unused_kwargs.pop("data_processor", None)

        assert len(unused_kwargs) == 0, "unrecognized params passed: %s" % ",".join(unused_kwargs.keys())

        self.datasets: Dict[str, TensorDataset] = {}
        self.processor: DataProcessor
        self.model_class: PreTrainedModel
        self.dataset_info: DatasetInfo

    def setup_training(self, checkpoint_path: str = "checkpoints", log_path: str = "logs", data_cache_dir: str = "cache", model_cache_dir: Optional[str] = None) -> None:
        """
        Setup the different steps to train a model:
        - check if all the data are given
        - create the proper strategy
        - create the features
        - prepare the model settings

        Args:
        checkpoint_path: the directory path where the model checkpoints will be saved, "./checkpoints" folder by default.
        log_path: the directory path where the Tensorboard logs will be saved, "./logs" folder by default.
        data_cache_dir: the directory path where the data will be cached, "./cache" folder by default.
        model_cache_dir (optional): the directory path where the pretrained model will be cached.
        """
        self._prepare_dataset(data_cache_dir, model_cache_dir)
        self._config_trainer(model_cache_dir)

    def _config_trainer(self, model_cache_dir: Optional[str] = None) -> None:
        """
        This method set all the required fields for a specific task. For example
        in case of a classification set all the labels.
        Args:
          model_cache_dir (optional): the directory path where the pretrained model will be cached.
        """
        self.model_config = AutoConfig.from_pretrained(self.config.pretrained_model_name_or_path, cache_dir=model_cache_dir)

    def _prepare_dataset(self, data_cache_dir: str, model_cache_dir: str) -> None:
        """
        Prepare the training, validation and test data.
        Args:
          data_cache_dir: the directory path where the cached data are / should be saved.
        """
        self.datasets, self.dataset_info, self.tokenizer = self.processor.preprocess_data(data_cache_dir, model_cache_dir, self.config.task, self.config.pretrained_model_name_or_path, return_dataset="pt")


class TrainerForSequenceClassification(Trainer):
    def __init__(self, config_path: str = None, config: TrainerConfig = None, **kwargs):
        super().__init__(config_path, config, **kwargs)
        self.processor = DataProcessorForSequenceClassification(**self.data_processor_config)
        self.model_class = AutoModelForSequenceClassification
        self.labels: List[str] = []

    def get_labels(self) -> List[str]:
        """
        Returns the list of labels associated to the trained model.
        """
        return self.dataset_info.labels

    def _config_trainer(self, model_cache_dir: str) -> None:
        self.labels = self.dataset_info.labels
        self.label2id = {label: i for i, label in enumerate(self.labels)}
        self.id2label = {i: label for i, label in enumerate(self.labels)}
        self.model_config = AutoConfig.from_pretrained(self.config.pretrained_model_name_or_path, num_labels=len(self.labels), id2label=self.id2label, label2id=self.label2id, cache_dir=model_cache_dir)
