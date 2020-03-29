# coding=utf-8
"""Data processors per task"""

import logging
from abc import ABC, abstractmethod

from tfds import SequenceClassification
import tensorflow_datasets as tfds

from transformers import is_tf_available, is_torch_available
from transformers import InputExample, InputFeatures


logger = logging.getLogger(__name__)


class DataProcessor(ABC):
    """Base class for data converters for sequence classification data sets."""
    def __init__(self, **config):
        self.examples = {}
        self.examples["train"] = []
        self.examples["validation"] = []
        self.examples["test"] = []
        self.files = {}
        self.files["train"] = config.pop("train_file", None)
        self.files["validation"] = config.pop("dev_file", None)
        self.files["test"] = config.pop("test_file", None)
        self.ds = None
        self.info = None

        assert len(config) == 0, "unrecognized params passed: %s" % ",".join(config.keys())

    def num_examples(self, mode):
        return self.info.splits[mode].num_examples

    def create_examples(self):
        for mode in ["train", "validation", "test"]:
            tf_dataset = self.ds[mode]

            for ex_index, entry in enumerate(tf_dataset):
                if ex_index % 10000 == 0:
                    logger.info("Creating example %d", ex_index)

                example = self.get_example_from_tensor_dict(entry, ex_index)

                self.examples[mode].append(example)

    @abstractmethod
    def get_example_from_tensor_dict(self, tensor_dict, id):
        pass

    @abstractmethod
    def convert_examples_to_features(self, mode, tokenizer, max_len, return_dataset="tf"):
        pass


class DataProcessorForSequenceClassification(DataProcessor):
    def __init__(self, **config):
        self.guid = config.pop("guid", "guid")
        self.text_a = config.pop("text_a", "text_a")
        self.text_b = config.pop("text_b", "text_b")
        self.label = config.pop("label", "label")
        self.dataset_name = config.pop("dataset_name", None)
        self.is_column_id = config.pop("is_column_id", False)
        self.skip_first_row = config.pop("skip_first_row", False)
        self.delimiter = config.pop("delimiter", "\t")
        self.quotechar = config.pop("quotechar", "\"")        
        super().__init__(**config)

        if self.dataset_name.split("/")[0] in tfds.list_builders():
            self.ds, self.info = tfds.load(self.dataset_name, with_info=True)
        else:
            self.ds, self.info = tfds.load("sequence_classification",
                                           builder_kwargs={
                                               "train_file": self.files["train"],
                                               "dev_file": self.files["validation"],
                                               "test_file": self.files["test"],
                                               "dataset_name": self.dataset_name,
                                               "skip_first_row": self.skip_first_row,
                                               "is_column_id": self.is_column_id,
                                               "delimiter": self.delimiter,
                                               "quotechar": self.quotechar}, with_info=True)

        if self.text_a not in list(self.info.features.keys()):
            raise ValueError("The feature name " + self.text_a + " doesn't exists in " + list(self.info.features.keys()))

        if self.label not in list(self.info.features.keys()):
            raise ValueError("The feature name " + self.label + " doesn't exists in " + list(self.info.features.keys()))

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return self.info.features[self.label].names

    def convert_examples_to_features(self, mode, tokenizer, max_len, return_dataset="tf"):
        if max_len is None:
            max_len = tokenizer.max_len

        features = []

        for (ex_index, example) in enumerate(self.examples[mode]):
            if ex_index % 10000 == 0:
                logger.info("Tokenizing example %d", ex_index)
            
            feature = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_len, pad_to_max_length=True)
            label = self.info.features[self.label].str2int(example.label)

            assert len(feature["input_ids"]) == max_len
            assert len(feature["attention_mask"]) == max_len
            assert len(feature["token_type_ids"]) == max_len

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("input_ids: %s" % " ".join([str(x) for x in feature["input_ids"]]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in feature["attention_mask"]]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in feature["token_type_ids"]]))
                logger.info("label: %s (id = %d)" % (example.label, label))

            features.append(InputFeatures(input_ids=feature["input_ids"],
                                          attention_mask=feature["attention_mask"],
                                          token_type_ids=feature["token_type_ids"],
                                          label=label))
        if len(features) == 0:
            return None

        if return_dataset == "tf":
            if not is_tf_available():
                raise RuntimeError("return_dataset set to 'tf' but TensorFlow 2.0 can't be imported")

            import tensorflow as tf

            def gen():
                for ex in features:
                    yield ({"input_ids": ex.input_ids, "attention_mask": ex.attention_mask, "token_type_ids": ex.token_type_ids}, ex.label)

            dataset = tf.data.Dataset.from_generator(
                gen,
                ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
                ({"input_ids": tf.TensorShape([None]), "attention_mask": tf.TensorShape([None]), "token_type_ids": tf.TensorShape([None])}, tf.TensorShape([])),
            )

            return dataset
        elif return_dataset == "pt":
            if not is_torch_available():
                raise RuntimeError("return_dataset set to 'pt' but PyTorch can't be imported")

            import torch
            from torch.utils.data import TensorDataset

            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)

            return dataset
        else:
            raise ValueError("return_tensors should be one of 'tf' or 'pt'")

    def get_example_from_tensor_dict(self, tensor_dict, id):
        """Get an example from a dict with tensorflow tensors
        Args:
            tensor_dict: Keys and values should match the corresponding Glue
                tensorflow_dataset examples.
        """
        if self.guid in list(self.info.features.keys()):
            guid = tensor_dict[self.guid].numpy()
        else:
            guid = id

        if self.text_b in list(self.info.features.keys()):
            text_b = tensor_dict[self.text_b].numpy().decode("utf-8")
        else:
            text_b = None

        return InputExample(
            guid,
            tensor_dict[self.text_a].numpy().decode("utf-8"),
            text_b,
            self.info.features[self.label].int2str(tensor_dict[self.label].numpy())
        )
