# coding=utf-8
"""Data processors per task"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List
import csv
import json
import os

from transformers import InputExample, InputFeatures
from transformers import PreTrainedTokenizer
from transformers import is_tf_available, is_torch_available


logger = logging.getLogger(__name__)


class DatasetInfo(object):
    def __init__(self, labels, sizes):
        self.labels = labels
        self.sizes = sizes

    def save(self, cache_dir):
        d = {"labels": self.labels, "sizes": self.sizes}
        json_string = json.dumps(d, indent=2, sort_keys=True) + "\n"
        path = os.path.join(cache_dir, "dataset_info.json")

        with open(path, "w", encoding="utf-8") as writer:
            writer.write(json_string)

    @classmethod
    def load(cls, cache_dir):
        path = os.path.join(cache_dir, "dataset_info.json")

        with open(path, "r", encoding="utf-8") as reader:
            text = reader.read()

        json_string = json.loads(text)

        return cls(**json_string)


class DataProcessor(ABC):
    """Base class for data converters for sequence classification data sets."""
    def __init__(self, **config):
        self.examples: Dict[str, List[InputExample]] = {}
        self.examples["train"] = []
        self.examples["validation"] = []
        self.examples["test"] = []
        self.files: Dict[str, str] = {}
        self.files["train"] = config.pop("train_file", None)
        self.files["validation"] = config.pop("dev_file", None)
        self.files["test"] = config.pop("test_file", None)
        self.format = config.pop("format", None)
        self.skip_first_row: bool = config.pop("skip_first_row", False)
        self.delimiter: str = config.pop("delimiter", "\t")
        self.quotechar: str = config.pop("quotechar", "\"")
        self.info = None

        assert len(config) == 0, "unrecognized params passed: %s" % ",".join(config.keys())

    def datasetinfo(self):
        return self.info

    @abstractmethod
    def _read_csv_and_create_examples(self, mode):
        pass

    @abstractmethod
    def _read_tfds_and_create_examples(self, mode):
        pass

    @abstractmethod
    def _read_json_and_create_examples(self, mode):
        pass

    def create_examples(self):
        for mode in ["train", "validation", "test"]:
            if self.format == "csv":
                self._read_csv_and_create_examples(mode)
            elif self.format == "json":
                self._read_json_and_create_examples(mode)
            elif self.format == "tfds":
                self._read_tfds_and_create_examples(mode)
            else:
                raise ValueError("The format {} is not allowed, take one of csv, json or tfds".format(self.format))

        self.info = DatasetInfo(self.labels, {k: len(v) for k, v in self.examples.items()})

    @abstractmethod
    def convert_examples_to_features(self, mode: str, tokenizer: PreTrainedTokenizer, max_len: int, return_dataset: str = "tf"):
        pass


class DataProcessorForSequenceClassification(DataProcessor):
    def __init__(self, **config):
        self.guid = config.pop("guid", None)
        self.text_a = config.pop("text_a", None)
        self.text_b = config.pop("text_b", None)
        self.label = config.pop("label", None)
        self.dataset_name: str = config.pop("dataset_name", None)
        self.labels = []
        super().__init__(**config)

        if self.text_a is None:
            raise ValueError("The text_a parameter is missing in the configuation.")

        if self.label is None:
            raise ValueError("The label parameter is missing in the configuration")

        if self.format == "csv" and self.guid is not None and type(self.guid) != int:
            raise ValueError("If the format is CSV the guid parameter must be an integer.")

        if self.format == "csv" and type(self.text_a) != int:
            raise ValueError("If the format is CSV the text_a parameter must be an integer.")

        if self.format == "csv" and self.text_b is not None and type(self.text_b) != int:
            raise ValueError("If the format is CSV the text_b parameter must be an integer.")

        if self.format == "csv" and type(self.label) != int:
            raise ValueError("If the format is CSV the label parameter must be an integer.")

        if self.format == "tfds" and self.guid is not None and type(self.guid) != str:
            raise ValueError("If the format is TFDS the guid parameter must be an integer.")

        if self.format == "tfds" and type(self.text_a) != str:
            raise ValueError("If the format is TFDS the text_a parameter must be an integer.")

        if self.format == "tfds" and self.text_b is not None and type(self.text_b) != str:
            raise ValueError("If the format is TFDS the text_b parameter must be an integer.")

        if self.format == "tfds" and type(self.label) != str:
            raise ValueError("If the format is TFDS the label parameter must be an integer.")

        if self.format == "tfds" and self.dataset_name is None:
            raise ValueError("If the format is TFDS the dataset_name parameter is mandatory")

    def get_labels(self):
        return self.labels

    def _read_csv_and_create_examples(self, mode):
        if not self.files[mode]:
            return

        with open(self.files[mode], encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=self.delimiter, quotechar=self.quotechar)

            if self.skip_first_row:
                lines = list(reader)[1:]
            else:
                lines = list(reader)

        seen_labels = set()

        for (i, line) in enumerate(lines):
            if i % 10000 == 0:
                logger.info("Creating example %d", i)

            if self.guid == -1:
                id = i
            else:
                id = line[self.guid]

            if self.text_b == -1:
                text_b = ""
            else:
                text_b = line[self.text_b]

            seen_labels.add(line[self.label])

            self.examples[mode].append(InputExample(id,
                                                    line[self.text_a],
                                                    text_b,
                                                    line[self.label]))

        self.labels = list(set(self.labels).union(seen_labels))

    def _read_tfds_and_create_examples(self, mode):
        import tensorflow_datasets as tfds
        try:
            ds, dsinfo = tfds.load(self.dataset_name, split=mode, with_info=True)
        except KeyError:
            return

        seen_labels = set()

        for ex_index, entry in enumerate(ds):
            if ex_index % 10000 == 0:
                logger.info("Creating example %d", ex_index)

            if self.guid in list(dsinfo.features.keys()):
                guid = entry[self.guid].numpy()
            else:
                guid = id

            if self.text_b in list(dsinfo.features.keys()):
                text_b = entry[self.text_b].numpy().decode("utf-8")
            else:
                text_b = None

            label = dsinfo.features[self.label].int2str(entry[self.label].numpy())
            seen_labels.add(label)

            example = InputExample(
                guid,
                entry[self.text_a].numpy().decode("utf-8"),
                text_b,
                label
            )

            self.examples[mode].append(example)

        self.labels = list(set(self.labels).union(seen_labels))

    def _read_json_and_create_examples(self, mode):
        pass

    def convert_examples_to_features(self, mode: str, tokenizer: PreTrainedTokenizer, max_len: int, return_dataset: str = "tf"):
        if max_len is None:
            max_len = tokenizer.max_len

        features = []

        for (ex_index, example) in enumerate(self.examples[mode]):
            if ex_index % 10000 == 0:
                logger.info("Tokenizing example %d", ex_index)

            feature = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_len, pad_to_max_length=True)
            label = self.labels.index(example.label)

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
            token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_attention_mask, token_type_ids, all_labels)

            return dataset
        else:
            raise ValueError("return_tensors should be one of 'tf' or 'pt'")
