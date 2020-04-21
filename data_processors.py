# coding=utf-8
"""Data processors per task"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List
import csv
import json
import os
from collections import OrderedDict

from transformers import InputExample, InputFeatures
from transformers import PreTrainedTokenizer, AutoTokenizer
from transformers import is_tf_available, is_torch_available


# Move to src/transformers/data/__init__.py
try:
    import tensorflow_datasets  # noqa: F401

    _has_tfds = True
except ImportError:
    _has_tfds = False


def is_tfds_available():
    return _has_tfds


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

    @staticmethod
    def exists(cache_dir):
        path = os.path.join(cache_dir, "dataset_info.json")

        return os.path.exists(path)


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
        self.max_len: int = config.pop("max_len", None)
        self.format = config.pop("format", None)
        self.skip_first_row: bool = config.pop("skip_first_row", False)
        self.delimiter: str = config.pop("delimiter", "\t")
        self.quotechar: str = config.pop("quotechar", None)

        assert len(config) == 0, "unrecognized params passed: %s" % ",".join(config.keys())

    @abstractmethod
    def _read_csv_and_create_examples(self, mode):
        pass

    @abstractmethod
    def _read_tfds_and_create_examples(self, mode):
        pass

    @abstractmethod
    def _read_json_and_create_examples(self, mode):
        pass

    @abstractmethod
    def _load_cached_dataset(self, file, return_dataset="tf"):
        """
        Load a cached dataset.
        Args:
          file: the file path where the cache will be saved.
          dataset: the dataset we want to save.
          input_dataset: if the dataset is from a TF or PT format.
        """
        pass

    @abstractmethod
    def _cache_dataset(self, file, dataset, input_dataset="tf"):
        """
        Cache a dataset.
        Args:
          file: the file path where the cache will be saved.
          dataset: the dataset we want to save.
          input_dataset: if the dataset is from a TF or PT format.
        """
        pass

    def preprocess_data(self, data_cache_dir: str, model_cache_dir: str, task: str, pretrained_model_name_or_path: str, return_dataset: str = "tf"):
        """
        Preprocess the training and validation data.
        Args:
          cache_dir: the directory path where the cached data are / should be saved.
        """
        cached_training_features_file = os.path.join(
            data_cache_dir, "cached_train_{}_{}_{}.{}_record".format(
                task.replace("/", "-"), list(filter(None, pretrained_model_name_or_path.split("/"))).pop(), str(self.max_len), return_dataset
            ),
        )
        cached_validation_features_file = os.path.join(
            data_cache_dir, "cached_validation_{}_{}_{}.{}_record".format(
                task.replace("/", "-"), list(filter(None, pretrained_model_name_or_path.split("/"))).pop(), str(self.max_len), return_dataset
            ),
        )
        cached_test_features_file = os.path.join(
            data_cache_dir, "cached_test_{}_{}_{}.{}_record".format(
                task.replace("/", "-"), list(filter(None, pretrained_model_name_or_path.split("/"))).pop(), str(self.max_len), return_dataset
            ),
        )
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, cache_dir=model_cache_dir, use_fast=True)
        datasets = {}

        os.makedirs(data_cache_dir, exist_ok=True)

        if os.path.exists(cached_training_features_file):
            logger.info("Loading features from cached file %s", cached_training_features_file)
            datasets["train"] = self._load_cached_dataset(cached_training_features_file, return_dataset)
        else:
            self.create_examples("train")
            datasets["train"] = self._convert_examples_to_features("train", tokenizer, return_dataset)
            self._cache_dataset(cached_training_features_file, datasets["train"], input_dataset=return_dataset)

        if os.path.exists(cached_validation_features_file):
            logger.info("Loading features from cached file %s", cached_validation_features_file)
            datasets["validation"] = self._load_cached_dataset(cached_validation_features_file, return_dataset)
        else:
            self.create_examples("validation")
            datasets["validation"] = self._convert_examples_to_features("validation", tokenizer, return_dataset)
            self._cache_dataset(cached_validation_features_file, datasets["validation"], input_dataset=return_dataset)

        if os.path.exists(cached_test_features_file):
            logger.info("Loading features from cached file %s", cached_test_features_file)
            datasets["test"] = self._load_cached_dataset(cached_test_features_file, return_dataset)
        else:
            self.create_examples("test")
            datasets["test"] = self._convert_examples_to_features("test", tokenizer, return_dataset)

            if datasets["test"] is None:
                datasets["test"] = datasets["validation"]

            self._cache_dataset(cached_test_features_file, datasets["test"], input_dataset=return_dataset)

        if DatasetInfo.exists(data_cache_dir):
            dataset_info = DatasetInfo.load(data_cache_dir)
        else:
            dataset_info = DatasetInfo(self.labels, {k: len(v) for k, v in self.examples.items()})

            dataset_info.save(data_cache_dir)

        return datasets, dataset_info, tokenizer

    def create_examples(self, mode):
        if self.format == "csv":
            self._read_csv_and_create_examples(mode)
        elif self.format == "json":
            self._read_json_and_create_examples(mode)
        elif self.format == "tfds":
            self._read_tfds_and_create_examples(mode)
        else:
            raise ValueError("The format {} is not allowed, take one of csv, json or tfds".format(self.format))

    @abstractmethod
    def _convert_examples_to_features(self, mode: str, tokenizer: PreTrainedTokenizer, return_dataset: str = "tf"):
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
        if not is_tfds_available():
            raise RuntimeError("The package tensorflow_datasets can't be imported")

        import tensorflow_datasets as tfds
        try:
            ds, dsinfo = tfds.load(self.dataset_name, split=mode, with_info=True)
        except KeyError:
            raise ValueError("The dataset {} does not exists in the tensorflow_datasets catalog.".format(self.dataset_name))

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

    def _load_cached_dataset(self, cached_file, return_dataset="tf"):
        if return_dataset == "tf":
            if not is_tf_available():
                raise RuntimeError("return_dataset set to 'tf' but TensorFlow 2.0 can't be imported")

            import tensorflow as tf

            name_to_features = {
                "input_ids": tf.io.FixedLenFeature([self.max_len], tf.int64),
                "attention_mask": tf.io.FixedLenFeature([self.max_len], tf.int64),
                "token_type_ids": tf.io.FixedLenFeature([self.max_len], tf.int64),
                "label": tf.io.FixedLenFeature([1], tf.int64),
            }

            def _decode_record(record):
                example = tf.io.parse_single_example(record, name_to_features)

                return {k : example[k] for k in ('input_ids', 'attention_mask', 'token_type_ids') if k in example}, example["label"]

            d = tf.data.TFRecordDataset(cached_file)
            d = d.map(_decode_record, num_parallel_calls=4)

            return d

        elif return_dataset == "pt":
            if not is_torch_available():
                raise RuntimeError("return_dataset set to 'pt' but PyTorch can't be imported")

            import torch

            d = torch.load(cached_file)

            return d
        else:
            raise ValueError("return_dataset should be one of 'tf' or 'pt'")

    def _cache_dataset(self, file, dataset, input_dataset="tf"):
        if input_dataset == "tf":
            if not is_tf_available():
                raise RuntimeError("return_dataset set to 'tf' but TensorFlow 2.0 can't be imported")

            import tensorflow as tf

            writer = tf.io.TFRecordWriter(file)
            ds = dataset.enumerate()

            for (index, (feature, label)) in ds:
                if index % 10000 == 0:
                    logger.info("Writing example %d", index)

                def create_list_int_feature(values):
                    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                    return f

                def create_int_feature(value):
                    f = tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
                    return f

                record_feature = OrderedDict()
                record_feature["input_ids"] = create_list_int_feature(feature["input_ids"])
                record_feature["attention_mask"] = create_list_int_feature(feature["attention_mask"])
                record_feature["token_type_ids"] = create_list_int_feature(feature["token_type_ids"])
                record_feature["label"] = create_int_feature(label)
                tf_example = tf.train.Example(features=tf.train.Features(feature=record_feature))

                writer.write(tf_example.SerializeToString())

            writer.close()
        elif input_dataset == "pt":
            if not is_torch_available():
                raise RuntimeError("return_dataset set to 'pt' but PyTorch can't be imported")

            import torch

            torch.save(dataset, file)
        else:
            raise ValueError("input_dataset should be one of 'tf' or 'pt'")

    def _convert_examples_to_features(self, mode: str, tokenizer: PreTrainedTokenizer, return_dataset: str = "tf"):
        features = []

        for (ex_index, example) in enumerate(self.examples[mode]):
            if ex_index % 10000 == 0:
                logger.info("Tokenizing example %d", ex_index)

            feature = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=self.max_len, pad_to_max_length=True)
            label = self.labels.index(example.label)

            assert len(feature["input_ids"]) == self.max_len
            assert len(feature["attention_mask"]) == self.max_len
            assert len(feature["token_type_ids"]) == self.max_len

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
            raise ValueError("return_dataset should be one of 'tf' or 'pt'")


class DataProcessorForTokenClassification(DataProcessor):
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

        if self.format == "csv" and type(self.text_a) != int:
            raise ValueError("If the format is CSV the text_a parameter must be an integer.")

        if self.format == "csv" and type(self.label) != int:
            raise ValueError("If the format is CSV the label parameter must be an integer.")

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
        words = []
        labels = []
        guid_index = 1

        for (i, line) in enumerate(lines):
            if i % 10000 == 0:
                logger.info("Creating example %d", i)

            if len(line) == 0 or line[0].startswith("-DOCSTART-"):
                if words:
                    self.examples[mode].append(InputExample(guid_index,
                                                            words,
                                                            "",
                                                            labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                words.append(line[self.text_a])
                if len(line) > 1:
                    labels.append(line[self.label])
                    seen_labels.add(line[self.label])
                else:
                    labels.append("O")
        if words:
            self.examples[mode].append(InputExample(guid_index,
                                                    words,
                                                    "",
                                                    labels))

        self.labels = list(set(self.labels).union(seen_labels))

    def _read_tfds_and_create_examples(self, mode):
        pass

    def _read_json_and_create_examples(self, mode):
        pass

    def _load_cached_dataset(self, cached_file, return_dataset="tf"):
        if return_dataset == "tf":
            if not is_tf_available():
                raise RuntimeError("return_dataset set to 'tf' but TensorFlow 2.0 can't be imported")

            import tensorflow as tf

            name_to_features = {
                "input_ids": tf.io.FixedLenFeature([self.max_len], tf.int64),
                "attention_mask": tf.io.FixedLenFeature([self.max_len], tf.int64),
                "token_type_ids": tf.io.FixedLenFeature([self.max_len], tf.int64),
                "label": tf.io.FixedLenFeature([self.max_len], tf.int64),
            }

            def _decode_record(record):
                example = tf.io.parse_single_example(record, name_to_features)

                return {k : example[k] for k in ('input_ids', 'attention_mask', 'token_type_ids') if k in example}, example["label"]

            d = tf.data.TFRecordDataset(cached_file)
            d = d.map(_decode_record, num_parallel_calls=4)

            return d

        elif return_dataset == "pt":
            if not is_torch_available():
                raise RuntimeError("return_dataset set to 'pt' but PyTorch can't be imported")

            import torch

            d = torch.load(cached_file)

            return d
        else:
            raise ValueError("return_dataset should be one of 'tf' or 'pt'")

    def _cache_dataset(self, file, dataset, input_dataset="tf"):
        if input_dataset == "tf":
            if not is_tf_available():
                raise RuntimeError("return_dataset set to 'tf' but TensorFlow 2.0 can't be imported")

            import tensorflow as tf

            writer = tf.io.TFRecordWriter(file)
            ds = dataset.enumerate()

            for (index, (feature, label)) in ds:
                if index % 10000 == 0:
                    logger.info("Writing example %d", index)

                def create_list_int_feature(values):
                    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                    return f

                record_feature = OrderedDict()
                record_feature["input_ids"] = create_list_int_feature(feature["input_ids"])
                record_feature["attention_mask"] = create_list_int_feature(feature["attention_mask"])
                record_feature["token_type_ids"] = create_list_int_feature(feature["token_type_ids"])
                record_feature["label"] = create_list_int_feature(label)
                tf_example = tf.train.Example(features=tf.train.Features(feature=record_feature))

                writer.write(tf_example.SerializeToString())

            writer.close()
        elif input_dataset == "pt":
            if not is_torch_available():
                raise RuntimeError("return_dataset set to 'pt' but PyTorch can't be imported")

            import torch

            torch.save(dataset, file)
        else:
            raise ValueError("input_dataset should be one of 'tf' or 'pt'")

    def _convert_examples_to_features(self, mode: str, tokenizer: PreTrainedTokenizer, return_dataset: str = "tf"):
        features = []

        for (ex_index, example) in enumerate(self.examples[mode]):
            if ex_index % 10000 == 0:
                logger.info("Tokenizing example %d", ex_index)

            tokens = []
            label_ids = []
            for word, label in zip(example.text_a, example.label):
                word_tokens = tokenizer.tokenize(word)

                if len(word_tokens) > 0:
                    tokens.extend(word_tokens)
                    label_ids.extend([self.labels.index(label)] + [-1] * (len(word_tokens) - 1))

            special_tokens_count = tokenizer.num_special_tokens_to_add()

            if len(tokens) > self.max_len - 2:
                tokens = tokens[: (self.max_len - special_tokens_count)]
                label_ids = label_ids[: (self.max_len - special_tokens_count)]

            tokens += [tokenizer.sep_token]
            label_ids += [-1]
            segment_ids = [0] * len(tokens)
            tokens = [tokenizer.cls_token] + tokens
            label_ids = [-1] + label_ids
            segment_ids = [0] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)
            padding_length = self.max_len - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * padding_length
            input_mask += [0] * padding_length
            segment_ids += [tokenizer.pad_token_type_id] * padding_length
            label_ids += [-1] * padding_length

            assert len(input_ids) == self.max_len
            assert len(input_mask) == self.max_len
            assert len(segment_ids) == self.max_len
            assert len(label_ids) == self.max_len

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in segment_ids]))
                logger.info("label: %s " % (label_ids))

            features.append(InputFeatures(input_ids=input_ids,
                                          attention_mask=input_mask,
                                          token_type_ids=segment_ids,
                                          label=label_ids))
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
                ({"input_ids": tf.TensorShape([None]), "attention_mask": tf.TensorShape([None]), "token_type_ids": tf.TensorShape([None])}, tf.TensorShape([None])),
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
            raise ValueError("return_dataset should be one of 'tf' or 'pt'")
