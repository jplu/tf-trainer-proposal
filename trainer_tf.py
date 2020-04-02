# coding=utf-8
""" Trainer class."""

import os
import logging
from collections import OrderedDict
import math
import itertools
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple

import numpy as np

from sklearn.metrics import classification_report

import tensorflow as tf
from transformers import WarmUp, AdamWeightDecay
from transformers import AutoConfig
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification, TFPreTrainedModel

from data_processors import DataProcessor, DataProcessorForSequenceClassification, DatasetInfo
from configuration_trainer import TrainerConfig

logger = logging.getLogger(__name__)


class TFTrainer(ABC):
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

        self.datasets: Dict[str, tf.data.Dataset] = {}
        self.processor: DataProcessor
        self.model_class: TFPreTrainedModel
        self.dataset_info: DatasetInfo

        if self.strategy_name == "mirrored":
            self.strategy = tf.distribute.MirroredStrategy()
        elif self.strategy_name == "onedevice":
            if len(tf.config.list_physical_devices('GPU')) >= 1:
                self.strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            else:
                self.strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        else:
            raise ValueError("The strategy {} does not exists.".format(self.strategy_name))

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
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.pretrained_model_name_or_path, cache_dir=model_cache_dir, use_fast=True)

        self._preprocess_data(data_cache_dir)
        self._config_trainer(model_cache_dir)

        with self.strategy.scope():
            self.model = self.model_class.from_pretrained(self.config.pretrained_model_name_or_path, config=self.model_config, cache_dir=model_cache_dir)
            self._create_optimizer()
            self._set_loss_and_metric()
            self._create_checkpoint_manager(checkpoint_path)
            self._create_summary_writer(log_path)

    def _config_trainer(self, model_cache_dir: Optional[str] = None) -> None:
        """
        This method set all the required fields for a specific task. For example
        in case of a classification set all the labels.
        Args:
          model_cache_dir (optional): the directory path where the pretrained model will be cached.
        """
        self.model_config = AutoConfig.from_pretrained(self.config.pretrained_model_name_or_path, cache_dir=model_cache_dir)

    def _set_loss_and_metric(self) -> None:
        """
        Create the training loss and metric with their name. Allowed names are those listed
        in the Tensorflow documentation and those contained in the transformers library.
        """
        try:
            self.loss = tf.keras.losses.get({"class_name": self.config.loss_name, "config": {"from_logits": True, "reduction": tf.keras.losses.Reduction.NONE}})
        except TypeError:
            self.loss = tf.keras.losses.get({"class_name": self.config.loss_name, "config": {"reduction": tf.keras.losses.Reduction.NONE}})

        self.train_acc_metric = tf.keras.metrics.get({"class_name": self.config.metric_name, "config": {"name": "train_accuracy"}})
        self.test_acc_metric = tf.keras.metrics.get({"class_name": self.config.metric_name, "config": {"name": "test_accuracy"}})

        self.test_loss_metric = tf.keras.metrics.Mean(name='test_loss')

    def _create_summary_writer(self, log_path: str) -> None:
        """
        Create a summary writer to be able to read the logs in Tensorboard.
        Args:
          log_path: the directory path where the Tensorboard logs will be saved.
        """
        self.log_path = log_path
        self.train_writer = tf.summary.create_file_writer(log_path + "/train")
        self.test_writer = tf.summary.create_file_writer(log_path + "/test")

    @abstractmethod
    def _create_features(self) -> None:
        """
        Create the features for the training and validation data.
        """
        pass

    @abstractmethod
    def _load_cache(self, cached_file: str) -> tf.data.Dataset:
        """
        Load a cached TFRecords dataset.
        Args:
          cached_file: the TFRecords file path.
        """
        pass

    @abstractmethod
    def _save_cache(self, mode: str, cached_file: str) -> None:
        """
        Save a cached TFRecords dataset.
        Args:
          mode: the dataset to be cached.
          cached_file: the file path where the TFRecords will be saved.
        """
        pass

    def _preprocess_data(self, cache_dir: str) -> None:
        """
        Preprocess the training and validation data.
        Args:
          cache_dir: the directory path where the cached data are / should be saved.
        """
        cached_training_features_file = os.path.join(
            cache_dir, "cached_train_{}_{}_{}.tf_record".format(
                self.config.task.replace("/", "-"), list(filter(None, self.config.pretrained_model_name_or_path.split("/"))).pop(), str(self.config.max_len)
            ),
        )
        cached_validation_features_file = os.path.join(
            cache_dir, "cached_validation_{}_{}_{}.tf_record".format(
                self.config.task.replace("/", "-"), list(filter(None, self.config.pretrained_model_name_or_path.split("/"))).pop(), str(self.config.max_len)
            ),
        )
        cached_test_features_file = os.path.join(
            cache_dir, "cached_test_{}_{}_{}.tf_record".format(
                self.config.task.replace("/", "-"), list(filter(None, self.config.pretrained_model_name_or_path.split("/"))).pop(), str(self.config.max_len)
            ),
        )

        if os.path.exists(cached_training_features_file) and os.path.exists(cached_validation_features_file):
            self.dataset_info = DatasetInfo.load(cache_dir)
            logger.info("Loading features from cached file %s", cached_training_features_file)
            self.datasets["train"] = self._load_cache(cached_training_features_file)
            logger.info("Loading features from cached file %s", cached_validation_features_file)
            self.datasets["validation"] = self._load_cache(cached_validation_features_file)
            logger.info("Loading features from cached file %s", cached_test_features_file)
            self.datasets["test"] = self._load_cache(cached_test_features_file)
        else:
            os.makedirs(cache_dir, exist_ok=True)
            self.processor.create_examples()
            self._create_features()
            self.dataset_info = self.processor.datasetinfo()
            logger.info("Create cache file %s", cached_training_features_file)
            self._save_cache("train", cached_training_features_file)
            logger.info("Create cache file %s", cached_validation_features_file)
            self._save_cache("validation", cached_validation_features_file)
            logger.info("Create cache file %s", cached_test_features_file)
            self._save_cache("test", cached_test_features_file)
            self.dataset_info.save(cache_dir)

        train_batch = self.config.train_batch_size * self.strategy.num_replicas_in_sync
        eval_batch = self.config.eval_batch_size * self.strategy.num_replicas_in_sync
        test_batch = self.config.eval_batch_size
        self.train_steps = math.ceil(self.dataset_info.sizes["train"] / train_batch)
        self.datasets["train"] = self.datasets["train"].shuffle(128).batch(train_batch).repeat(-1)
        self.datasets["train"] = self.strategy.experimental_distribute_dataset(self.datasets["train"])
        self.validation_steps = math.ceil(self.dataset_info.sizes["validation"] / eval_batch)
        self.datasets["validation"] = self.datasets["validation"].batch(eval_batch)
        self.datasets["validation"] = self.strategy.experimental_distribute_dataset(self.datasets["validation"])
        self.test_steps = math.ceil(self.dataset_info.sizes["test"] / test_batch)
        self.datasets["test"] = self.datasets["test"].batch(test_batch)

    def _create_optimizer(self) -> None:
        """
        Create the training optimizer with its name. Allowed names are those listed
        in the Tensorflow documentation and those contained in the transformers library.
        """
        if self.config.optimizer_name == "adamw":
            learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=self.config.learning_rate,
                                                                             decay_steps=self.train_steps,
                                                                             end_learning_rate=0.0)
            if self.config.warmup_steps:
                learning_rate_fn = WarmUp(initial_learning_rate=self.config.learning_rate, decay_schedule_fn=learning_rate_fn,
                                          warmup_steps=self.config.warmup_steps)

            self.optimizer = AdamWeightDecay(learning_rate=learning_rate_fn, weight_decay_rate=0.01, epsilon=self.config.adam_epsilon,
                                             exclude_from_weight_decay=["layer_norm", "bias"])
        else:
            try:
                self.optimizer = tf.keras.optimizers.get({"class_name": self.config.optimizer_name, "config" : {"learning_rate": self.config.learning_rate, "epsilon": self.config.adam_epsilon}})
            except TypeError:
                # This is for the case where the optimizer is not Adam-like such as SGD
                self.optimizer = tf.keras.optimizers.get({"class_name": self.config.optimizer_name, "config" : {"learning_rate": self.config.learning_rate}})

    def _create_checkpoint_manager(self, checkpoint_path: str, max_to_keep: int = 5, load_model: bool = True) -> None:
        """
        Create a checkpoint manager in order to be able to make the training
        fault-tolerant.
        Args:
          checkpoint_path: the directory path where the model checkpoints will be saved.
          max_to_keep: the maximum number of checkpoints to keep in the checkpoint path.
          load_model: if we want to start the training from the latest checkpoint.
        """
        ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.model.ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=max_to_keep)

        if load_model:
            ckpt.restore(self.model.ckpt_manager.latest_checkpoint)

    def _evaluate_during_training(self) -> None:
        """
        Evaluate the model during the training at the end of each epoch.
        """
        num_batches = 0
        test_step = 1

        for batch in self.datasets["validation"]:
            test_step = tf.convert_to_tensor(test_step, dtype=tf.int64)
            self._distributed_test_step(batch)
            num_batches += 1

            with self.test_writer.as_default():
                tf.summary.scalar("loss", self.test_loss_metric.result(), step=test_step)

            if test_step % self.validation_steps == 0:
                break

            test_step += 1

    def train(self) -> None:
        """
        Train method to train the model.
        """
        with self.strategy.scope():
            tf.summary.trace_on(graph=True, profiler=True)
            step = 1
            train_loss = 0.0

            for epoch in range(1, self.config.epochs + 1):
                total_loss = 0.0
                num_batches = 0

                for batch in self.datasets["train"]:
                    step = tf.convert_to_tensor(step, dtype=tf.int64)
                    total_loss += self._distributed_train_step(batch)
                    num_batches += 1

                    with self.train_writer.as_default():
                        tf.summary.scalar("loss", total_loss / num_batches, step=step)

                    if step == 1:
                        with self.train_writer.as_default():
                            tf.summary.trace_export(name="training", step=step, profiler_outdir=self.log_path)

                    if step % 10 == 0:
                        logger.info("Epoch {} Step {} Loss {:.4f} Train Accuracy {:.4f}".format(epoch, step.numpy(), total_loss.numpy() / num_batches, self.train_acc_metric.result()))

                    if step % 100 == 0:
                        ckpt_save_path = self.model.ckpt_manager.save()
                        logger.info("Saving checkpoint for step {} at {}".format(step, ckpt_save_path))

                    if step % self.train_steps == 0:
                        step += 1
                        break

                    step += 1

                train_loss = total_loss / num_batches

                self._evaluate_during_training()

                logger.info("Epoch {} Step {} Train Loss {:.4f} Train Accuracy {:.4f}".format(epoch, step.numpy() - 1, train_loss.numpy(), self.train_acc_metric.result()))
                logger.info("Epoch {} Validation Loss {:.4f} Validation Accuracy {:.4f}".format(epoch, self.test_loss_metric.result(), self.test_acc_metric.result()))

            if epoch != self.epochs:
                self.train_acc_metric.reset_states()
                self.test_acc_metric.reset_states()

    @abstractmethod
    def _distributed_test_step(self, dist_inputs: Tuple[Dict[str, tf.Tensor], tf.Tensor]) -> None:
        """
        Method that represents a custom test step in distributed mode
        Args:
          dist_inputs: the features batch of the test data
        """
        pass

    @abstractmethod
    def _distributed_train_step(self, dist_inputs: Tuple[Dict[str, tf.Tensor], tf.Tensor]) -> float:
        """
        Method that represents a custom training step in distributed mode.
        Args:
          dist_inputs: the features batch of the training data
        """
        pass

    @abstractmethod
    def evaluate(self) -> None:
        """
        Evaluate the model over the test dataset and print a report.
        """
        pass

    def save_model(self, save_path: str) -> None:
        """
        Save the pretrained model and create a Tensorflow saved model.
        Args:
          save_path: directory path where the pretrained model and
            Tensorflow saved model will be saved
        """
        logger.info("Saving model in {}".format(save_path))

        path = os.path.join(save_path, "saved_model")

        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        self.config.save_trainer(save_path)
        tf.saved_model.save(self.model, path)


class TFTrainerForSequenceClassification(TFTrainer):
    def __init__(self, config_path: str = None, config: TrainerConfig = None, **kwargs):
        super().__init__(config_path, config, **kwargs)
        self.processor = DataProcessorForSequenceClassification(**self.data_processor_config)
        self.model_class = TFAutoModelForSequenceClassification
        self.labels: List[str] = []

    def _create_features(self) -> None:
        self.datasets["train"] = self.processor.convert_examples_to_features("train", self.tokenizer, self.config.max_len, return_dataset="tf")
        self.datasets["validation"] = self.processor.convert_examples_to_features("validation", self.tokenizer, self.config.max_len, return_dataset="tf")
        self.datasets["test"] = self.processor.convert_examples_to_features("test", self.tokenizer, self.config.max_len, return_dataset="tf")

        if self.datasets["test"] is None:
            self.datasets["test"] = self.datasets["validation"]

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

    def _load_cache(self, cached_file: str) -> tf.data.Dataset:
        name_to_features = {
            "input_ids": tf.io.FixedLenFeature([self.config.max_len], tf.int64),
            "attention_mask": tf.io.FixedLenFeature([self.config.max_len], tf.int64),
            "token_type_ids": tf.io.FixedLenFeature([self.config.max_len], tf.int64),
            "label": tf.io.FixedLenFeature([1], tf.int64),
        }

        def _decode_record(record):
            example = tf.io.parse_single_example(record, name_to_features)

            return {k : example[k] for k in ('input_ids', 'attention_mask', 'token_type_ids') if k in example}, example["label"]

        d = tf.data.TFRecordDataset(cached_file)
        d = d.map(_decode_record, num_parallel_calls=4)

        return d

    def _save_cache(self, mode: str, cached_file: str) -> None:
        writer = tf.io.TFRecordWriter(cached_file)
        ds = self.datasets[mode].enumerate()

        # as_numpy_iterator() is available since TF 2.1
        for (index, (feature, label)) in ds.as_numpy_iterator():
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

    @tf.function
    def _distributed_test_step(self, dist_inputs: Tuple[Dict[str, tf.Tensor], tf.Tensor]) -> None:
        def step_fn(inputs):
            features, labels = inputs
            logits = self.model(features, training=False)
            loss = self.loss(labels, logits[0]) + sum(self.model.losses)

            self.test_acc_metric(labels, logits[0])
            self.test_loss_metric(loss)

        self.strategy.experimental_run_v2(step_fn, args=(dist_inputs,))

    @tf.function
    def _distributed_train_step(self, dist_inputs: Tuple[Dict[str, tf.Tensor], tf.Tensor]) -> float:
        def step_fn(inputs):
            features, labels = inputs

            with tf.GradientTape() as tape:
                logits = self.model(features, training=True)
                per_example_loss = self.loss(labels, logits[0])
                loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.config.train_batch_size)

            gradients = tape.gradient(loss, self.model.trainable_variables)

            if self.config.optimizer_name == "adamw":
                self.optimizer.apply_gradients(list(zip(gradients, self.model.trainable_variables)), self.config.max_grad_norm)
            else:
                gradients = [(tf.clip_by_value(grad, -self.config.max_grad_norm, self.config.max_grad_norm)) for grad in gradients]
                self.optimizer.apply_gradients(list(zip(gradients, self.model.trainable_variables)))

            self.train_acc_metric(labels, logits[0])

            return loss

        per_replica_losses = self.strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
        sum_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        return sum_loss

    def evaluate(self) -> None:
        y_true = []
        results = self.model.predict(self.datasets["test"], steps=self.test_steps)

        for batch in self.datasets["test"]:
            y_true.extend(batch[1].numpy().tolist())

        y_pred = np.reshape(np.argmax(results, axis=-1), (-1, 1)).tolist()
        y_true = list(itertools.chain.from_iterable(y_true))
        y_pred = list(itertools.chain.from_iterable(y_pred))

        logger.info(classification_report(y_true, y_pred, target_names=self.label2id.keys()))
