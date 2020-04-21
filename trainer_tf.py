# coding=utf-8
"""Tensorflow trainer class."""

import os
import logging
import math
import itertools
from typing import Dict, Optional

import numpy as np

from sklearn.metrics import classification_report

import tensorflow as tf
from transformers import WarmUp, AdamWeightDecay, GradientAccumulator
from transformers import AutoConfig
from transformers import TFAutoModelForSequenceClassification, TFAutoModelForTokenClassification

from data_processors import DataProcessorForSequenceClassification, DataProcessorForTokenClassification, DatasetInfo
from configuration_trainer import TrainerConfig

logger = logging.getLogger(__name__)


class TFTrainer():
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
        self.dataset_info: DatasetInfo
        self.gradient_accumulator = GradientAccumulator()
        self.accum_steps = 1

        if self.config.mode == "classification":
            self.processor = DataProcessorForSequenceClassification(**self.data_processor_config)
            self.model_class = TFAutoModelForSequenceClassification
        elif self.config.mode == "labelling":
            self.processor = DataProcessorForTokenClassification(**self.data_processor_config)
            self.model_class = TFAutoModelForTokenClassification

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
        self._prepare_dataset(data_cache_dir, model_cache_dir)
        self._config_trainer(model_cache_dir)

        with self.strategy.scope():
            self.model = self.model_class.from_pretrained(self.config.pretrained_model_name_or_path, config=self.model_config, cache_dir=model_cache_dir)
            self._create_optimizer()
            _ = self.optimizer.iterations
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
        if self.config.mode == "classification":
            label2id = {label: i for i, label in enumerate(self.dataset_info.labels)}
            id2label = {i: label for i, label in enumerate(self.dataset_info.labels)}
            self.model_config = AutoConfig.from_pretrained(self.config.pretrained_model_name_or_path, num_labels=len(self.dataset_info.labels), id2label=id2label, label2id=label2id, cache_dir=model_cache_dir)
        else:
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

    def _create_summary_writer(self, log_path: str) -> None:
        """
        Create a summary writer to be able to read the logs in Tensorboard.
        Args:
          log_path: the directory path where the Tensorboard logs will be saved.
        """
        self.log_path = log_path
        self.train_writer = tf.summary.create_file_writer(log_path + "/train")
        self.test_writer = tf.summary.create_file_writer(log_path + "/test")

    def _prepare_dataset(self, data_cache_dir: str, model_cache_dir: str) -> None:
        """
        Prepare the training, validation and test data.
        Args:
          data_cache_dir: the directory path where the cached data are / should be saved.
        """
        self.datasets, self.dataset_info, self.tokenizer = self.processor.preprocess_data(data_cache_dir, model_cache_dir, self.config.task, self.config.pretrained_model_name_or_path)
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

    def _evaluate(self, dataset) -> None:
        """
        Evaluate the model during the training at the end of each epoch.
        """
        step = 1
        loss = 0.0

        for features, labels in dataset:
            step = tf.convert_to_tensor(step, dtype=tf.int64)
            loss = self._run_model(features, labels, False)

            with self.test_writer.as_default():
                tf.summary.scalar("loss", loss, step=step)

            if step % self.validation_steps == 0:
                break

            step += 1

        return loss

    def train(self) -> None:
        """
        Train method to train the model.
        """
        tf.summary.trace_on(graph=True, profiler=True)
        self.gradient_accumulator.reset()

        iterations = self.optimizer.iterations
        tf.summary.experimental.set_step(iterations)

        for epoch in range(1, self.config.epochs + 1):
            for training_loss in self._training_steps(self.datasets["train"]):
                step = iterations.numpy()

                with self.train_writer.as_default():
                    tf.summary.scalar("loss", training_loss, step=step)

                if step == 1:
                    with self.train_writer.as_default():
                        tf.summary.trace_export(name="training", step=step, profiler_outdir=self.log_path)

                if step % 10 == 0:
                    logger.info("Epoch {} Step {} Loss {:.4f} Train Accuracy {:.4f}".format(epoch, step, training_loss.numpy(), self.train_acc_metric.result()))

                if step % 100 == 0:
                    ckpt_save_path = self.model.ckpt_manager.save()
                    logger.info("Saving checkpoint for step {} at {}".format(step, ckpt_save_path))

                if step % self.train_steps == 0:
                    break

            test_loss = self._evaluate(self.datasets["validation"])

            logger.info("Epoch {} Step {} Train Loss {:.4f} Train Accuracy {:.4f}".format(epoch, step, training_loss.numpy(), self.train_acc_metric.result()))
            logger.info("Epoch {} Validation Loss {:.4f} Validation Accuracy {:.4f}".format(epoch, test_loss.numpy(), self.test_acc_metric.result()))

        if epoch != self.epochs:
            self.train_acc_metric.reset_states()
            self.test_acc_metric.reset_states()

    def _training_steps(self, dataset):
        """
        Returns a generator over training steps (i.e. parameters update).
        Args:
          dataset: The training dataset.
        Returns:
          A generator that yields a loss value to report for this step.
        """
        for i, loss in enumerate(self._accumulate_next_gradients(dataset)):
            if i % self.accum_steps == 0:
                self._apply_gradients()
                yield loss

    @tf.function
    def _apply_gradients(self):
        """Applies the gradients (cross-replica)."""
        self.strategy.experimental_run_v2(self._step)

    def _step(self):
        """Applies gradients and resets accumulation."""
        gradient_scale = self.gradient_accumulator.step * self.strategy.num_replicas_in_sync
        gradients = [gradient / tf.cast(gradient_scale, gradient.dtype) for gradient in self.gradient_accumulator.gradients]
        gradients = [(tf.clip_by_value(grad, -self.config.max_grad_norm, self.config.max_grad_norm)) for grad in gradients]
        self.optimizer.apply_gradients(list(zip(gradients, self.model.trainable_variables)))
        self.gradient_accumulator.reset()

    def _accumulate_next_gradients(self, dataset):
        """Accumulates the gradients from the next element in dataset."""
        iterator = iter(dataset)

        @tf.function
        def _accumulate_next():
            per_replica_features, per_replica_labels = next(iterator)

            return self._accumulate_gradients(per_replica_features, per_replica_labels)

        while True:
            try:
                yield _accumulate_next()
            except tf.errors.OutOfRangeError:
                break

    def _accumulate_gradients(self, per_replica_features, per_replica_labels):
        """Accumulates the gradients across all the replica."""
        per_replica_loss = self.strategy.experimental_run_v2(self._forward, args=(per_replica_features, per_replica_labels))

        return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)

    def _forward(self, features, labels):
        """Forwards a training example and accumulates the gradients."""
        per_example_loss = self._run_model(features, labels, True)
        loss = tf.nn.compute_average_loss(per_example_loss, global_batch_size=self.config.train_batch_size)
        
        gradients = self.optimizer.get_gradients(loss, self.model.trainable_variables)

        self.gradient_accumulator(gradients)

        return loss

    def _run_model(self, features, labels, training):
        """
        Computes the loss of the given features and labels pair.
        Args:
          features: the batched features.
          labels: the batched labels.
        """
        if self.config.mode == "classification" or self.config.mode == "labelling":
            logits = self.model(features, training=training)[0]
        else:
            logits = self.mode(features, training=training)
        """
        if self.config.mode == "labelling":
            active_loss = tf.reshape(labels, (-1,)) != -1
            logits = tf.boolean_mask(tf.reshape(logits, (-1, len(labels))), active_loss)
            labels = tf.boolean_mask(tf.reshape(labels, (-1,)), active_loss)
        """
        loss = self.loss(labels, logits)

        if training:
            self.train_acc_metric(labels, logits)
        else:
            self.test_acc_metric(labels, logits)

        return loss

    def test(self) -> None:
        """
        Test the model over the test dataset and print a report.
        """
        y_true = []
        results = self.model.predict(self.datasets["test"], steps=self.test_steps)

        if self.config.mode == "classification":
            for batch in self.datasets["test"]:
                y_true.extend(batch[1].numpy().tolist())

            y_pred = np.reshape(np.argmax(results, axis=-1), (-1, 1)).tolist()
            y_true = list(itertools.chain.from_iterable(y_true))
            y_pred = list(itertools.chain.from_iterable(y_pred))

            logger.info(classification_report(y_true, y_pred, target_names=self.dataset_info.labels))

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
