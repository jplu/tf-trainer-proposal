# coding=utf-8
"Trainer configuration class"

import copy
import json
import logging
import os
from typing import Dict, Tuple


logger = logging.getLogger(__name__)

TRAINER_CONFIG_NAME = "trainer_config.json"


class TrainerConfig(object):
    def __init__(self, **kwargs):
        self.pretrained_model_name_or_path: str = kwargs.pop("pretrained_model_name_or_path", None)
        self.optimizer_name: str = kwargs.pop("optimizer_name", None)
        self.warmup_steps: int = kwargs.pop("warmup_steps", 0.0)
        self.learning_rate: float = kwargs.pop("learning_rate", None)
        self.adam_epsilon: float = kwargs.pop("adam_epsilon", 1e-08)
        self.loss_name: str = kwargs.pop("loss_name", None)
        self.train_batch_size: int = kwargs.pop("train_batch_size", None)
        self.eval_batch_size: int = kwargs.pop("eval_batch_size", None)
        self.epochs: int = kwargs.pop("epochs", None)
        self.max_grad_norm: float = kwargs.pop("max_grad_norm", 1.0)
        self.metric_name: str = kwargs.pop("metric_name", None)
        self.task: str = kwargs.pop("task", None)
        self.mode: str = kwargs.pop("mode", None)

    def save_trainer(self, save_directory):
        """
        Save a configuration object to the directory `save_directory`, so that it
        can be re-loaded using the :func:`~transformers.TrainerConfig.from_trainer` class method.

        Args:
            save_directory (:obj:`string`):
                Directory where the configuration JSON file will be saved.
        """
        assert os.path.isdir(
            save_directory
        ), "Saving path should be a directory where the model and configuration can be saved"

        output_config_file = os.path.join(save_directory, TRAINER_CONFIG_NAME)

        self.to_json_file(output_config_file)
        logger.info("Configuration saved in {}".format(output_config_file))

    @classmethod
    def from_trainer(cls, trainer_path, **kwargs) -> "TrainerConfig":
        r"""

        Instantiate a :class:`~transformers.TrainerConfig` (or a derived class) from a pre-trained model configuration.

        Args:
            trainer_path (:obj:`string`):
                either:
                  - a path to a `directory` containing a configuration file saved using the
                    :func:`~transformers.TrainerConfig.save_trainer` method, e.g.: ``./my_trainer_directory/``.
                  - a path or url to a saved configuration JSON `file`, e.g.:
                    ``./my_trainer_directory/configuration.json``.
            kwargs (:obj:`Dict[str, any]`, `optional`):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is
                controlled by the `return_unused_kwargs` keyword parameter.
            return_unused_kwargs: (`optional`) bool:
                If False, then this function returns just the final configuration object.
                If True, then this functions returns a :obj:`Tuple(config, unused_kwargs)` where `unused_kwargs` is a
                dictionary consisting of the key/value pairs whose keys are not configuration attributes: ie the part
                of kwargs which has not been used to update `config` and is otherwise ignored.

        Returns:
            :class:`TrainerConfig`: An instance of a configuration object

        Examples::
            config = TrainerConfig.from_trainer('./my_trainer_directory/')  # E.g. config was saved using `save_trainer('./my_trainer_directory/')`
            config = TrainerConfig.from_trainer('./my_trainer_directory/my_configuration.json')
            config = TrainerConfig.from_trainer('./my_trainer_directory/', optimizer_name='adam', foo=False)
            assert config.optimizer_name == 'adam'
            config, unused_kwargs = TrainerConfig.from_trainer('./my_trainer_directory/', optimizer_name='adam',
                                                               foo=False, return_unused_kwargs=True)
            assert config.optimizer_name == 'adam'
            assert unused_kwargs == {'foo': False}
        """
        config_dict, kwargs = cls.get_config_dict(trainer_path, **kwargs)
        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def get_config_dict(cls, trainer_path: str, **kwargs) -> Tuple[Dict, Dict]:
        """
        From a `trainer_path`, resolve to a dictionary of parameters, to be used
        for instantiating a Config using `from_dict`.

        Parameters:
            trainer_path (:obj:`string`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            :obj:`Tuple[Dict, Dict]`: The dictionary that will be used to instantiate the configuration object.

        """
        if not os.path.exists(trainer_path):
            msg = ("Can't load '{}'. Make sure that:\n\n"
                   "'{}' is the correct path to a directory containing a '{}' file\n\n".format(
                       trainer_path,
                       trainer_path,
                       TRAINER_CONFIG_NAME,
                   )
                   )
            raise EnvironmentError(msg)

        config_file = trainer_path

        if os.path.isdir(trainer_path):
            config_file = os.path.join(trainer_path, TRAINER_CONFIG_NAME)

        try:
            config_dict = cls._dict_from_json_file(config_file)
        except json.JSONDecodeError:
            msg = (
                "Configuration file is not a valid JSON file. "
                "Please check file content here: {}.".format(config_file)
            )
            raise EnvironmentError(msg)

        logger.info("loading configuration file {}".format(config_file))

        return config_dict, kwargs

    @classmethod
    def from_dict(cls, config_dict: Dict, **kwargs) -> "TrainerConfig":
        """
        Constructs a `Config` from a Python dictionary of parameters.

        Args:
            config_dict (:obj:`Dict[str, any]`):
                Dictionary that will be used to instantiate the configuration object. Such a dictionary can be retrieved
                from a pre-trained checkpoint by leveraging the :func:`~transformers.TrainerConfig.get_config_dict`
                method.
            kwargs (:obj:`Dict[str, any]`):
                Additional parameters from which to initialize the configuration object.

        Returns:
            :class:`TrainerConfig`: An instance of a configuration object
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        return_unused_config = kwargs.pop("return_unused_config", False)

        config = cls(**config_dict)

        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        if return_unused_config:
            for key, value in config_dict.items():
                if not hasattr(config, key):
                    kwargs[key] = value

        logger.info("Model config %s", str(config))
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_json_file(cls, json_file: str) -> "TrainerConfig":
        """
        Constructs a `Config` from the path to a json file of parameters.

        Args:
            json_file (:obj:`string`):
                Path to the JSON file containing the parameters.

        Returns:
            :class:`TrainerConfig`: An instance of a configuration object

        """
        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file: str):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return "{} {}".format(self.__class__.__name__, self.to_json_string())

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        if hasattr(self.__class__, "model_type"):
            output["model_type"] = self.__class__.model_type
        return output

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.

        Returns:
            :obj:`string`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """
        Save this instance to a json file.

        Args:
            json_file_path (:obj:`string`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())
