# tf-trainer-proposal
Tensorflow trainer proposal for the Hugging Face transformers library.

## TL;DR
This is the Tensorflow version of @julien-c [trainer proposal](https://github.com/julien-c/trainer-proposal). The guidelines are written in a [shared document](https://docs.google.com/document/d/1WAR9uuOZpu7T_6TNPHrxIvrNbrUs7Ggy3WjjDwjayzA/edit#heading=h.tzvypnpe7axy).

## Architecture
### The trainer
The trainer itself is an abstract class that will contain methods that can be shared across all the subclasses. Each subclass represents a specific task, here for example we have `TFTrainerForSequenceClassification`. Then, we can also do subclasses for QA or Token classification. The trainer is configurable with dictionary that contains multiple parameters and some parameters will be either specific for the abstract class or for the subclasses. General parameters for a trainer are:

* `pretrained_model_name_or_path`: path/name of the pretrained model.
* `optimizer_name`: name of the optimizer we want to use. All these available in Tensorflow's Keras [list](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers?hl=en) including those available in the transformers lib (for now only `AdamW`).
* `warmup_steps`: the number of warmup steps for the `AdamW` scheduler. 
* `decay_steps`: the number of decay steps for the `AdamW` scheduler.
* `learning_rate`: learning rate value used by the optimizer. 
* `adam_epsilon`: epsilon value used by the Adam-like optimizers and RMSProp. (optional)
* `loss_name`: name of the loss we want to use. All these available in Tensorflow's Keras [list](https://www.tensorflow.org/api_docs/python/tf/keras/losses?hl=en) including those available in the transformers lib (for now none)
* `train_batch_size`: size of the batches used for the training.
* `eval_batch_size`: size of the batches used for the evaluation.
* `distributed`: if we want to run the training on multiple GPUs (on the same host) or not.
* `epochs`: number of epochs the training has to run.
* `max_grad_norm`: max gradient norm. (optional)
* `metric_name`: name of the metric we want to use. All these available in Tensorflow's Keras [list](https://www.tensorflow.org/api_docs/python/tf/keras/metrics?hl=en) including those available in the transformers lib (for now none)
* `max_len`: length of the sequences.
* `task`: name of the task. Can be anything it is mostly to make the difference between multiple run of the same train.

### The data processors
The logic behind the data processors are basically the same than for the trainer. One abstract class that will share its methods to all the subclasses and each subclass represents a specific data processor for a task, here for example we have `DataProcessorForSequenceClassification`. The role of this class is to translate CSV files (or derivated such as TSV or PSV) into Tensorflow datasets with the `tensorflow_datasets` [API](https://www.tensorflow.org/datasets/api_docs/python/tfds). The same data processor is also able to use the textual datasets contained in the [catalog](https://www.tensorflow.org/datasets/catalog/overview#text). General parameters for a data processor are:

* `train_file`: the location of the training CSV file. (not used if using a dataset from the tensoflow datasets catalog)
* `dev_file`: the location of the validation CSV file. (not used if using a dataset from the tensoflow datasets catalog)
* `test_file`: the location of the test CSV file (optional). If not given will take the validation dataset as test. (not used if using a dataset from the tensoflow datasets catalog)

Now the `DataProcessorForSequenceClassification` subclass needs its own parameters that are:

* `guid`: feature name representing the id of the examples in a tensorflow dataset (not used if using CSV files or if not necessary)
* `text_a`: feature name representing the first sentence of the examples in a tensorflow dataset (not used if using CSV files)
* `text_b`: feature name representing the second sentence of the examples in a tensorflow dataset (not used if using CSV files or if not necessary) 
* `label`: feature name representing the label of the examples in a tensorflow dataset (not used if using CSV files)
* `dataset_name`: name of the dataset. Takes either the name of a dataset from the Tensorflow datasets catalog or a custom name that will use the CSV files.
* `is_column_id`: tells if the first column of the CSVs represents the IDs or not (not used if using Tensorflow datasets)
* `skip_first_row`: tells if we have to skip the first row of the CSVs or not (not used if using Tensorflow datasets)
* `delimiter`: the delimiter used in the CSVs (not used if using Tensorflow datasets)
* `quotechar`: the quotechar used in the CSVs (not used if using Tensorflow datasets)

## Installation and usage
The requirements to run the examples are:
```
pip install tensorflow tensorflow_datasets transformers
```

To run the text classification example that uses CSV files:
```
python text_classification.py
```
The used dataset for this example is the [BBC articles fulltext and category](https://www.kaggle.com/yufengdev/bbc-fulltext-and-category).

To run the GLUE Mrpc example that uses the GLUE dataset from the tensorflow datasets catalog:
```
python glue.py
```
