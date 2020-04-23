import logging
from dataclasses import dataclass, field

from data_processors import DataProcessorForSequenceClassification
from tf_training_args import TFTrainingArguments
from trainer_tf_barebones import TFTrainer
from transformers import AutoConfig, AutoTokenizer
from transformers import TFAutoModelForSequenceClassification


logging.basicConfig(level=logging.INFO)



@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


@dataclass
class DataTrainingArguments:
    task_name: str = field(metadata={"help": "The name of the task to train on"})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )


# TODO(Parse from nested JSON file if necessary)

model_args = ModelArguments(model_name_or_path="bert-base-cased")
data_args = DataTrainingArguments(task_name="glue/mrpc", data_dir=".", max_seq_length=128)
training_args = TFTrainingArguments(output_dir="./save")



tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)


# This DataProcessorForSequenceClassification will not be needed in the future b/c Thomwolf's
# datasets will do that.
processor = DataProcessorForSequenceClassification(
    **vars(data_args), 
    dataset_name="glue/mrpc",
    format="tfds",
    guid="idx",
    text_a="sentence1",
    text_b="sentence2",
    label="label",
)

datasets, dataset_info = processor.preprocess_data(args=data_args, tokenizer=tokenizer)

label2id = {label: i for i, label in enumerate(dataset_info.labels)}
id2label = {i: label for i, label in enumerate(dataset_info.labels)}

config = AutoConfig.from_pretrained(
    model_args.model_name_or_path,
    finetuning_task=data_args.task_name,
    num_labels=len(dataset_info.labels),
    id2label=id2label,
    label2id=label2id,
)

model = TFAutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config)

trainer = TFTrainer(
    model=model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["validation"],
    test_dataset=datasets["test"],
    dataset_info=dataset_info,
)
trainer.train()
trainer.save_model("./save")
trainer.test()

print()
