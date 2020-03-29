import logging

logging.basicConfig(level=logging.INFO)

from trainer_tf import TFTrainerForSequenceClassification

# 12 epochs normalement
config = {
    "model_config": {
        "pretrained_model_name_or_path": "bert-base-cased",
        "optimizer_name": "adam",
        "learning_rate": 5e-5,
        "loss_name": "SparseCategoricalCrossentropy",
        "train_batch_size": 8,
        "eval_batch_size": 8,
        "distributed": False,
        "epochs": 4,
        "metric_name": "SparseCategoricalAccuracy",
        "max_len": 512,
        "task": "bbc-classifier"
    },
    "data_processor_config": {
        "train_file": "data/train.csv",
        "dev_file": "data/test.csv",
        "skip_first_row": True,
        "delimiter": ",",
        "dataset_name": "bbc-classifier",
    }
}

trainer = TFTrainerForSequenceClassification(**config)
trainer.setup_training(data_cache_dir="data_cache")
trainer.train()
trainer.save_model("save")
trainer.evaluate()
