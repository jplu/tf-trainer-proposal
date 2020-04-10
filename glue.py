import logging

logging.basicConfig(level=logging.INFO)

from trainer import TrainerForSequenceClassification

trainer = TrainerForSequenceClassification(config_path="conf/glue_mrpc/trainer_config.json")
trainer.setup_training(data_cache_dir="data_cache")
