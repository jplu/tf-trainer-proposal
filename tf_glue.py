import logging

logging.basicConfig(level=logging.INFO)

from trainer_tf import TFTrainerForSequenceClassification

trainer = TFTrainerForSequenceClassification(config_path="conf/glue_mrpc")
trainer.setup_training(data_cache_dir="data_cache")
trainer.train()
trainer.save_model("save")
trainer.evaluate()
