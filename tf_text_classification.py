import logging

logging.basicConfig(level=logging.INFO)

import logging

logging.basicConfig(level=logging.INFO)

from trainer_tf import TFTrainerForSequenceClassification


trainer = TFTrainerForSequenceClassification(config_path="conf/bbc-news")
trainer.setup_training(data_cache_dir="data_cache")
trainer.train()
trainer.save_model("save")
trainer.evaluate()
