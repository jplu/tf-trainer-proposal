from dataclasses import dataclass, field

from transformers import TrainingArguments


# I am not sure, but maybe this shouldn't even be a child subclass,
# but rather update TrainingArguments with some additional TF helpers.
# Not sure about this...

@dataclass
class TFTrainingArguments(TrainingArguments):
    strategy_name: str = field(default="onedevice")
    optimizer_name: str = field(default="adam")
    mode: str = field(default="classification")
    loss_name: str = field(default="SparseCategoricalCrossentropy")
    metric_name: str = field(default="SparseCategoricalAccuracy")
