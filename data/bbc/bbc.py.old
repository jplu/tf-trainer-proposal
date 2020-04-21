import csv

import nlp


class Bbc(nlp.GeneratorBasedBuilder):
    VERSION = nlp.Version("1.0.0")

    def __init__(self, **config):
        self.train = config.pop("train", None)
        self.validation = config.pop("validation", None)
        super(Bbc, self).__init__(**config)

    def _info(self):
        return nlp.DatasetInfo(builder=self, description="bla", features=nlp.features.FeaturesDict({"id": nlp.int32, "text": nlp.string, "label": nlp.string}))

    def _split_generators(self, dl_manager):
        return [nlp.SplitGenerator(name=nlp.Split.TRAIN, gen_kwargs={"filepath": self.train}),
                nlp.SplitGenerator(name=nlp.Split.VALIDATION, gen_kwargs={"filepath": self.validation}),
                nlp.SplitGenerator(name=nlp.Split.TEST, gen_kwargs={"filepath": None})]

    def _generate_examples(self, filepath):
        if not filepath:
            return None, {}

        with open(filepath) as f:
            reader = csv.reader(f, delimiter=',', quotechar="\"")
            lines = list(reader)[1:]

            for idx, line in enumerate(lines):
                yield idx, {"id": idx, "text": line[1], "label": line[0]}
