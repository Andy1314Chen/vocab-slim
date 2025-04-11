import unittest

from vocabslim import VocabSlim


class TestVocabSlim(unittest.TestCase):
    def setUp(self):
        self.model_path = "Qwen/Qwen2.5-0.5B"
        self.save_path = "test_output"

    def test_initialization(self):
        slim = VocabSlim(
            self.model_path,
            self.save_path,
            dataset_config={
                "name": "wikitext",
                "config": "wikitext-103-raw-v1",
                "split": "train",
                "text_column": "text"
            },
            target_vocab_size=1000
        )
        self.assertIsNotNone(slim)

    def test_prune(self):
        pass
