import unittest
import torch

from data_handler import DataHandler

class TestDataHandler(unittest.TestCase):
    def setUp(self):
        self.handler_cqa = DataHandler("commonsense_qa", batch_size=2, max_length=64)
        self.handler_obqa = DataHandler("openbookqa", batch_size=2, max_length=64)

    def test_commonsenseqa_loader(self):
        train_loader, val_loader = self.handler_cqa.load_dataset()
        batch = next(iter(train_loader))

        self.assertIn('input_ids', batch)
        self.assertIn('attention_mask', batch)
        self.assertIn('labels', batch)

        # Check shapes
        self.assertEqual(batch['input_ids'].shape, (2, 64))
        self.assertEqual(batch['labels'].shape, (2, 64))

        # Ensure label values are either >= 0 or == -100
        # (GPT2 vocab IDs are >= 0; ignoring -100).
        self.assertTrue(((batch['labels'] >= 0) | (batch['labels'] == -100)).all())

        # Check if at least one token is -100 per sample (the prompt portion).
        # Convert to CPU if needed:
        labels_cpu = batch['labels'].cpu()
        for row in labels_cpu:
            self.assertTrue((-100 in row), "Expected some -100 tokens in label row")

    def test_openbookqa_loader(self):
        train_loader, val_loader = self.handler_obqa.load_dataset()
        batch = next(iter(train_loader))

        self.assertIn('input_ids', batch)
        self.assertIn('attention_mask', batch)
        self.assertIn('labels', batch)

        self.assertEqual(batch['input_ids'].shape, (2, 64))
        self.assertEqual(batch['labels'].shape, (2, 64))
        self.assertTrue(((batch['labels'] >= 0) | (batch['labels'] == -100)).all())

        # Check if at least one token is -100
        labels_cpu = batch['labels'].cpu()
        for row in labels_cpu:
            self.assertTrue((-100 in row))

    def test_invalid_dataset(self):
        with self.assertRaises(ValueError):
            DataHandler("invalid_dataset").load_dataset()


if __name__ == '__main__':
    unittest.main()
