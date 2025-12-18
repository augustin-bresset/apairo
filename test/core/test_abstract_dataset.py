import unittest
from test.utils.create_mock_dataset import create_mock_dataset


class TestAbstractDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = create_mock_dataset()

    def test_keys_setter(self):
        self.assertEqual(self.dataset.keys, ["key"])
        with self.assertRaises(Exception):
            self.dataset.keys = []
        with self.assertRaises(Exception):
            self.dataset.keys = ["key_a", "key_a"]
        self.dataset.keys = ["key_a", "key_b"]
        self.assertEqual(self.dataset.keys, ["key_a", "key_b"])
        self.dataset.keys = ["key"]

    def test_load(self):
        self.assertEqual(self.dataset.load("key", 0), "value")

    def test_len(self):
        self.assertEqual(len(self.dataset), 1)

    def test_shape(self):
        self.assertEqual(self.dataset.shape, (1,))

    def test_iter(self):
        self.assertEqual(next(iter(self.dataset)), {"key": ["value"]})

    

if __name__ == '__main__':
    unittest.main()