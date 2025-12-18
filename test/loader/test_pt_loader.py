import unittest
import os
from src.loader import PTLoader
from src.core.utils.exceptions import EmptyLoaderWarning

from test.paths import tmp_path
from test.utils import create_random_pt_file

class TestPTLoader(unittest.TestCase):
    file_name = "test.pt"
    dir_name = "pt_loader_test"

    
    def setUp(self) -> None:
        self.keys = ["data_a", "data_b"]
        self.len_ = 10
        self.data_shape = [(7), (3, 3)]
        self.shapes = {key: self.data_shape[i] for i,key in enumerate(self.keys)}
        create_random_pt_file(self.keys, self.len_, self.data_shape, self.file_name, self.dir_name)
        self.loader = PTLoader(os.path.join(tmp_path, self.dir_name, self.file_name))

    def test_len(self):
        self.assertEqual(len(self.loader), self.len_)

    def test_getitem(self):
        self.assertIsNotNone(self.loader[0])

    def test_shape(self):
        for key in self.loader.data.keys():
            if key == "dt": continue
            if isinstance(self.shapes[key], int):
                self.assertEqual(int(self.loader.shape[key][0]), self.shapes[key])
            else:
                self.assertTupleEqual(tuple(self.loader.shape[key]), tuple(self.shapes[key]))

    def test_set_keys(self):
        self.loader.set_keys(["data_a"])
        self.assertEqual(self.loader.data.keys(), {"data_a", "dt"})
        
        self.assertRaises(EmptyLoaderWarning, self.loader.set_keys, ["data_c"])
        self.loader = PTLoader(os.path.join(tmp_path, self.dir_name, self.file_name))

    def test_reset(self):
        
        self.loader.set_keys(["data_a"])
        self.loader.reset()
        self.assertEqual(self.loader.data.keys(), {"data_a", "data_b", "dt"})

    def tearDown(self):
        os.remove(os.path.join(tmp_path, self.dir_name, self.file_name))
        os.rmdir(os.path.join(tmp_path, self.dir_name))

if __name__ == "__main__":
    unittest.main()