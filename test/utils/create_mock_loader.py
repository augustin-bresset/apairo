from src.core import AbstractLoader
from test.paths import tmp_path

class MockLoader(AbstractLoader):
    def __init__(self, *args, **kwargs):
        self.data = {"key": "value"}

    def load(self):
        return self.data
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.data
    
    @property
    def shape(self):
        return (1,)

def create_mock_loader():
    return MockLoader()


