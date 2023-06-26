class BaseDataLoader(object):
    def __init__(self, base_path):
        self.basePath = base_path

    def sample_dataset_from_test(self):
        raise NotImplementedError("You need to implement testDataset")

    def get_dataset(self):
        raise NotImplementedError("You need to implement trainDataset")

    def sample_dataset(self):
        return self.get_dataset()
