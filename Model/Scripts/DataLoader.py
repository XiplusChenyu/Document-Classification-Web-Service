import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
from Settings import Settings


class TorchData(Dataset):

    def __init__(self, dataset_path):
        """
        Take the h5py dataset
        """
        super(TorchData, self).__init__()
        self.dataset = h5py.File(dataset_path, 'r')
        self.label = self.dataset['label']
        self.words = self.dataset['chunk']

        self.len = self.words.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        words = self.words[index].astype(np.float32)
        words = torch.from_numpy(words)
        label = torch.from_numpy(self.label[index].astype(np.float32))
        sample = {"words": words, "label": label}
        return sample


# define the data loaders
def torch_dataset_loader(dataset, batch_size, shuffle, kwargs):
    """
    take the h5py dataset
    """
    loader = DataLoader(TorchData(dataset),
                        batch_size=batch_size,
                        shuffle=shuffle,
                        **kwargs)
    return loader


if __name__ == '__main__':
    train_loader = torch_dataset_loader(Settings.train_path, Settings.batch_size, True, Settings.kwargs)
    validation_loader = torch_dataset_loader(Settings.valid_path, Settings.batch_size, False, Settings.kwargs)
    test_loader = torch_dataset_loader(Settings.test_path, Settings.batch_size, False, Settings.kwargs)

    for index, data_item in enumerate(train_loader):
        print(data_item['words'].shape)
        print(data_item['label'].shape)
        break
