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

    def __getitem__(self, idx):
        new_words = self.words[idx].astype(int)
        new_words = torch.from_numpy(new_words)
        label = torch.from_numpy(self.label[idx].astype(np.float32))
        sample = {"words": new_words, "label": label}
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


train_loader = torch_dataset_loader(Settings.train_path, Settings.batch_size, True, Settings.kwargs)
validation_loader = torch_dataset_loader(Settings.valid_path, Settings.batch_size, False, Settings.kwargs)
test_loader = torch_dataset_loader(Settings.test_path, Settings.batch_size, False, Settings.kwargs)

if __name__ == '__main__':

    for index, data_item in enumerate(train_loader):
        words = data_item['words']
        print(data_item['words'].shape)
        print(data_item['label'].shape)
        break
