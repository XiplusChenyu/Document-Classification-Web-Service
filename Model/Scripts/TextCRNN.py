import torch.nn as nn
import torch
from Settings import Settings
import torch.nn.functional as F

torch.manual_seed(1)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.embedding = nn.Embedding(Settings.vocab_size, Settings.embedding_dim, padding_idx=1)
        # we set 1 as padding index

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, Settings.num_filters, (k, Settings.embedding_dim)) for k in Settings.filter_sizes])
        self.dropout = nn.Dropout(Settings.dropout)
        self.fc = nn.Linear(Settings.num_filters * len(Settings.filter_sizes), Settings.class_num)
        self.softmax = nn.Softmax(dim=1)

    @staticmethod
    def conv_and_pool(x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.softmax(out)
        return out


if __name__ == '__main__':

    # use these code to test model IO

    from DataLoader import torch_dataset_loader

    TestModel = Model()
    test_loader = torch_dataset_loader(Settings.test_path, Settings.batch_size, False, Settings.kwargs)

    for index, data in enumerate(test_loader):
        word_input, target = data['words'], data['label']
        print(word_input.shape, target.shape)

        TestModel.eval()
        predicted = TestModel(word_input)
        print(predicted)
        break



