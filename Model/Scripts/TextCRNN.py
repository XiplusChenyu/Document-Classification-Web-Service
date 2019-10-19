import torch.nn as nn
import torch
from Settings import Settings
import torch.nn.functional as F

torch.manual_seed(1)


class CRNNModel(nn.Module):
    def __init__(self):
        super(CRNNModel, self).__init__()

        self.embedding = nn.Embedding(Settings.vocab_size, Settings.embedding_dim, padding_idx=1)
        # we set 1 as padding index
        self.gruLayer = nn.GRU(input_size=Settings.embedding_dim,
                               hidden_size=128,
                               num_layers=1,
                               bidirectional=False)

        self.gruLayerF = nn.Sequential(nn.BatchNorm1d(1024),
                                       nn.Dropout(0.6))

        cov1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov1.weight)
        self.convBlock1 = nn.Sequential(cov1,
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=3))

        cov2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov2.weight)
        self.convBlock2 = nn.Sequential(cov2,
                                        nn.BatchNorm2d(128),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=4))

        cov3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=6, stride=1, padding=1)
        torch.nn.init.xavier_uniform_(cov3.weight)
        self.convBlock3 = nn.Sequential(cov3,
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=5))

        self.fcBlock1 = nn.Sequential(nn.Linear(in_features=512, out_features=256),
                                      nn.ReLU(),
                                      nn.Dropout(0.6))

        self.fcBlock2 = nn.Sequential(nn.Linear(in_features=256, out_features=128),
                                      nn.ReLU(),
                                      nn.Dropout(0.5))

        self.output = nn.Sequential(nn.Linear(in_features=128, out_features=Settings.class_num),
                                    nn.Softmax(dim=1))

    def forward(self, inp):
        # batch word
        inp = self.embedding(inp)
        # batch word freq
        # print(inp.size())
        inp = inp.contiguous().view(inp.size()[1], inp.size()[0], -1)
        # print(inp.size())
        out, _ = self.gruLayer(inp)
        out = out.contiguous().view(out.size()[1], 1,  out.size()[0], out.size()[2])
        # print(out.size())
        out = self.convBlock1(out)
        # print(out.size())
        out = self.convBlock2(out)
        # print(out.size())
        out = self.convBlock3(out)
        # print(out.size())
        out = out.contiguous().view(out.size()[0], -1)
        # print(out.size())
        out = self.fcBlock1(out)
        out = self.fcBlock2(out)
        out = self.output(out)
        # print(out.size())
        return out


if __name__ == '__main__':

    # use these code to test model IO

    from DataLoader import torch_dataset_loader

    TestModel = CRNNModel()
    test_loader = torch_dataset_loader(Settings.test_path, Settings.batch_size, False, Settings.kwargs)

    for index, data in enumerate(test_loader):
        word_input, target = data['words'], data['label']
        print(word_input.shape, target.shape)

        TestModel.eval()
        predicted = TestModel(word_input)
        print(predicted)
        break



