import torch


class Settings:
    data_folder = "../Data/"
    data_path = "../Data/shuffled-full-set-hashed.csv"
    word_dict = "../Data/WordToIndex.json"
    seq_size = 200
    train_path = data_folder + "train.h5"
    valid_path = data_folder + "valid.h5"
    test_path = data_folder + "test.h5"
    batch_size = 50
    use_cuda = True
    cuda = torch.cuda.is_available() and use_cuda
    if cuda:
        kwargs = {'num_workers': 1, 'pin_memory': True}
    else:
        kwargs = {}
