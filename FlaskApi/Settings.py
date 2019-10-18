import torch


class Settings:

    # Path
    data_folder = "./static/"

    # Model Paras
    seq_size = 200
    vocab_size = 300997
    embedding_dim = 512
    class_num = 14
    num_filters = 256  # channels
    filter_sizes = (2, 3, 4)
    batch_size = 20
    dropout = 0.6

    # Train Paras
    use_cuda = True
    cuda = torch.cuda.is_available() and use_cuda
    if cuda:
        kwargs = {'num_workers': 1, 'pin_memory': True}
    else:
        kwargs = {}

    learning_rate = 1e-5
    epoch_num = 20
    dataset_len = None
    log_step = None
    print_count = 5  # print 5 count for 1 epoch
