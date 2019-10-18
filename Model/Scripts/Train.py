import time
import numpy as np
import torch
import json
from torch import optim
from Settings import Settings
from TrainUtils import accuracy_function, bce_loss


def train(model, epoch, train_loader, optimizer, versatile=True):
    """
    Use this function to train model, train model for one epoch
    :param model:
    :param epoch: epoch index
    :param train_loader:
    :param optimizer:
    :param versatile: print training stages or not
    :return:
    """
    start_time = time.time()
    model = model.train()
    train_loss = 0.
    accuracy = 0.
    batch_num = len(train_loader)
    _index = 0

    for _index, data in enumerate(train_loader):
        words, target = data['words'], data['label']

        if Settings.cuda:
            words = words.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        predicted = model(words)

        loss_value = bce_loss(predicted, target)
        accuracy_value = accuracy_function(predicted, target)

        loss_value.backward()
        optimizer.step()

        train_loss += loss_value.data.item()
        accuracy += accuracy_value

        if versatile:

            # used for print training process
            if (_index + 1) % Settings.log_step == 0:
                elapsed = time.time() - start_time
                print('Epoch{:3d} | {:3d}/{:3d} batches | {:5.2f}ms/ batch | BCE: {:5.4f} | Accuracy: {:5.2f}% |'
                      .format(epoch, _index + 1, batch_num,
                              elapsed * 1000 / (_index + 1),
                              train_loss / (_index + 1),
                              accuracy * 100 / (_index + 1)))

    train_loss /= (_index + 1)
    accuracy /= (_index + 1)

    print('-' * 99)
    print('End of training epoch {:3d} | time: {:5.2f}s | BCE: {:5.4f} | Accuracy: {:5.2f}% |'
          .format(epoch, (time.time() - start_time),
                  train_loss, accuracy * 100))

    return train_loss, accuracy


def validate_test(model, epoch, use_loader):
    """
    Use this function to evaluate model
    :param model:
    :param epoch: epoch index
    :param use_loader:
    :return:
    """
    start_time = time.time()
    model = model.eval()
    v_loss = 0.
    accuracy = 0.
    data_loader_use = use_loader
    _index = 0

    for _index, data in enumerate(data_loader_use):
        words, target = data['words'], data['label']

        if Settings.cuda:
            words = words.cuda()
            target = target.cuda()

        with torch.no_grad():

            predicted = model(words)

            loss_value = bce_loss(predicted, target)
            accuracy_value = accuracy_function(predicted, target)

            v_loss += loss_value.data.item()
            accuracy += accuracy_value

    v_loss /= (_index + 1)
    accuracy /= (_index + 1)

    print('End of validation epoch {:3d} | time: {:5.2f}s | BCE: {:5.4f} | Accuracy: {:5.2f}% |'
          .format(epoch, (time.time() - start_time),
                  v_loss, accuracy * 100))
    print('-' * 99)

    return v_loss, accuracy


def main_train(model, train_loader, valid_loader, log_name, save_name,
               lr=Settings.learning_rate,
               epoch_num=Settings.epoch_num):

    Settings.dataset_len = len(train_loader)
    Settings.log_step = len(train_loader) // Settings.print_count

    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)
    t_loss, t_accu, v_loss, v_accu = [], [], [], []

    decay_cnt = 0
    for epoch in range(1,  epoch_num + 1):
        if Settings.cuda:
            model.cuda()
        train_loss, train_accuracy = train(model, epoch, train_loader, optimizer)
        validation_loss, validation_accuracy = validate_test(model, epoch, use_loader=valid_loader)

        t_loss.append(train_loss)
        t_accu.append(train_accuracy)

        v_loss.append(validation_loss)
        v_accu.append(validation_accuracy)

        # use accuracy to find the best model
        if np.max(t_accu) == t_accu[-1]:
            print('***Found Best Training Model***')
        if np.max(v_accu) == v_accu[-1]:
            with open(Settings.model_save_folder + save_name, 'wb') as f:
                torch.save(model.cpu().state_dict(), f)
                print('***Best Validation Model Found and Saved***')

        print('-' * 99)

        # Use BCE loss value for learning rate scheduling
        decay_cnt += 1

        if np.min(t_loss) not in t_loss[-3:] and decay_cnt > 2:
            scheduler.step()
            decay_cnt = 0
            print('***Learning rate decreased***')
            print('-' * 99)

    build_dict = {
        "train_loss": t_loss,
        "train_accu": t_accu,
        "valid_loss": v_loss,
        "valid_accu": v_accu,
    }

    with open(Settings.log_save_folder + log_name, 'w+') as lf:
        json.dump(build_dict, lf)

    return build_dict


if __name__ == '__main__':
    from DataLoader import train_loader, validation_loader, test_loader
    from TextCNN import Model

    txtModel = Model()
    main_train(txtModel, train_loader, validation_loader, "test.json", "model.h5")
