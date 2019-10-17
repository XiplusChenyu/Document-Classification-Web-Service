import numpy as np
from torch import nn
from Settings import Settings


def accuracy_function(output, target):
    """
    This function used for calculate predict accuracy
    :param output:
    :param target:
    :return:
    """
    # shape: (batch, labels)
    f_output = output.cpu() if Settings.cuda else output.clone()
    f_target = target.cpu() if Settings.cuda else target.clone()

    output_res = f_output.detach().numpy()
    target_res = f_target.detach().numpy()
    predicted_index = np.argmax(output_res, axis=1)

    target_index = np.argmax(target_res, axis=1)

    # counter
    correct = np.sum(predicted_index == target_index)
    accuracy = correct / (output.shape[0])
    return accuracy


def matrix_tuple(output, target):
    """
    This function used for generate (predicted, label) tuple
    :param output:
    :param target:
    :return:
    """
    f_output = output.cpu() if Settings.cuda else output.clone()
    f_target = target.cpu() if Settings.cuda else target.clone()

    output_res = f_output.detach().numpy()
    target_res = f_target.detach().numpy()
    predicted_index = np.argmax(output_res, axis=1)
    target_index = np.argmax(target_res, axis=1)
    result_list = [[int(predicted_index[i]), int(target_index[i])] for i in range(len(predicted_index))]
    return result_list


def bce_loss(output, target):
    loss_mlp = nn.BCELoss()
    loss = loss_mlp(output, target)
    return loss


# use this block for test
if __name__ == '__main__':
    from DataLoader import test_loader
    from TextCNN import Model

    model = Model()

    for index, data_item in enumerate(test_loader):
        tag = data_item['label']
        model.eval()
        predicted = model(data_item['words'])
        break
