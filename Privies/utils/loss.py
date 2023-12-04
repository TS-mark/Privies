import torch.nn.functional as F
import numpy as np

def rpn_cross_entropy(input, target):
    r"""
    :param input: (15x15x5,2)
    :param target: (15x15x5,)
    :return:
    """
    mask_ignore = target == -1
    mask_calcu = 1 - mask_ignore

    loss = F.cross_entropy(input = input[mask_calcu], target = target[mask_calcu], size_average=False)
    return loss


def rpn_smoothL1(input, target, label):
    r"""
    :param input: torch.Size([1, 1125, 4])
    :param target: torch.Size([1, 1125, 4])
            label: (torch.Size([1, 1125]) pos neg or ignore
    :return:
    """

    pos_index = np.where(label.cpu() == 1)
    loss = F.smooth_l1_loss(input[pos_index], target[pos_index], size_average=False)
    # loss = torch.div(torch.sum(loss), 64)

    return loss